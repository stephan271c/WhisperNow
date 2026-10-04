//! Reads key events from every input device, reports chord releases and
//! presses paste shortcuts.

use std::{
    collections::HashSet,
    fs::File,
    io::{self, BufRead as _, Read, Write},
    os::fd::AsRawFd,
    path::{Path, PathBuf},
    process::ExitCode,
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

use crate::{
    chord::ChordTracker,
    virtual_keyboard::{self, PasteShortcut, VirtualKeyboard},
};

/// `struct input_event` on 64-bit Linux: a `timeval`, then type, code, value.
const EVENT_SIZE: usize = 24;
const EV_SYN: u16 = 0;
const EV_KEY: u16 = 1;
const SYN_DROPPED: u16 = 3;
/// `EVIOCSCLOCKID`, `_IOW('E', 0xa0, int)`.
const EVIOCSCLOCKID: u64 = 0x4004_45a0;
/// `EVIOCGNAME(256)`, `_IOC(_IOC_READ, 'E', 0x06, 256)`.
const EVIOCGNAME: u64 = 0x8100_4506;
const RESCAN_INTERVAL: Duration = Duration::from_secs(2);
/// Shift, Ctrl, Alt and Super, left and right. A paste waits until none is
/// physically held, because a held one would combine with the shortcut.
const MODIFIERS: [u16; 8] = [29, 97, 42, 54, 56, 100, 125, 126];
/// How long a paste waits for held modifiers before it is refused.
const MODIFIER_WAIT: Duration = Duration::from_secs(1);

enum Message {
    Key {
        device: u32,
        code: u16,
        value: i32,
        millis: u64,
    },
    /// The device overflowed or went away; its held keys are unknown.
    Lost {
        device: u32,
        path: Option<PathBuf>,
    },
    /// A line of standard input.
    Command(String),
    InputClosed,
}

/// Watches the tracker's chords and, with a keyboard, answers each paste
/// command with `sent` or, when modifiers stay held, `held`.
pub fn run(mut tracker: ChordTracker, mut keyboard: Option<VirtualKeyboard>) -> ExitCode {
    let (sender, messages) = mpsc::channel();
    let stdin_sender = sender.clone();
    thread::spawn(move || {
        for line in io::stdin().lock().lines().map_while(Result::ok) {
            if stdin_sender.send(Message::Command(line)).is_err() {
                return;
            }
        }
        let _ = stdin_sender.send(Message::InputClosed);
    });

    let mut open = HashSet::new();
    let mut next_device = 0;
    scan(&mut open, &mut next_device, &sender);
    if open.is_empty() && keyboard.is_none() {
        eprintln!("no input device is readable; install this helper setgid `input`");
        return ExitCode::from(2);
    }
    let mut next_scan = Instant::now() + RESCAN_INTERVAL;
    let mut pending: Option<(PasteShortcut, Instant)> = None;
    let mut stdout = io::stdout().lock();
    if keyboard.is_some() && reply(&mut stdout, "ready").is_err() {
        return ExitCode::SUCCESS;
    }
    loop {
        if let (Some(keyboard), Some((shortcut, deadline))) = (&mut keyboard, pending) {
            let answer = if !tracker.holds_any(&MODIFIERS) {
                Some(if keyboard.press(shortcut).is_ok() {
                    "sent"
                } else {
                    "failed"
                })
            } else if Instant::now() >= deadline {
                Some("held")
            } else {
                None
            };
            if let Some(answer) = answer {
                pending = None;
                if reply(&mut stdout, answer).is_err() {
                    return ExitCode::SUCCESS;
                }
            }
        }
        let wake = pending.map_or(next_scan, |(_, deadline)| deadline.min(next_scan));
        let message = match messages.recv_timeout(wake.saturating_duration_since(Instant::now())) {
            Ok(message) => message,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if Instant::now() >= next_scan {
                    scan(&mut open, &mut next_device, &sender);
                    next_scan = Instant::now() + RESCAN_INTERVAL;
                }
                continue;
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => return ExitCode::SUCCESS,
        };
        let (released, millis) = match message {
            Message::Key {
                device,
                code,
                value,
                millis,
            } => (tracker.key(device, code, value), millis),
            Message::Lost { device, path } => {
                if let Some(path) = path {
                    open.remove(&path);
                }
                (tracker.forget(device), monotonic_millis())
            }
            Message::Command(command) => {
                match PasteShortcut::parse(command.trim()) {
                    Some(shortcut) if keyboard.is_some() && pending.is_none() => {
                        pending = Some((shortcut, Instant::now() + MODIFIER_WAIT));
                    }
                    _ => {
                        if reply(&mut stdout, "refused").is_err() {
                            return ExitCode::SUCCESS;
                        }
                    }
                }
                continue;
            }
            Message::InputClosed => return ExitCode::SUCCESS,
        };
        for id in released {
            if writeln!(stdout, "{id:x} {millis}")
                .and_then(|()| stdout.flush())
                .is_err()
            {
                return ExitCode::SUCCESS;
            }
        }
    }
}

fn reply(stdout: &mut io::StdoutLock<'_>, line: &str) -> io::Result<()> {
    writeln!(stdout, "{line}")?;
    stdout.flush()
}

/// Opens event devices not already being read, each on its own thread.
fn scan(open: &mut HashSet<PathBuf>, next_device: &mut u32, sender: &mpsc::Sender<Message>) {
    let Ok(entries) = std::fs::read_dir("/dev/input") else {
        return;
    };
    for path in entries.flatten().map(|entry| entry.path()) {
        let is_event = path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("event"));
        if !is_event || open.contains(&path) {
            continue;
        }
        let Some(file) = open_device(&path) else {
            continue;
        };
        let device = *next_device;
        *next_device += 1;
        open.insert(path.clone());
        let sender = sender.clone();
        thread::spawn(move || read_device(file, device, path, &sender));
    }
}

/// Opens a device whose event times are `CLOCK_MONOTONIC`, the clock portal
/// shortcut signals use. The paste keyboard is skipped.
fn open_device(path: &Path) -> Option<File> {
    let file = File::open(path).ok()?;
    let mut name = [0_u8; 256];
    // SAFETY: EVIOCGNAME(256) writes at most 256 bytes into `name`, which is
    // valid for the call, from an open evdev descriptor owned by `file`.
    #[allow(unsafe_code, reason = "evdev's name is only readable by ioctl")]
    let length = unsafe { libc::ioctl(file.as_raw_fd(), EVIOCGNAME as _, name.as_mut_ptr()) };
    let length = usize::try_from(length).unwrap_or_default();
    let name = name.get(..length).unwrap_or_default();
    if name.strip_suffix(b"\0").unwrap_or(name) == virtual_keyboard::NAME.as_bytes() {
        return None;
    }
    let clock: libc::c_int = libc::CLOCK_MONOTONIC;
    // SAFETY: EVIOCSCLOCKID reads one `int` through the pointer, which is valid
    // for the call, from an open evdev descriptor owned by `file`.
    #[allow(unsafe_code, reason = "evdev's clock is only selectable by ioctl")]
    let result = unsafe { libc::ioctl(file.as_raw_fd(), EVIOCSCLOCKID as _, &raw const clock) };
    (result == 0).then_some(file)
}

fn read_device(mut file: File, device: u32, path: PathBuf, sender: &mpsc::Sender<Message>) {
    let mut buffer = [0_u8; EVENT_SIZE * 64];
    loop {
        let length = match file.read(&mut buffer) {
            Ok(length) if length > 0 => length,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            _ => break,
        };
        for event in buffer[..length].chunks_exact(EVENT_SIZE) {
            let Some(message) = message(device, event) else {
                continue;
            };
            if sender.send(message).is_err() {
                return;
            }
        }
    }
    let _ = sender.send(Message::Lost {
        device,
        path: Some(path),
    });
}

fn message(device: u32, event: &[u8]) -> Option<Message> {
    let field = |range: std::ops::Range<usize>| event.get(range).unwrap_or_default();
    let seconds = i64::from_ne_bytes(field(0..8).try_into().ok()?);
    let micros = i64::from_ne_bytes(field(8..16).try_into().ok()?);
    let kind = u16::from_ne_bytes(field(16..18).try_into().ok()?);
    let code = u16::from_ne_bytes(field(18..20).try_into().ok()?);
    let value = i32::from_ne_bytes(field(20..24).try_into().ok()?);
    match (kind, code) {
        (EV_KEY, _) => Some(Message::Key {
            device,
            code,
            value,
            millis: u64::try_from(seconds * 1000 + micros / 1000).ok()?,
        }),
        (EV_SYN, SYN_DROPPED) => Some(Message::Lost { device, path: None }),
        _ => None,
    }
}

fn monotonic_millis() -> u64 {
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `now` is a valid, writable timespec for the call.
    #[allow(unsafe_code, reason = "std does not expose CLOCK_MONOTONIC values")]
    unsafe {
        libc::clock_gettime(libc::CLOCK_MONOTONIC, &raw mut now);
    }
    u64::try_from(now.tv_sec * 1000 + now.tv_nsec / 1_000_000).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(seconds: i64, micros: i64, kind: u16, code: u16, value: i32) -> Vec<u8> {
        [
            &seconds.to_ne_bytes()[..],
            &micros.to_ne_bytes(),
            &kind.to_ne_bytes(),
            &code.to_ne_bytes(),
            &value.to_ne_bytes(),
        ]
        .concat()
    }

    #[test]
    fn key_events_carry_their_millisecond_time() {
        let Some(Message::Key {
            device,
            code,
            value,
            millis,
        }) = message(3, &event(12, 345_678, EV_KEY, 57, 0))
        else {
            panic!("expected a key event");
        };
        assert_eq!((device, code, value, millis), (3, 57, 0, 12_345));
    }

    #[test]
    fn overflow_loses_the_device_state_and_other_events_are_ignored() {
        assert!(matches!(
            message(1, &event(0, 0, EV_SYN, SYN_DROPPED, 0)),
            Some(Message::Lost {
                device: 1,
                path: None
            })
        ));
        assert!(message(1, &event(0, 0, EV_SYN, 0, 0)).is_none());
        assert!(message(1, &event(0, 0, 2, 0, 5)).is_none());
    }
}
