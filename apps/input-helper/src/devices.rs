//! Reads key events from every input device, reports chord releases and
//! presses paste shortcuts.

use std::{
    collections::{HashMap, HashSet},
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
    chord::{Chord, ChordTracker, ChordTransition},
    codes::{EV_KEY, EV_SYN, MAX_KEY_CODE, MODIFIERS, SYN_DROPPED},
    virtual_keyboard::{self, PasteShortcut, VirtualKeyboard},
};

/// `struct input_event` on 64-bit Linux: a `timeval`, then type, code, value.
const EVENT_SIZE: usize = 24;
const _: () = assert!(size_of::<libc::input_event>() == EVENT_SIZE);
const EVDEV_IOCTL_BASE: u32 = b'E' as u32;
const EVIOCSCLOCKID: libc::Ioctl = libc::_IOW::<libc::c_int>(EVDEV_IOCTL_BASE, 0xa0);
const EVIOCGNAME: libc::Ioctl = libc::_IOR::<[u8; 256]>(EVDEV_IOCTL_BASE, 0x06);
const KEY_STATE_BYTES: usize = (MAX_KEY_CODE as usize + 1) / 8;
const EVIOCGKEY: libc::Ioctl = libc::_IOR::<[u8; KEY_STATE_BYTES]>(EVDEV_IOCTL_BASE, 0x18);
/// `EVIOCGBIT(EV_KEY, KEY_STATE_BYTES)`.
const EVIOCGBIT_KEY: libc::Ioctl =
    libc::_IOR::<[u8; KEY_STATE_BYTES]>(EVDEV_IOCTL_BASE, 0x20 + EV_KEY as u32);
const RESCAN_INTERVAL: Duration = Duration::from_secs(2);
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

pub enum Mode {
    Watch(Chord),
    Paste(VirtualKeyboard),
}

pub fn run(mode: Mode) -> ExitCode {
    let (mut tracker, mut keyboard) = match mode {
        Mode::Watch(chord) => (ChordTracker::new(Some(chord)), None),
        Mode::Paste(keyboard) => (ChordTracker::new(None), Some(keyboard)),
    };
    let (sender, messages) = mpsc::channel();
    let stdin_sender = sender.clone();
    thread::spawn(move || read_commands(&stdin_sender));

    let mut devices = HashMap::new();
    let mut next_device = 0;
    scan(&mut devices, &mut next_device, &sender, &mut tracker);
    let mut available =
        keyboard.is_some() || tracker.can_watch(devices.values().flatten().copied());
    if !available {
        eprintln!(
            "no readable input devices support the chord; install this helper setgid `input`"
        );
        return ExitCode::from(2);
    }
    let mut next_scan = Instant::now() + RESCAN_INTERVAL;
    let mut devices_changed = false;
    let mut pending_paste: Option<(PasteShortcut, Instant)> = None;
    let mut stdout = io::stdout().lock();
    if reply_availability(&mut stdout, true, &tracker).is_err() {
        return ExitCode::SUCCESS;
    }
    loop {
        if devices_changed && keyboard.is_none() {
            let ready = tracker.can_watch(devices.values().flatten().copied());
            if ready != available {
                available = ready;
                if reply_availability(&mut stdout, ready, &tracker).is_err() {
                    return ExitCode::SUCCESS;
                }
            }
        }
        devices_changed = false;
        if let (Some(keyboard), Some((shortcut, deadline))) = (&mut keyboard, pending_paste)
            && let Some(answer) = paste_answer(keyboard, &tracker, shortcut, deadline)
        {
            pending_paste = None;
            if reply(&mut stdout, answer).is_err() {
                return ExitCode::SUCCESS;
            }
        }
        let wake = pending_paste.map_or(next_scan, |(_, deadline)| deadline.min(next_scan));
        let message = match messages.recv_timeout(wake.saturating_duration_since(Instant::now())) {
            Ok(message) => message,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if Instant::now() >= next_scan {
                    devices_changed = scan(&mut devices, &mut next_device, &sender, &mut tracker);
                    next_scan = Instant::now() + RESCAN_INTERVAL;
                }
                continue;
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => return ExitCode::SUCCESS,
        };
        let (transition, millis) = match message {
            Message::Key {
                device,
                code,
                value,
                millis,
            } => (tracker.key(device, code, value), millis),
            Message::Lost { device, path } => {
                if let Some(path) = path {
                    devices_changed = devices.remove(&path).is_some();
                }
                (tracker.forget(device), monotonic_millis())
            }
            Message::Command(command) => {
                match PasteShortcut::parse(command.trim()) {
                    Some(shortcut) if keyboard.is_some() && pending_paste.is_none() => {
                        pending_paste = Some((shortcut, Instant::now() + MODIFIER_WAIT));
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
        if let Some(transition) = transition {
            let line = match transition {
                ChordTransition::Pressed => format!("held {millis}"),
                ChordTransition::Released => millis.to_string(),
            };
            if reply(&mut stdout, &line).is_err() {
                return ExitCode::SUCCESS;
            }
        }
    }
}

fn paste_answer(
    keyboard: &mut VirtualKeyboard,
    tracker: &ChordTracker,
    shortcut: PasteShortcut,
    deadline: Instant,
) -> Option<&'static str> {
    if !tracker.holds_any(&MODIFIERS) {
        Some(if keyboard.press(shortcut).is_ok() {
            "sent"
        } else {
            "failed"
        })
    } else if Instant::now() >= deadline {
        Some("held")
    } else {
        None
    }
}

fn read_commands(sender: &mpsc::Sender<Message>) {
    for line in io::stdin().lock().lines().map_while(Result::ok) {
        if sender.send(Message::Command(line)).is_err() {
            return;
        }
    }
    let _ = sender.send(Message::InputClosed);
}

fn reply(stdout: &mut io::StdoutLock<'_>, line: &str) -> io::Result<()> {
    writeln!(stdout, "{line}")?;
    stdout.flush()
}

fn reply_availability(
    stdout: &mut io::StdoutLock<'_>,
    available: bool,
    tracker: &ChordTracker,
) -> io::Result<()> {
    reply(stdout, if available { "ready" } else { "unavailable" })?;
    if available && tracker.chord_held() {
        reply(stdout, &format!("held {}", monotonic_millis()))?;
    }
    Ok(())
}

/// Opens event devices not already being read, each on its own thread.
fn scan(
    devices: &mut HashMap<PathBuf, HashSet<u16>>,
    next_device: &mut u32,
    sender: &mpsc::Sender<Message>,
    tracker: &mut ChordTracker,
) -> bool {
    let Ok(entries) = std::fs::read_dir("/dev/input") else {
        return false;
    };
    let mut changed = false;
    for path in entries.flatten().map(|entry| entry.path()) {
        let is_event = path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("event"));
        if !is_event || devices.contains_key(&path) {
            continue;
        }
        let Some(file) = open_device(&path) else {
            continue;
        };
        let mut supported = [0_u8; KEY_STATE_BYTES];
        // SAFETY: EVIOCGBIT_KEY writes this device's supported key bits into
        // `supported`, whose size is encoded in the request.
        #[allow(
            unsafe_code,
            reason = "evdev key capabilities are only readable by ioctl"
        )]
        if unsafe { libc::ioctl(file.as_raw_fd(), EVIOCGBIT_KEY, supported.as_mut_ptr()) } < 0 {
            continue;
        }
        let mut held = [0_u8; KEY_STATE_BYTES];
        // SAFETY: EVIOCGKEY reads this open device's key state into `held`,
        // whose size is exactly the size encoded in the request.
        #[allow(
            unsafe_code,
            reason = "evdev's initial key state is only readable by ioctl"
        )]
        if unsafe { libc::ioctl(file.as_raw_fd(), EVIOCGKEY, held.as_mut_ptr()) } < 0 {
            continue;
        }
        let device = *next_device;
        *next_device += 1;
        for code in 1..=MAX_KEY_CODE {
            if held[usize::from(code / 8)] & (1 << (code % 8)) != 0 {
                tracker.key(device, code, 1);
            }
        }
        let codes = (1..=MAX_KEY_CODE)
            .filter(|code| supported[usize::from(code / 8)] & (1 << (code % 8)) != 0)
            .collect();
        devices.insert(path.clone(), codes);
        changed = true;
        let sender = sender.clone();
        thread::spawn(move || read_device(file, device, path, &sender));
    }
    changed
}

/// Opens a device whose event times are `CLOCK_MONOTONIC`, the clock portal
/// shortcut signals use. The paste keyboard is skipped.
fn open_device(path: &Path) -> Option<File> {
    let file = File::open(path).ok()?;
    let mut name = [0_u8; 256];
    // SAFETY: EVIOCGNAME(256) writes at most 256 bytes into `name`, which is
    // valid for the call, from an open evdev descriptor owned by `file`.
    #[allow(unsafe_code, reason = "evdev's name is only readable by ioctl")]
    let length = unsafe { libc::ioctl(file.as_raw_fd(), EVIOCGNAME, name.as_mut_ptr()) };
    let length = usize::try_from(length).unwrap_or_default();
    let name = name.get(..length).unwrap_or_default();
    if name.strip_suffix(b"\0").unwrap_or(name) == virtual_keyboard::NAME.as_bytes() {
        return None;
    }
    let clock: libc::c_int = libc::CLOCK_MONOTONIC;
    // SAFETY: EVIOCSCLOCKID reads one `int` through the pointer, which is valid
    // for the call, from an open evdev descriptor owned by `file`.
    #[allow(unsafe_code, reason = "evdev's clock is only selectable by ioctl")]
    let result = unsafe { libc::ioctl(file.as_raw_fd(), EVIOCSCLOCKID, &raw const clock) };
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
        for event in buffer[..length].as_chunks::<EVENT_SIZE>().0 {
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

fn message(device: u32, event: &[u8; EVENT_SIZE]) -> Option<Message> {
    let seconds = i64::from_ne_bytes([
        event[0], event[1], event[2], event[3], event[4], event[5], event[6], event[7],
    ]);
    let micros = i64::from_ne_bytes([
        event[8], event[9], event[10], event[11], event[12], event[13], event[14], event[15],
    ]);
    let kind = u16::from_ne_bytes([event[16], event[17]]);
    let code = u16::from_ne_bytes([event[18], event[19]]);
    let value = i32::from_ne_bytes([event[20], event[21], event[22], event[23]]);
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

    fn event(seconds: i64, micros: i64, kind: u16, code: u16, value: i32) -> [u8; EVENT_SIZE] {
        let mut event = [0; EVENT_SIZE];
        event[..8].copy_from_slice(&seconds.to_ne_bytes());
        event[8..16].copy_from_slice(&micros.to_ne_bytes());
        event[16..18].copy_from_slice(&kind.to_ne_bytes());
        event[18..20].copy_from_slice(&code.to_ne_bytes());
        event[20..].copy_from_slice(&value.to_ne_bytes());
        event
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
