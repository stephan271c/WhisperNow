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
    event::{InputEvent, KeyValue},
    virtual_keyboard::{self, PasteShortcut, VirtualKeyboard},
};

const EVDEV_IOCTL_BASE: u32 = b'E' as u32;
const EVIOCSCLOCKID: libc::Ioctl = libc::_IOW::<libc::c_int>(EVDEV_IOCTL_BASE, 0xa0);
const NAME_BYTES: usize = 256;
const KEY_STATE_BYTES: usize = (MAX_KEY_CODE as usize + 1) / 8;
/// Request numbers for [`eviocg`].
const EVIOCGNAME: u32 = 0x06;
const EVIOCGKEY: u32 = 0x18;
const EVIOCGBIT_KEY: u32 = 0x20 + EV_KEY as u32;
const RESCAN_INTERVAL: Duration = Duration::from_secs(2);
const MODIFIER_WAIT: Duration = Duration::from_secs(1);

enum Message {
    Key {
        device: u32,
        code: u16,
        value: KeyValue,
        millis: u64,
    },
    /// Events were dropped, so the device's held keys are unknown.
    Overflowed {
        device: u32,
    },
    Gone {
        device: u32,
        path: PathBuf,
    },
    Command(String),
    InputClosed,
}

pub enum Mode {
    Watch(Chord),
    Paste(VirtualKeyboard),
}

enum Session {
    Watch,
    Paste {
        keyboard: VirtualKeyboard,
        pending: Option<(PasteShortcut, Instant)>,
    },
}

impl Session {
    fn available(&self, tracker: &ChordTracker, devices: &Devices) -> bool {
        match self {
            Self::Watch => tracker.can_watch(devices.supported_keys()),
            Self::Paste { .. } => true,
        }
    }

    /// Whether the command starts a paste.
    fn accept(&mut self, command: &str) -> bool {
        let Self::Paste { pending, .. } = self else {
            return false;
        };
        let Some(shortcut) = PasteShortcut::parse(command.trim()) else {
            return false;
        };
        if pending.is_some() {
            return false;
        }
        *pending = Some((shortcut, Instant::now() + MODIFIER_WAIT));
        true
    }

    fn deadline(&self) -> Option<Instant> {
        match self {
            Self::Paste {
                pending: Some((_, deadline)),
                ..
            } => Some(*deadline),
            _ => None,
        }
    }

    /// Presses a pending paste once no modifier is held, or gives up at its deadline.
    fn paste_answer(&mut self, tracker: &ChordTracker) -> Option<&'static str> {
        let Self::Paste { keyboard, pending } = self else {
            return None;
        };
        let (shortcut, deadline) = (*pending)?;
        let answer = if !tracker.holds_any(&MODIFIERS) {
            if keyboard.press(shortcut).is_ok() {
                "sent"
            } else {
                "failed"
            }
        } else if Instant::now() >= deadline {
            "held"
        } else {
            return None;
        };
        *pending = None;
        Some(answer)
    }
}

pub fn run(mode: Mode) -> ExitCode {
    // A reply that cannot be written means WhisperNow closed the pipe and is gone.
    serve(mode).unwrap_or(ExitCode::SUCCESS)
}

fn serve(mode: Mode) -> io::Result<ExitCode> {
    let (mut tracker, mut session) = match mode {
        Mode::Watch(chord) => (ChordTracker::new(Some(chord)), Session::Watch),
        Mode::Paste(keyboard) => (
            ChordTracker::new(None),
            Session::Paste {
                keyboard,
                pending: None,
            },
        ),
    };
    let (sender, messages) = mpsc::channel();
    let stdin_sender = sender.clone();
    thread::spawn(move || read_commands(&stdin_sender));

    let mut devices = Devices::new(sender);
    devices.scan(&mut tracker);
    let mut available = session.available(&tracker, &devices);
    if !available {
        eprintln!(
            "no readable input devices support the chord; install this helper setgid `input`"
        );
        return Ok(ExitCode::from(2));
    }
    let mut next_scan = Instant::now() + RESCAN_INTERVAL;
    let mut devices_changed = false;
    let mut stdout = io::stdout().lock();
    reply_availability(&mut stdout, true, &tracker)?;
    loop {
        if devices_changed {
            let ready = session.available(&tracker, &devices);
            if ready != available {
                available = ready;
                reply_availability(&mut stdout, ready, &tracker)?;
            }
        }
        devices_changed = false;
        if let Some(answer) = session.paste_answer(&tracker) {
            reply(&mut stdout, answer)?;
        }
        let wake = session
            .deadline()
            .map_or(next_scan, |deadline| deadline.min(next_scan));
        let message = match messages.recv_timeout(wake.saturating_duration_since(Instant::now())) {
            Ok(message) => message,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if Instant::now() >= next_scan {
                    devices_changed = devices.scan(&mut tracker);
                    next_scan = Instant::now() + RESCAN_INTERVAL;
                }
                continue;
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => return Ok(ExitCode::SUCCESS),
        };
        let (transition, millis) = match message {
            Message::Key {
                device,
                code,
                value,
                millis,
            } => (tracker.key(device, code, value), millis),
            Message::Overflowed { device } => (tracker.forget(device), monotonic_millis()),
            Message::Gone { device, path } => {
                devices_changed = devices.remove(&path);
                (tracker.forget(device), monotonic_millis())
            }
            Message::Command(command) => {
                if !session.accept(&command) {
                    reply(&mut stdout, "refused")?;
                }
                continue;
            }
            Message::InputClosed => return Ok(ExitCode::SUCCESS),
        };
        if let Some(transition) = transition {
            let line = match transition {
                ChordTransition::Pressed => format!("held {millis}"),
                ChordTransition::Released => millis.to_string(),
            };
            reply(&mut stdout, &line)?;
        }
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

struct Devices {
    /// The key codes each device supports.
    supported: HashMap<PathBuf, HashSet<u16>>,
    next_id: u32,
    sender: mpsc::Sender<Message>,
}

impl Devices {
    fn new(sender: mpsc::Sender<Message>) -> Self {
        Self {
            supported: HashMap::new(),
            next_id: 0,
            sender,
        }
    }

    fn supported_keys(&self) -> impl Iterator<Item = u16> {
        self.supported.values().flatten().copied()
    }

    fn remove(&mut self, path: &Path) -> bool {
        self.supported.remove(path).is_some()
    }

    /// Starts reading every new device, and returns whether there was one.
    fn scan(&mut self, tracker: &mut ChordTracker) -> bool {
        let Ok(entries) = std::fs::read_dir("/dev/input") else {
            return false;
        };
        let mut changed = false;
        for path in entries.flatten().map(|entry| entry.path()) {
            let is_event = path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("event"));
            if !is_event || self.supported.contains_key(&path) {
                continue;
            }
            let Some(file) = open_device(&path) else {
                continue;
            };
            let Ok(supported) = eviocg::<KEY_STATE_BYTES>(&file, EVIOCGBIT_KEY) else {
                continue;
            };
            let Ok(held) = eviocg::<KEY_STATE_BYTES>(&file, EVIOCGKEY) else {
                continue;
            };
            let device = self.next_id;
            self.next_id += 1;
            for code in 1..=MAX_KEY_CODE {
                if held[usize::from(code / 8)] & (1 << (code % 8)) != 0 {
                    tracker.key(device, code, KeyValue::Press);
                }
            }
            let codes = (1..=MAX_KEY_CODE)
                .filter(|code| supported[usize::from(code / 8)] & (1 << (code % 8)) != 0)
                .collect();
            self.supported.insert(path.clone(), codes);
            changed = true;
            let sender = self.sender.clone();
            thread::spawn(move || read_device(file, device, path, &sender));
        }
        changed
    }
}

/// Reads the `N` bytes an `EVIOCG*` request with number `number` writes.
fn eviocg<const N: usize>(file: &File, number: u32) -> io::Result<[u8; N]> {
    let mut bytes = [0_u8; N];
    // SAFETY: the request encodes `N` as the size the kernel may write, `bytes`
    // holds `N`, and the descriptor is an open evdev device owned by `file`.
    #[allow(unsafe_code, reason = "evdev's device state is only readable by ioctl")]
    let result = unsafe {
        libc::ioctl(
            file.as_raw_fd(),
            libc::_IOR::<[u8; N]>(EVDEV_IOCTL_BASE, number),
            bytes.as_mut_ptr(),
        )
    };
    if result < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(bytes)
    }
}

/// Opens a device whose event times are `CLOCK_MONOTONIC`, the clock portal
/// shortcut signals use. The paste keyboard is skipped.
fn open_device(path: &Path) -> Option<File> {
    let file = File::open(path).ok()?;
    let is_paste_keyboard = eviocg::<NAME_BYTES>(&file, EVIOCGNAME).is_ok_and(|name| {
        name.split(|&byte| byte == 0).next() == Some(virtual_keyboard::NAME.as_bytes())
    });
    if is_paste_keyboard {
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
    let mut buffer = [0_u8; InputEvent::SIZE * 64];
    loop {
        let length = match file.read(&mut buffer) {
            Ok(length) if length > 0 => length,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            _ => break,
        };
        for event in buffer[..length].as_chunks::<{ InputEvent::SIZE }>().0 {
            let Some(message) = message(device, InputEvent::from_bytes(event)) else {
                continue;
            };
            if sender.send(message).is_err() {
                return;
            }
        }
    }
    let _ = sender.send(Message::Gone { device, path });
}

fn message(device: u32, event: InputEvent) -> Option<Message> {
    match (event.kind(), event.code()) {
        (EV_KEY, code) => Some(Message::Key {
            device,
            code,
            value: KeyValue::from_raw(event.value())?,
            millis: event.millis()?,
        }),
        (EV_SYN, SYN_DROPPED) => Some(Message::Overflowed { device }),
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

    #[test]
    fn key_events_carry_their_millisecond_time() {
        let event = InputEvent::key(57, KeyValue::Release).at(12, 345_678);

        let Some(Message::Key {
            device,
            code,
            value,
            millis,
        }) = message(3, event)
        else {
            panic!("expected a key event");
        };
        assert_eq!(
            (device, code, value, millis),
            (3, 57, KeyValue::Release, 12_345)
        );
    }

    #[test]
    fn overflow_loses_the_device_state_and_other_events_are_ignored() {
        assert!(matches!(
            message(1, InputEvent::new(EV_SYN, SYN_DROPPED, 0)),
            Some(Message::Overflowed { device: 1 })
        ));
        assert!(message(1, InputEvent::report()).is_none());
        assert!(message(1, InputEvent::new(2, 0, 5)).is_none());
        assert!(message(1, InputEvent::new(EV_KEY, 57, 3)).is_none());
    }
}
