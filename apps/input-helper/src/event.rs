//! `struct input_event`, which evdev devices produce and uinput consumes.

use crate::codes::{EV_KEY, EV_SYN, SYN_REPORT};

/// The `value` of an `EV_KEY` event.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KeyValue {
    Release = 0,
    Press = 1,
    Repeat = 2,
}

impl KeyValue {
    pub const fn from_raw(value: i32) -> Option<Self> {
        match value {
            0 => Some(Self::Release),
            1 => Some(Self::Press),
            2 => Some(Self::Repeat),
            _ => None,
        }
    }
}

#[derive(Clone, Copy)]
pub struct InputEvent(libc::input_event);

const _: () = assert!(size_of::<libc::input_event>() == InputEvent::SIZE);

impl InputEvent {
    /// On 64-bit Linux: a `timeval`, then type, code, value.
    pub const SIZE: usize = 24;

    /// Without a time: uinput stamps events as it receives them.
    pub const fn new(kind: u16, code: u16, value: i32) -> Self {
        Self(libc::input_event {
            time: libc::timeval {
                tv_sec: 0,
                tv_usec: 0,
            },
            type_: kind,
            code,
            value,
        })
    }

    pub const fn key(code: u16, value: KeyValue) -> Self {
        Self::new(EV_KEY, code, value as i32)
    }

    pub const fn report() -> Self {
        Self::new(EV_SYN, SYN_REPORT, 0)
    }

    #[cfg(test)]
    pub const fn at(mut self, seconds: libc::time_t, micros: libc::suseconds_t) -> Self {
        self.0.time = libc::timeval {
            tv_sec: seconds,
            tv_usec: micros,
        };
        self
    }

    pub fn from_bytes(bytes: &[u8; Self::SIZE]) -> Self {
        // SAFETY: `input_event` is plain integers, valid for every bit pattern,
        // and `bytes` holds one; `read_unaligned` takes any alignment.
        #[allow(unsafe_code, reason = "the kernel's event layout is libc's struct")]
        Self(unsafe { std::ptr::read_unaligned(bytes.as_ptr().cast()) })
    }

    pub fn to_bytes(self) -> [u8; Self::SIZE] {
        // SAFETY: `input_event` has no padding, so every byte is initialized;
        // `transmute` checks that the sizes match.
        #[allow(unsafe_code, reason = "the kernel's event layout is libc's struct")]
        unsafe {
            std::mem::transmute(self.0)
        }
    }

    pub const fn kind(self) -> u16 {
        self.0.type_
    }

    pub const fn code(self) -> u16 {
        self.0.code
    }

    pub const fn value(self) -> i32 {
        self.0.value
    }

    /// The device's `CLOCK_MONOTONIC` time, which `open_device` selects.
    pub fn millis(self) -> Option<u64> {
        u64::try_from(self.0.time.tv_sec * 1000 + self.0.time.tv_usec / 1000).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_event_is_a_timeval_then_type_code_and_value() {
        let event = InputEvent::new(EV_KEY, 57, -2).at(12, 345_678);

        let expected = [
            &12_i64.to_ne_bytes()[..],
            &345_678_i64.to_ne_bytes(),
            &EV_KEY.to_ne_bytes(),
            &57_u16.to_ne_bytes(),
            &(-2_i32).to_ne_bytes(),
        ]
        .concat();
        assert_eq!(event.to_bytes().as_slice(), expected);
        let decoded = InputEvent::from_bytes(&event.to_bytes());
        assert_eq!(
            (decoded.kind(), decoded.code(), decoded.value()),
            (EV_KEY, 57, -2)
        );
        assert_eq!(decoded.millis(), Some(12_345));
    }

    #[test]
    fn key_values_are_the_numbers_evdev_reports() {
        for value in [KeyValue::Release, KeyValue::Press, KeyValue::Repeat] {
            assert_eq!(KeyValue::from_raw(value as i32), Some(value));
        }
        assert_eq!(KeyValue::from_raw(3), None);
        assert_eq!(KeyValue::from_raw(-1), None);
    }
}
