//! The desktop decides the phase and the window's visibility; the meter only adds what
//! arrives between snapshots: the smoothed level history and the elapsed time.

use whispernow_domain::ipc::AudioLevel;

pub(crate) const LEVEL_BARS: usize = 9;

/// Microphone RMS, in dBFS, shown as an empty and as a full bar. Normal speech fills about
/// half; the floor sits above the room tone between words, so the bars drop in each pause.
const FLOOR_DB: f32 = -45.0;
const CEILING_DB: f32 = -5.0;

/// Weight of a louder and of a quieter reading: bars jump with each syllable and fall back
/// within about 100 ms, so words read as separate peaks.
const ATTACK: f32 = 0.8;
const RELEASE: f32 = 0.6;

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Meter {
    pub(crate) levels: [f32; LEVEL_BARS],
    pub(crate) elapsed_ms: u64,
    /// The recording the levels were heard in.
    session: Option<u64>,
}

impl Meter {
    /// Adds a level heard during `session`; the first level of a new session starts from
    /// empty bars.
    pub(crate) fn apply_level(&mut self, session: u64, level: AudioLevel) {
        if self.session != Some(session) {
            *self = Self {
                session: Some(session),
                ..Self::default()
            };
        }
        let decibels = 20.0 * level.rms.clamp(1e-6, 1.0).log10();
        let activity = ((decibels - FLOOR_DB) / (CEILING_DB - FLOOR_DB)).clamp(0.0, 1.0);
        let previous = self.levels[LEVEL_BARS - 1];
        let weight = if activity > previous { ATTACK } else { RELEASE };
        self.levels.rotate_left(1);
        self.levels[LEVEL_BARS - 1] = previous + (activity - previous) * weight;
        self.elapsed_ms = level.elapsed_ms;
    }

    /// Whether samples arrived for `session`; until then its bars are arming.
    pub(crate) fn heard(&self, session: u64) -> bool {
        self.session == Some(session)
    }
}

#[must_use]
pub(crate) fn format_elapsed(elapsed_ms: u64) -> String {
    let seconds = elapsed_ms / 1_000;
    format!("{}:{:02}", seconds / 60, seconds % 60)
}
