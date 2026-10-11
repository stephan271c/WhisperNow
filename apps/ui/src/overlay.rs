//! The desktop decides the phase and the window's visibility; this model only adds what
//! arrives between snapshots: the smoothed level history and the elapsed time.

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
pub struct OverlayView {
    /// Newest last; each value is in `0.0..=1.0`.
    pub levels: [f32; LEVEL_BARS],
    /// Whether real samples arrived for this recording; until then the bars are arming.
    pub levels_seen: bool,
    pub elapsed_ms: u64,
    /// The recording the meter belongs to.
    session: Option<u64>,
    /// Why the last overlay action failed; cleared by the next phase.
    pub action_error: Option<String>,
}

impl OverlayView {
    pub fn start_recording(&mut self, session: u64) {
        if self.session != Some(session) {
            self.session = Some(session);
            self.levels = [0.0; LEVEL_BARS];
            self.levels_seen = false;
            self.elapsed_ms = 0;
        }
    }

    /// Adds one throttled meter reading while the microphone may be live.
    pub fn apply_level(&mut self, rms: f32, elapsed_ms: u64) {
        let decibels = 20.0 * rms.clamp(1e-6, 1.0).log10();
        let activity = ((decibels - FLOOR_DB) / (CEILING_DB - FLOOR_DB)).clamp(0.0, 1.0);
        let previous = self.levels[LEVEL_BARS - 1];
        let weight = if activity > previous { ATTACK } else { RELEASE };
        self.levels.rotate_left(1);
        self.levels[LEVEL_BARS - 1] = previous + (activity - previous) * weight;
        self.levels_seen = true;
        self.elapsed_ms = elapsed_ms;
    }
}

/// `m:ss` for the elapsed recording time.
#[must_use]
pub(crate) fn format_elapsed(elapsed_ms: u64) -> String {
    let seconds = elapsed_ms / 1_000;
    format!("{}:{:02}", seconds / 60, seconds % 60)
}
