//! Entries carry UTC milliseconds; everything here turns them into the viewer's local day and
//! time. The clock is a context value so tests can pin both the time and the offset from UTC.

use whispernow_domain::{HistoryEntry, HistoryLimit, HistoryRetention, UtcMillis};

const MINUTE_MILLIS: i64 = 60_000;
const DAY_MILLIS: i64 = 86_400_000;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LocalClock {
    now: UtcMillis,
    zone: Zone,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Zone {
    /// A fixed offset, in minutes east of UTC.
    #[cfg(test)]
    Fixed(i64),
    /// The browser's time zone, including its daylight-saving changes.
    System,
}

impl LocalClock {
    #[cfg(test)]
    pub(crate) const fn fixed(now: UtcMillis, offset_minutes: i64) -> Self {
        Self {
            now,
            zone: Zone::Fixed(offset_minutes),
        }
    }

    /// The system's current time and zone. Without a browser the zone is UTC.
    pub(crate) fn system() -> Self {
        Self {
            now: system_now(),
            zone: Zone::System,
        }
    }

    fn offset_millis(self, at: UtcMillis) -> i64 {
        match self.zone {
            #[cfg(test)]
            Zone::Fixed(minutes) => minutes * MINUTE_MILLIS,
            Zone::System => system_offset_minutes(at) * MINUTE_MILLIS,
        }
    }

    /// Milliseconds since the Unix epoch on the local wall clock.
    fn local_millis(self, at: UtcMillis) -> i64 {
        at.0.saturating_add(self.offset_millis(at))
    }

    /// The local calendar day `at` falls on, counted from 1970-01-01.
    pub(crate) fn local_day(self, at: UtcMillis) -> i64 {
        self.local_millis(at).div_euclid(DAY_MILLIS)
    }

    /// The group heading for the local day `day`: Today, Yesterday, a weekday and date this
    /// year, or a full date.
    pub(crate) fn day_label(self, day: i64) -> String {
        let today = self.local_day(self.now);
        if day == today {
            return "Today".to_owned();
        }
        if day == today - 1 {
            return "Yesterday".to_owned();
        }
        let date = CivilDate::from_days(day);
        if date.year == CivilDate::from_days(today).year {
            format!(
                "{}, {} {}",
                weekday_name(day),
                date.day,
                month_name(date.month)
            )
        } else {
            format!("{} {} {}", date.day, month_name(date.month), date.year)
        }
    }

    /// How long ago `at` was: "Just now", minutes within the hour, then the local time.
    pub(crate) fn relative_time(self, at: UtcMillis) -> String {
        let elapsed = self.now.0.saturating_sub(at.0);
        if (0..MINUTE_MILLIS).contains(&elapsed) {
            return "Just now".to_owned();
        }
        if (MINUTE_MILLIS..60 * MINUTE_MILLIS).contains(&elapsed) {
            return format!("{} min ago", elapsed / MINUTE_MILLIS);
        }
        self.clock_time(at)
    }

    /// The local wall-clock time of `at`, as 24-hour `HH:MM`.
    pub(crate) fn clock_time(self, at: UtcMillis) -> String {
        let minutes = self.local_millis(at).rem_euclid(DAY_MILLIS) / MINUTE_MILLIS;
        format!("{:02}:{:02}", minutes / 60, minutes % 60)
    }

    /// The local date and time of `at` in full, for a tooltip.
    pub(crate) fn full_time(self, at: UtcMillis) -> String {
        let day = self.local_day(at);
        let date = CivilDate::from_days(day);
        format!(
            "{}, {} {} {} at {}",
            weekday_name(day),
            date.day,
            month_name(date.month),
            date.year,
            self.clock_time(at)
        )
    }
}

/// `at` as an ISO 8601 UTC instant for a `<time datetime>` attribute.
pub(crate) fn machine_time(at: UtcMillis) -> String {
    let day = at.0.div_euclid(DAY_MILLIS);
    let date = CivilDate::from_days(day);
    let seconds = at.0.rem_euclid(DAY_MILLIS) / 1_000;
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        date.year,
        date.month,
        date.day,
        seconds / 3_600,
        seconds % 3_600 / 60,
        seconds % 60
    )
}

/// Entries, newest first, split into runs that share a local day.
pub(crate) fn group_by_day(
    clock: LocalClock,
    entries: &[HistoryEntry],
) -> Vec<(i64, Vec<HistoryEntry>)> {
    let mut groups: Vec<(i64, Vec<HistoryEntry>)> = Vec::new();
    for entry in entries {
        let day = clock.local_day(entry.created_at);
        match groups.last_mut() {
            Some((group_day, members)) if *group_day == day => members.push(entry.clone()),
            _ => groups.push((day, vec![entry.clone()])),
        }
    }
    groups
}

/// A recording's length as `m:ss`, or `h:mm:ss` from an hour.
pub(crate) fn duration_label(milliseconds: u64) -> String {
    let seconds = milliseconds.saturating_add(500) / 1_000;
    let (hours, minutes, seconds) = (seconds / 3_600, seconds % 3_600 / 60, seconds % 60);
    if hours > 0 {
        format!("{hours}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes}:{seconds:02}")
    }
}

pub(crate) fn retention_summary(
    limit: HistoryLimit,
    retention: HistoryRetention,
    keep_audio: bool,
) -> String {
    let transcripts = limit.max_entries().map_or_else(
        || "every transcript".to_owned(),
        |entries| format!("the last {entries} transcripts"),
    );
    let window = retention
        .days()
        .map_or_else(String::new, |days| format!(" from the last {days} days"));
    let audio = if keep_audio {
        "their audio"
    } else {
        "audio only for failed and cancelled dictations"
    };
    format!("Keeping {transcripts}{window} and {audio}")
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CivilDate {
    year: i64,
    month: i64,
    day: i64,
}

impl CivilDate {
    /// The proleptic Gregorian date `days` after 1970-01-01.
    fn from_days(days: i64) -> Self {
        let shifted = days + 719_468;
        let era = shifted.div_euclid(146_097);
        let day_of_era = shifted - era * 146_097;
        let year_of_era =
            (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
        let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
        let month_piece = (5 * day_of_year + 2) / 153;
        let month = if month_piece < 10 {
            month_piece + 3
        } else {
            month_piece - 9
        };
        Self {
            year: year_of_era + era * 400 + i64::from(month <= 2),
            month,
            day: day_of_year - (153 * month_piece + 2) / 5 + 1,
        }
    }
}

const fn weekday_name(day: i64) -> &'static str {
    // 1970-01-01 was a Thursday.
    match day.rem_euclid(7) {
        0 => "Thursday",
        1 => "Friday",
        2 => "Saturday",
        3 => "Sunday",
        4 => "Monday",
        5 => "Tuesday",
        _ => "Wednesday",
    }
}

const fn month_name(month: i64) -> &'static str {
    match month {
        1 => "January",
        2 => "February",
        3 => "March",
        4 => "April",
        5 => "May",
        6 => "June",
        7 => "July",
        8 => "August",
        9 => "September",
        10 => "October",
        11 => "November",
        _ => "December",
    }
}

#[cfg(target_arch = "wasm32")]
fn system_now() -> UtcMillis {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "Date.now() is a whole number of milliseconds well inside i64"
    )]
    UtcMillis(js_sys::Date::now() as i64)
}

#[cfg(not(target_arch = "wasm32"))]
fn system_now() -> UtcMillis {
    let elapsed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    UtcMillis(i64::try_from(elapsed.as_millis()).unwrap_or(i64::MAX))
}

/// Minutes east of UTC in the browser's zone at `at`.
#[cfg(target_arch = "wasm32")]
fn system_offset_minutes(at: UtcMillis) -> i64 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "history timestamps are far inside the exactly representable range"
    )]
    let date = js_sys::Date::new(&wasm_bindgen::JsValue::from_f64(at.0 as f64));
    #[expect(
        clippy::cast_possible_truncation,
        reason = "time-zone offsets are whole minutes within a day"
    )]
    let west = date.get_timezone_offset() as i64;
    -west
}

#[cfg(not(target_arch = "wasm32"))]
const fn system_offset_minutes(_at: UtcMillis) -> i64 {
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    // 2026-09-27 10:30 UTC, a Sunday.
    const NOW: UtcMillis = UtcMillis(1_790_505_000_000);

    fn at(offset_millis: i64) -> UtcMillis {
        UtcMillis(NOW.0 + offset_millis)
    }

    #[test]
    fn civil_dates_and_weekdays_match_the_calendar() {
        assert_eq!(
            CivilDate::from_days(0),
            CivilDate {
                year: 1970,
                month: 1,
                day: 1
            }
        );
        let clock = LocalClock::fixed(NOW, 0);
        assert_eq!(clock.full_time(NOW), "Sunday, 27 September 2026 at 10:30");
        assert_eq!(machine_time(NOW), "2026-09-27T10:30:00Z");
        assert_eq!(
            CivilDate::from_days(-1),
            CivilDate {
                year: 1969,
                month: 12,
                day: 31
            }
        );
    }

    #[test]
    fn day_labels_name_today_yesterday_this_year_and_earlier_years() {
        let clock = LocalClock::fixed(NOW, 0);
        let today = clock.local_day(NOW);
        assert_eq!(clock.day_label(today), "Today");
        assert_eq!(clock.day_label(today - 1), "Yesterday");
        assert_eq!(clock.day_label(today - 6), "Monday, 21 September");
        assert_eq!(clock.day_label(today - 365), "27 September 2025");
    }

    #[test]
    fn relative_time_counts_minutes_within_the_hour_then_shows_local_time() {
        let clock = LocalClock::fixed(NOW, 120);
        assert_eq!(clock.relative_time(at(-30_000)), "Just now");
        assert_eq!(clock.relative_time(at(-5 * MINUTE_MILLIS)), "5 min ago");
        assert_eq!(clock.relative_time(at(-90 * MINUTE_MILLIS)), "11:00");
        // A timestamp ahead of the clock shows its time rather than a negative age.
        assert_eq!(clock.relative_time(at(MINUTE_MILLIS)), "12:31");
    }

    #[test]
    fn durations_round_to_the_second() {
        assert_eq!(duration_label(0), "0:00");
        assert_eq!(duration_label(4_600), "0:05");
        assert_eq!(duration_label(75_000), "1:15");
        assert_eq!(duration_label(3_725_000), "1:02:05");
    }

    #[test]
    fn retention_summary_names_both_settings() {
        assert_eq!(
            retention_summary(HistoryLimit::Last100, HistoryRetention::Forever, true),
            "Keeping the last 100 transcripts and their audio"
        );
        assert_eq!(
            retention_summary(HistoryLimit::Unlimited, HistoryRetention::Forever, false),
            "Keeping every transcript and audio only for failed and cancelled dictations"
        );
        assert_eq!(
            retention_summary(HistoryLimit::Last20, HistoryRetention::Days30, false),
            "Keeping the last 20 transcripts from the last 30 days and audio only for failed and cancelled dictations"
        );
        assert_eq!(
            retention_summary(HistoryLimit::Unlimited, HistoryRetention::Days7, true),
            "Keeping every transcript from the last 7 days and their audio"
        );
    }
}
