use std::time::{Duration, Instant};

/// Which download chunks publish progress: one per `INTERVAL`, and one whenever the whole
/// percent changes.
#[derive(Default)]
pub(crate) struct ProgressThrottle {
    published_at: Option<Instant>,
    percent: Option<u64>,
}

impl ProgressThrottle {
    const INTERVAL: Duration = Duration::from_millis(100);

    pub(crate) fn poll_due(&mut self, now: Instant, downloaded: u64, total: Option<u64>) -> bool {
        let percent = total
            .filter(|total| *total > 0)
            .map(|total| downloaded.saturating_mul(100) / total);
        let waited = self
            .published_at
            .is_none_or(|at| now.saturating_duration_since(at) >= Self::INTERVAL);
        if !waited && percent == self.percent {
            return false;
        }
        self.published_at = Some(now);
        self.percent = percent;
        true
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::ProgressThrottle;

    #[test]
    fn progress_is_published_every_hundred_milliseconds_and_at_each_whole_percent() {
        let start = Instant::now();
        let at = |millis| start + Duration::from_millis(millis);
        let mut throttle = ProgressThrottle::default();
        assert!(throttle.poll_due(at(0), 1, Some(1_000)), "the first chunk");
        assert!(!throttle.poll_due(at(10), 5, Some(1_000)), "too soon");
        assert!(
            throttle.poll_due(at(20), 10, Some(1_000)),
            "a whole percent"
        );
        assert!(!throttle.poll_due(at(119), 15, Some(1_000)), "99 ms later");
        assert!(throttle.poll_due(at(120), 16, Some(1_000)), "100 ms later");

        let mut unknown = ProgressThrottle::default();
        assert!(unknown.poll_due(at(0), 1, None));
        assert!(
            !unknown.poll_due(at(99), 1_000_000, None),
            "no total, no percent"
        );
        assert!(unknown.poll_due(at(100), 1_000_001, None));
        assert!(
            !unknown.poll_due(at(150), 1_000_002, Some(0)),
            "an empty total is no percent"
        );
    }

    #[test]
    fn many_tiny_chunks_publish_a_bounded_number_of_states() {
        let start = Instant::now();
        let (chunk, total) = (256_u64, 1024 * 1024_u64);
        let mut throttle = ProgressThrottle::default();
        let published = (1..=total / chunk)
            .filter(|index| {
                let now = start + Duration::from_micros(index * 100);
                throttle.poll_due(now, index * chunk, Some(total))
            })
            .count();
        // 4096 chunks in 410 ms: the first, 100 whole percents, and one per 100 ms between them.
        assert!((101..=105).contains(&published), "{published}");
    }
}
