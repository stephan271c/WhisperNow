use std::{collections::BTreeMap, fmt, time::Duration};

use zeroize::Zeroizing;

use super::portal::WaylandPortalError;

pub(crate) const TRANSCRIPT_MIME_TYPES: [&str; 2] = ["text/plain;charset=utf-8", "text/plain"];
pub(crate) const SNAPSHOT_MIME_TYPES: [&str; 6] = [
    "text/plain;charset=utf-8",
    "text/plain",
    "text/html",
    "text/rtf",
    "image/png",
    "text/uri-list",
];
pub(crate) const READ_TIMEOUT: Duration = Duration::from_millis(500);
pub(crate) const MAX_FORMAT_BYTES: usize = 8 * 1024 * 1024;
const MAX_TOTAL_BYTES: usize = 16 * 1024 * 1024;

pub(crate) type SecretBytes = Zeroizing<Vec<u8>>;

/// Bounded common-MIME snapshot held only for one output transaction.
pub(crate) struct PortalClipboardSnapshot {
    formats: BTreeMap<String, SecretBytes>,
}

impl PortalClipboardSnapshot {
    /// `read_selection` already bounds each format to [`MAX_FORMAT_BYTES`];
    /// this bounds their total.
    pub(crate) fn bounded(formats: Vec<(String, SecretBytes)>) -> Result<Self, WaylandPortalError> {
        let total = formats.iter().fold(0usize, |total, (_, bytes)| {
            total.saturating_add(bytes.len())
        });
        if total > MAX_TOTAL_BYTES {
            return Err(WaylandPortalError::ClipboardSnapshotTooLarge);
        }
        Ok(Self {
            formats: formats.into_iter().collect(),
        })
    }

    pub(crate) fn as_offer(&self) -> Vec<(&str, &[u8])> {
        self.formats
            .iter()
            .map(|(mime, bytes)| (mime.as_str(), bytes.as_slice()))
            .collect()
    }
}

impl fmt::Debug for PortalClipboardSnapshot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PortalClipboardSnapshot")
            .field("format_count", &self.formats.len())
            .finish()
    }
}

/// What the session offers on the clipboard and whether it still owns it.
/// Once another owner has taken the clipboard only a new offer makes it ours
/// again, so copying identical visible text still prevents restoration.
#[derive(Default)]
pub(crate) struct PortalClipboardState {
    offer: BTreeMap<String, SecretBytes>,
    owns_offer: bool,
}

impl PortalClipboardState {
    pub(crate) fn replace_offer(&mut self, offer: Vec<(&str, &[u8])>) {
        self.offer = offer
            .into_iter()
            .map(|(mime, bytes)| (mime.to_owned(), Zeroizing::new(bytes.to_vec())))
            .collect();
        self.owns_offer = true;
    }

    pub(crate) fn owner_changed(&mut self, session_is_owner: Option<bool>) {
        if session_is_owner != Some(true) {
            self.owns_offer = false;
        }
    }

    pub(crate) fn transfer(&self, mime_type: &str) -> Option<SecretBytes> {
        self.offer.get(mime_type).cloned()
    }

    pub(crate) const fn can_restore(&self) -> bool {
        self.owns_offer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_snapshot_is_bounded_in_total_and_redacted() {
        let snapshot = PortalClipboardSnapshot::bounded(vec![
            (
                "text/plain;charset=utf-8".to_owned(),
                Zeroizing::new(b"secret".to_vec()),
            ),
            (
                "text/html".to_owned(),
                Zeroizing::new(b"<b>secret</b>".to_vec()),
            ),
        ])
        .expect("snapshot");
        assert_eq!(snapshot.as_offer().len(), 2);
        assert!(!format!("{snapshot:?}").contains("secret"));
        let half = || Zeroizing::new(vec![0; MAX_TOTAL_BYTES / 2 + 1]);
        assert_eq!(
            PortalClipboardSnapshot::bounded(vec![
                ("text/plain".to_owned(), half()),
                ("text/html".to_owned(), half()),
            ])
            .expect_err("oversized snapshot"),
            WaylandPortalError::ClipboardSnapshotTooLarge
        );
    }

    #[test]
    fn replacement_with_same_visible_text_still_blocks_restore() {
        let mut state = PortalClipboardState::default();
        state.replace_offer(vec![("text/plain", b"same")]);
        assert!(state.can_restore());
        state.owner_changed(Some(false));
        state.owner_changed(Some(true));
        assert!(!state.can_restore());
    }

    #[test]
    fn a_session_that_has_offered_nothing_cannot_restore() {
        let mut state = PortalClipboardState::default();
        assert!(!state.can_restore());
        state.owner_changed(Some(true));
        assert!(!state.can_restore());
    }

    #[test]
    fn transfers_serve_only_the_offered_mime_types() {
        let mut state = PortalClipboardState::default();
        state.replace_offer(vec![("text/plain", b"transcript")]);
        assert_eq!(
            state.transfer("text/plain").as_deref().map(Vec::as_slice),
            Some(b"transcript".as_slice())
        );
        assert_eq!(state.transfer("text/html"), None);
    }
}
