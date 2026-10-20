//! The X11, Windows and macOS clipboard through `arboard`.
//!
//! Only text, HTML with its plain-text alternative, and images are captured and
//! restored. `arboard` reports a clipboard holding only other formats (file
//! lists, RTF, application-private formats) exactly as it reports an empty
//! one, so neither is ever cleared. Every value written is marked for
//! exclusion from clipboard history, cloud sync and monitors where the
//! platform has such a marker: the transcript is transient, and a restored
//! value is already in history.

use std::{borrow::Cow, fmt, mem};

#[cfg(target_os = "macos")]
use arboard::SetExtApple as _;
#[cfg(target_os = "linux")]
use arboard::SetExtLinux as _;
#[cfg(target_os = "windows")]
use arboard::SetExtWindows as _;
use arboard::{Clipboard, Error, ImageData, Set};
use zeroize::{Zeroize, Zeroizing};

use super::NativePasteError;

/// The previous clipboard, held only for one paste transaction and wiped on drop.
pub enum ClipboardContents {
    Text(String),
    Html {
        html: String,
        alt_text: Option<String>,
    },
    Image(ImageData<'static>),
}

impl Drop for ClipboardContents {
    fn drop(&mut self) {
        match self {
            Self::Text(text) => text.zeroize(),
            Self::Html { html, alt_text } => {
                html.zeroize();
                alt_text.zeroize();
            }
            Self::Image(image) => {
                if let Cow::Owned(bytes) = &mut image.bytes {
                    bytes.zeroize();
                }
            }
        }
    }
}

impl fmt::Debug for ClipboardContents {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Text(_) => "ClipboardContents::Text(<redacted>)",
            Self::Html { .. } => "ClipboardContents::Html(<redacted>)",
            Self::Image(_) => "ClipboardContents::Image(<redacted>)",
        })
    }
}

/// One long-lived `arboard` handle. On X11 it must outlive every value it
/// writes, because the handle's owner thread serves the selection.
pub struct NativeClipboard(Clipboard);

impl NativeClipboard {
    pub fn connect() -> Result<Self, NativePasteError> {
        Clipboard::new().map(Self).map_err(clipboard_error)
    }

    /// Captures the one representation `capture_order` selects. `Ok(None)`
    /// means none of text, HTML or an image is present: the clipboard may be
    /// empty or hold only formats `arboard` cannot read.
    pub fn snapshot(&mut self) -> Result<Option<ClipboardContents>, NativePasteError> {
        // Windows reports a missing HTML format as a generic error, so any HTML
        // read failure just means "no HTML".
        let mut html = self.0.get().html().ok().map(Zeroizing::new);
        let mut text = if html.is_some() {
            self.0.get().text().ok()
        } else {
            present(self.0.get().text())?
        }
        .map(Zeroizing::new);
        for format in capture_order(
            html.as_deref().map(String::as_str),
            text.as_deref().map(String::as_str),
        ) {
            match format {
                CaptureFormat::Html => {
                    if let Some(html) = &mut html {
                        return Ok(Some(ClipboardContents::Html {
                            html: mem::take(&mut **html),
                            alt_text: text.as_mut().map(|text| mem::take(&mut **text)),
                        }));
                    }
                }
                CaptureFormat::Text => {
                    if let Some(text) = &mut text {
                        return Ok(Some(ClipboardContents::Text(mem::take(&mut **text))));
                    }
                }
                CaptureFormat::Image => match self.0.get().image() {
                    Ok(image) => return Ok(Some(ClipboardContents::Image(image))),
                    Err(Error::ContentNotAvailable) => {}
                    // HTML is still there to restore when its image is unreadable.
                    Err(error) if html.is_some() => {
                        tracing::debug!(%error, "clipboard image could not be read; keeping its HTML");
                    }
                    Err(error) => return Err(clipboard_error(error)),
                },
            }
        }
        Ok(None)
    }

    pub fn write_text(&mut self, text: &str) -> Result<(), NativePasteError> {
        excluded(self.0.set()).text(text).map_err(clipboard_error)
    }

    pub fn holds_text(&mut self, text: &str) -> Result<bool, NativePasteError> {
        Ok(present(self.0.get().text())?.is_some_and(|mut current| {
            let holds = current == text;
            current.zeroize();
            holds
        }))
    }

    pub fn restore(&mut self, contents: &ClipboardContents) -> Result<(), NativePasteError> {
        let set = excluded(self.0.set());
        match contents {
            ClipboardContents::Text(text) => set.text(text.as_str()),
            ClipboardContents::Html { html, alt_text } => {
                set.html(html.as_str(), alt_text.as_deref())
            }
            ClipboardContents::Image(image) => set.image(ImageData {
                width: image.width,
                height: image.height,
                bytes: Cow::Borrowed(&image.bytes),
            }),
        }
        .map_err(clipboard_error)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CaptureFormat {
    Html,
    Text,
    Image,
}

/// The representations to try capturing, in order, given the HTML and plain
/// text present. `arboard` puts back HTML with its plain-text alternative,
/// text, or an image, never HTML and an image together. HTML wins when it
/// restores readable text; when it has no plain-text alternative or no visible
/// text (a browser's "Copy image" offers an `<img>` wrapper beside the
/// bitmap), an image is what the user copied and HTML is only the fallback.
fn capture_order(html: Option<&str>, text: Option<&str>) -> &'static [CaptureFormat] {
    match (html, text) {
        (Some(html), Some(text)) if !text.trim().is_empty() && has_visible_text(html) => {
            &[CaptureFormat::Html]
        }
        (Some(_), _) => &[CaptureFormat::Image, CaptureFormat::Html],
        (None, Some(_)) => &[CaptureFormat::Text],
        (None, None) => &[CaptureFormat::Image],
    }
}

/// Whether markup shows any text outside its tags.
fn has_visible_text(html: &str) -> bool {
    let mut in_tag = false;
    html.chars().any(|character| match character {
        '<' => {
            in_tag = true;
            false
        }
        '>' if in_tag => {
            in_tag = false;
            false
        }
        _ => !in_tag && !character.is_whitespace(),
    })
}

fn excluded(set: Set<'_>) -> Set<'_> {
    #[cfg(target_os = "windows")]
    {
        set.exclude_from_monitoring()
            .exclude_from_cloud()
            .exclude_from_history()
    }
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        set.exclude_from_history()
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        set
    }
}

fn present<T>(read: Result<T, Error>) -> Result<Option<T>, NativePasteError> {
    match read {
        Ok(value) => Ok(Some(value)),
        Err(Error::ContentNotAvailable) => Ok(None),
        Err(error) => Err(clipboard_error(error)),
    }
}

#[allow(
    clippy::needless_pass_by_value,
    reason = "Result::map_err supplies the owned arboard error"
)]
fn clipboard_error(error: Error) -> NativePasteError {
    tracing::debug!(%error, "native clipboard operation failed");
    NativePasteError::Unavailable
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_copied_image_wins_over_html_that_restores_no_text() {
        let browser_image =
            r#"<meta charset="utf-8"><img src="https://example.test/cat.png" alt="cat">"#;
        let windows_fragment = "<html><body>\r\n<!--StartFragment--><img src=\"cat.png\"><!--EndFragment-->\r\n</body></html>";
        for (html, text) in [
            (browser_image, None),
            (browser_image, Some("https://example.test/cat.png")),
            (windows_fragment, Some("")),
            ("<b>rich</b>", None),
            ("<b>rich</b>", Some("  ")),
        ] {
            assert_eq!(
                capture_order(Some(html), text),
                [CaptureFormat::Image, CaptureFormat::Html],
                "{html:?} with {text:?}"
            );
        }
    }

    #[test]
    fn html_with_readable_text_is_restored_with_its_alternative() {
        assert_eq!(
            capture_order(Some("<p>a &amp; <b>b</b></p>"), Some("a & b")),
            [CaptureFormat::Html]
        );
        assert_eq!(
            capture_order(Some("<p><img src=\"x.png\">caption</p>"), Some("caption")),
            [CaptureFormat::Html]
        );
    }

    #[test]
    fn without_html_text_wins_and_an_image_is_the_last_resort() {
        assert_eq!(capture_order(None, Some("plain")), [CaptureFormat::Text]);
        assert_eq!(capture_order(None, None), [CaptureFormat::Image]);
    }
}
