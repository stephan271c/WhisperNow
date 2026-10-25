use dioxus::prelude::*;
use whispernow_domain::ipc::PasteHelperSetup;

use super::copy_button::CopyButton;
use super::ui::Banner;
use crate::notice::Tone;

/// Shown while Wayland paste goes through the `RemoteDesktop` portal because the
/// paste helper is not set up: why the desktop shows its indicator, and how to
/// set the helper up.
#[component]
pub(super) fn PasteHelperNotice(setup: PasteHelperSetup) -> Element {
    rsx! { div { id: "paste-helper-notice",
        Banner { tone: Tone::Warning, title: "Your desktop shows a remote control indicator",
            p { "WhisperNow's paste helper is not set up, so your desktop presses the paste keys for it and shows its remote control indicator while WhisperNow runs." }
            match setup.command {
                Some(command) => rsx! {
                    p { "To paste without it, run this command in a terminal, then restart WhisperNow:" }
                    code { id: "paste-helper-command", class: "command", "{command}" }
                    div { class: "card-actions",
                        CopyButton { id: "paste-helper-copy",
                            text: command, label: "Copy command",
                        }
                    }
                },
                None => rsx! {
                    p { "Install WhisperNow from its .deb or .rpm package, which sets the helper up, then restart WhisperNow." }
                },
            }
        }
    } }
}
