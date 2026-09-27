//! An entry's kept audio, loaded and played on request.

use dioxus::prelude::*;
use whispernow_api::EntryId;

use crate::components::ui::{Button, ButtonSize, ButtonVariant, Icon, IconName};

/// Plays an entry's kept audio, fetched when asked for so the list never loads audio it won't
/// play.
#[component]
pub(super) fn AudioPlayer(id: EntryId) -> Element {
    let mut source = use_signal(|| AudioSource::Idle);
    use_drop(move || release(&source.peek()));
    rsx! { div { class: "history-audio", id: "history-audio-{id.0}",
        match source() {
            AudioSource::Ready(url) => rsx! {
                audio { controls: true, autoplay: true, preload: "auto", src: "{url}",
                    aria_label: "Recording audio",
                }
            },
            AudioSource::Loading => rsx! {
                p { role: "status", "Loading audio…" }
            },
            AudioSource::Failed(summary) => rsx! {
                p { role: "alert", "{summary}" }
            },
            AudioSource::Idle => rsx! {
                Button { id: "history-play-{id.0}", size: ButtonSize::Sm,
                    variant: ButtonVariant::Secondary,
                    onclick: move |_| {
                        source.set(AudioSource::Loading);
                        load_audio(id, source);
                    },
                    Icon { name: IconName::Play }
                    "Play"
                }
            },
        }
    } }
}

#[derive(Clone, Debug, PartialEq)]
enum AudioSource {
    Idle,
    Loading,
    /// An object URL owned by this player.
    Ready(String),
    Failed(String),
}

/// Fetches the audio and shows it in the player once it arrives. A player closed meanwhile
/// drops this task before the audio has a URL, so no URL outlives the player unrevoked.
fn load_audio(id: EntryId, mut source: Signal<AudioSource>) {
    spawn(async move {
        source.set(match audio_url(id).await {
            Ok(url) => AudioSource::Ready(url),
            Err(summary) => AudioSource::Failed(summary),
        });
    });
}

/// An object URL for the entry's audio, or why it could not be loaded. The URL is made
/// after the last await, in the same step that hands it to the player.
#[cfg(target_arch = "wasm32")]
async fn audio_url(id: EntryId) -> Result<String, String> {
    let request = whispernow_api::GetHistoryAudioRequest { id };
    let audio = crate::client::history_audio(&request)
        .await
        .map_err(|error| {
            error.api_error().map_or_else(
                || "The audio could not be loaded".to_owned(),
                |error| error.summary.clone(),
            )
        })?;
    Ok(audio.object_url())
}

/// Only the desktop app holds the audio.
#[cfg(not(target_arch = "wasm32"))]
fn audio_url(_id: EntryId) -> std::future::Ready<Result<String, String>> {
    std::future::ready(Err("Audio plays in the desktop app".to_owned()))
}

#[cfg(target_arch = "wasm32")]
fn release(source: &AudioSource) {
    if let AudioSource::Ready(url) = source {
        crate::client::revoke_object_url(url);
    }
}

#[cfg(not(target_arch = "wasm32"))]
const fn release(_source: &AudioSource) {}
