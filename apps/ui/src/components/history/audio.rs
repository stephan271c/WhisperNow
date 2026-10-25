use dioxus::prelude::*;
use whispernow_domain::EntryId;

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
    Ready(String),
    Failed(String),
}

/// Fetches the audio and shows it in the player once it arrives. A player closed meanwhile
/// drops this task before the audio has a URL, so no URL outlives the player unrevoked.
fn load_audio(id: EntryId, mut source: Signal<AudioSource>) {
    spawn(async move {
        source.set(match crate::client::history_audio_url(id).await {
            Ok(url) => AudioSource::Ready(url),
            Err(error) => AudioSource::Failed(error.summary),
        });
    });
}

fn release(source: &AudioSource) {
    if let AudioSource::Ready(url) = source {
        crate::client::revoke_object_url(url);
    }
}
