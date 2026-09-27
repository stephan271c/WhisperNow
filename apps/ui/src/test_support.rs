//! Snapshot fixtures and virtual-DOM helpers shared by the UI test modules.

#[cfg(not(target_arch = "wasm32"))]
use std::{any::Any, cell::RefCell, future::Future, pin::Pin, rc::Rc, sync::Once};

#[cfg(not(target_arch = "wasm32"))]
use dioxus::dioxus_core::{AttributeValue, Mutation, Mutations};
#[cfg(not(target_arch = "wasm32"))]
use dioxus::prelude::*;
#[cfg(not(target_arch = "wasm32"))]
use dioxus_html::{
    HtmlEventConverter, MountedResult, RenderedElementBacking, SerializedFormData,
    SerializedHtmlEventConverter, SerializedKeyboardData, SerializedMouseData,
};
use whispernow_api::{EntryId, EntryStatus, HistoryEntry, PhaseDto, UtcMillis};
#[cfg(not(target_arch = "wasm32"))]
use whispernow_api::{
    HistoryLimit, LlmProvider, ModelId, ProviderSettings, RecordingMode, Settings,
};

/// Provides the window context over `state` and `management`, on Home.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn use_test_context(
    state: Signal<crate::UiState>,
    management: Signal<crate::ManagementState>,
) -> crate::UiContext {
    let navigator = crate::Navigator::new(use_signal(|| crate::Route::Home), use_signal(|| 0));
    use_context_provider(|| crate::UiContext {
        state,
        management,
        navigator,
    })
}

/// The window context over `state` and `management`, on Home, for actions run outside a
/// component; the caller must be in a runtime scope.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn test_context(
    state: crate::UiState,
    management: crate::ManagementState,
) -> crate::UiContext {
    crate::UiContext {
        state: Signal::new(state),
        management: Signal::new(management),
        navigator: crate::Navigator::new(Signal::new(crate::Route::Home), Signal::new(0)),
    }
}

/// A toggle recording, the phase of most tests that need one.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) const RECORDING: PhaseDto = PhaseDto::Recording {
    mode: RecordingMode::Toggle,
    session: 1,
};

pub(crate) const PROCESSING: PhaseDto = PhaseDto::Processing { cleaning_up: false };

/// Valid saved settings with a hold-to-record shortcut and AI cleanup off.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn editable_settings() -> Settings {
    Settings {
        model_id: ModelId::new("model").expect("test model id"),
        first_run_complete: true,
        prompts: Vec::new(),
        history_limit: HistoryLimit::Last20,
        llm_provider: LlmProvider::OpenAi,
        llm_provider_settings: [(
            LlmProvider::OpenAi,
            ProviderSettings {
                model: "gpt-5".to_owned(),
                ..ProviderSettings::default()
            },
        )]
        .into(),
        ..Settings::default()
    }
}

/// A pasted dictation `id` saying `text`, made `id` milliseconds after the epoch.
pub(crate) fn history_entry(id: u64, text: &str) -> HistoryEntry {
    HistoryEntry {
        id: EntryId(id),
        created_at: UtcMillis(i64::try_from(id).expect("test timestamp")),
        duration_millis: None,
        raw_text: Some(text.to_owned()),
        cleaned_text: None,
        status: EntryStatus::Ok,
        error: None,
        has_audio: false,
        asr_model_id: None,
        llm_model: None,
    }
}

/// Clicks the element rendered with `target` as its id and renders the result.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn click_rendered(
    dom: &mut VirtualDom,
    rendered: &Mutations,
    target: &str,
) -> Mutations {
    fire_rendered(
        dom,
        rendered,
        target,
        "click",
        PlatformEventData::new(Box::<SerializedMouseData>::default()),
    )
}

/// Fires a form event (`"input"` or `"change"`) carrying `value` at the element rendered with
/// `target` as its id, and renders the result.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn form_event_rendered(
    dom: &mut VirtualDom,
    rendered: &Mutations,
    target: &str,
    name: &'static str,
    value: &str,
) -> Mutations {
    fire_rendered(
        dom,
        rendered,
        target,
        name,
        PlatformEventData::new(Box::new(SerializedFormData::new(
            value.to_owned(),
            Vec::new(),
        ))),
    )
}

/// Presses `key` on the element rendered with `target` as its id and renders the result.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn key_rendered(
    dom: &mut VirtualDom,
    rendered: &Mutations,
    target: &str,
    key: Key,
) -> Mutations {
    fire_rendered(
        dom,
        rendered,
        target,
        "keydown",
        PlatformEventData::new(Box::new(SerializedKeyboardData::new(
            key,
            Code::Unidentified,
            Location::Standard,
            false,
            Modifiers::empty(),
            false,
        ))),
    )
}

/// Mounts the element rendered with `target` as its id, as a browser does once it is in the
/// page, so a later [`focused`] reports when the page moves focus to it.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn mount_rendered(
    dom: &mut VirtualDom,
    rendered: &Mutations,
    target: &str,
) -> Mutations {
    fire_rendered(
        dom,
        rendered,
        target,
        "mounted",
        PlatformEventData::new(Box::new(ProbeElement(target.to_owned()))),
    )
}

#[cfg(not(target_arch = "wasm32"))]
thread_local! {
    static FOCUSED: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// The id of the mounted element that took focus last, which this call clears.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn focused() -> Option<String> {
    FOCUSED.with_borrow_mut(Option::take)
}

/// A mounted element that records when it takes focus.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
struct ProbeElement(String);

#[cfg(not(target_arch = "wasm32"))]
impl RenderedElementBacking for ProbeElement {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn set_focus(&self, focus: bool) -> Pin<Box<dyn Future<Output = MountedResult<()>>>> {
        if focus {
            FOCUSED.with_borrow_mut(|focused| *focused = Some(self.0.clone()));
        }
        Box::pin(async { Ok(()) })
    }
}

/// The serialized converter, except that mounted events carry a [`ProbeElement`].
#[cfg(not(target_arch = "wasm32"))]
struct ProbeConverter;

#[cfg(not(target_arch = "wasm32"))]
macro_rules! serialized {
    ($($method:ident -> $data:ty),* $(,)?) => {
        $(fn $method(&self, event: &PlatformEventData) -> $data {
            SerializedHtmlEventConverter.$method(event)
        })*
    };
}

#[cfg(not(target_arch = "wasm32"))]
impl HtmlEventConverter for ProbeConverter {
    serialized! {
        convert_animation_data -> AnimationData,
        convert_cancel_data -> CancelData,
        convert_clipboard_data -> ClipboardData,
        convert_composition_data -> CompositionData,
        convert_drag_data -> DragData,
        convert_focus_data -> FocusData,
        convert_form_data -> FormData,
        convert_image_data -> ImageData,
        convert_keyboard_data -> KeyboardData,
        convert_media_data -> MediaData,
        convert_mouse_data -> MouseData,
        convert_pointer_data -> PointerData,
        convert_resize_data -> ResizeData,
        convert_scroll_data -> ScrollData,
        convert_selection_data -> SelectionData,
        convert_toggle_data -> ToggleData,
        convert_touch_data -> TouchData,
        convert_transition_data -> TransitionData,
        convert_visible_data -> VisibleData,
        convert_wheel_data -> WheelData,
    }

    fn convert_mounted_data(&self, event: &PlatformEventData) -> MountedData {
        event
            .downcast::<ProbeElement>()
            .cloned()
            .map_or_else(|| MountedData::from(()), MountedData::new)
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn fire_rendered(
    dom: &mut VirtualDom,
    rendered: &Mutations,
    target: &str,
    name: &'static str,
    data: PlatformEventData,
) -> Mutations {
    static EVENT_CONVERTER: Once = Once::new();
    EVENT_CONVERTER.call_once(|| {
        set_event_converter(Box::new(ProbeConverter));
    });
    // The latest element given the id: an earlier one may have been removed and its slot reused.
    let element_id = rendered
        .edits
        .iter()
        .rev()
        .find_map(|mutation| match mutation {
            Mutation::SetAttribute {
                name: "id",
                value: AttributeValue::Text(value),
                id,
                ..
            } if value == target => Some(*id),
            _ => None,
        })
        .unwrap_or_else(|| panic!("rendered element {target}"));
    let event = Event::new(Rc::new(data) as Rc<dyn Any>, true);
    dom.runtime().handle_event(name, event, element_id);
    dom.render_immediate_to_vec()
}
