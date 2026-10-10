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
use whispernow_domain::ipc::EventPayload;
use whispernow_domain::ipc::{AppState, ModelState, Phase};
use whispernow_domain::{EntryId, EntryStatus, HistoryEntry, ModelId, UtcMillis};
#[cfg(not(target_arch = "wasm32"))]
use whispernow_domain::{
    HistoryFilter, HistoryLimit, HistoryPage, LlmProvider, ProviderSettings, Settings,
};

#[cfg(not(target_arch = "wasm32"))]
use crate::{UiContext, client::fake, components::UiActions};

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

/// For actions run outside a component; the caller must be in a runtime scope.
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

/// Hands a mounted surface's window context to the test after the first render.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Default)]
pub(crate) struct Probe(Rc<RefCell<Option<UiContext>>>);

#[cfg(not(target_arch = "wasm32"))]
impl PartialEq for Probe {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Probe {
    pub(crate) fn provide(
        &self,
        state: Signal<crate::UiState>,
        management: Signal<crate::ManagementState>,
    ) -> UiContext {
        let context = use_test_context(state, management);
        *self.0.borrow_mut() = Some(context);
        context
    }

    pub(crate) fn context(&self) -> UiContext {
        (*self.0.borrow()).expect("surface rendered")
    }
}

/// A mounted surface and every mutation it has rendered, so a control rendered earlier can
/// still be found.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) struct Page {
    pub(crate) dom: VirtualDom,
    rendered: Mutations,
    probe: Probe,
    /// How many desktop calls [`Self::sent`] has already returned.
    seen: usize,
}

#[cfg(not(target_arch = "wasm32"))]
impl Page {
    pub(crate) fn new(build: impl FnOnce(Probe) -> VirtualDom) -> Self {
        fake::reset();
        let probe = Probe::default();
        let mut dom = build(probe.clone());
        let rendered = dom.rebuild_to_vec();
        Self {
            dom,
            rendered,
            probe,
            seen: 0,
        }
    }

    pub(crate) fn html(&self) -> String {
        dioxus_ssr::render(&self.dom)
    }

    pub(crate) fn context(&self) -> UiContext {
        self.probe.context()
    }

    fn keep(&mut self, mutations: Mutations) {
        self.rendered.edits.extend(mutations.edits);
    }

    pub(crate) fn settle(&mut self) -> &mut Self {
        for _ in 0..4 {
            self.dom.process_events();
            let mutations = self.dom.render_immediate_to_vec();
            self.keep(mutations);
        }
        self
    }

    pub(crate) fn click(&mut self, id: &str) -> &mut Self {
        if tag(&self.html(), id).is_some_and(|element| element.contains(" disabled=")) {
            return self;
        }
        let mutations = click_rendered(&mut self.dom, &self.rendered, id);
        self.keep(mutations);
        self.settle()
    }

    pub(crate) fn fire(&mut self, id: &str, name: &'static str, value: &str) -> &mut Self {
        let mutations = form_event_rendered(&mut self.dom, &self.rendered, id, name, value);
        self.keep(mutations);
        self.settle()
    }

    pub(crate) fn input(&mut self, id: &str, value: &str) -> &mut Self {
        self.fire(id, "input", value)
    }

    pub(crate) fn key(&mut self, id: &str, key: Key) -> &mut Self {
        let mutations = key_rendered(&mut self.dom, &self.rendered, id, key);
        self.keep(mutations);
        self.settle()
    }

    /// Mounts `id` as the browser does, so taking focus is recorded.
    pub(crate) fn mount(&mut self, id: &str) -> &mut Self {
        let mutations = mount_rendered(&mut self.dom, &self.rendered, id);
        self.keep(mutations);
        self
    }

    pub(crate) fn wait(&mut self) -> &mut Self {
        fake::run_timers();
        self.settle()
    }

    pub(crate) fn answer(
        &mut self,
        command: &str,
        answer: Result<impl serde::Serialize, whispernow_domain::AppError>,
    ) -> &mut Self {
        fake::answer(command, answer);
        self.settle()
    }

    /// The commands called since the last look.
    pub(crate) fn sent(&mut self) -> Vec<&'static str> {
        let commands = fake::commands();
        let sent = commands[self.seen..].to_vec();
        self.seen = commands.len();
        sent
    }

    pub(crate) fn act(&mut self, act: impl FnOnce(UiActions)) -> &mut Self {
        let actions = UiActions::new(self.context());
        self.dom.in_scope(ScopeId::ROOT, || act(actions));
        self.settle()
    }

    pub(crate) fn publish(&mut self, event: EventPayload) -> &mut Self {
        self.act(|actions| actions.apply_event(event))
    }
}

/// A snapshot in `phase` with the tiny Whisper model ready.
pub(crate) fn snapshot(phase: Phase) -> AppState {
    AppState {
        phase,
        selected_model_id: ModelId::new("sherpa-onnx-whisper-tiny").expect("fixture model id"),
        model_state: ModelState::Ready,
        last_transcript: None,
        error: None,
        active_downloads: Vec::new(),
        setup_test: None,
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn tag<'a>(html: &'a str, id: &str) -> Option<&'a str> {
    let start = html.find(&format!("id=\"{id}\""))?;
    let open = html[..start].rfind('<')?;
    let end = start + html[start..].find('>')?;
    Some(&html[open..=end])
}

/// The text inside the element rendered with `id`, without the markup of what it nests.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn text(html: &str, id: &str) -> String {
    let opening = tag(html, id).unwrap_or_else(|| panic!("element {id}"));
    let start = html.find(opening).expect("opening tag") + opening.len();
    let name = opening[1..]
        .split(|character: char| character.is_whitespace() || character == '>')
        .next()
        .expect("tag name");
    let (open, close) = (format!("<{name}"), format!("</{name}>"));
    let mut depth = 1;
    let mut plain = String::new();
    let mut in_tag = false;
    for (index, character) in html[start..].char_indices() {
        let rest = &html[start + index..];
        if rest.starts_with(&close) {
            depth -= 1;
            if depth == 0 {
                break;
            }
        } else if rest.starts_with(&open)
            && rest[open.len()..].starts_with(|next: char| next.is_whitespace() || next == '>')
        {
            depth += 1;
        }
        match character {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => plain.push(character),
            _ => {}
        }
    }
    plain
}

#[cfg(not(target_arch = "wasm32"))]
/// The accessible names of interactive elements that leave out the words they show.
pub(crate) fn label_in_name_violations(html: &str) -> Vec<String> {
    const INTERACTIVE: [&str; 5] = ["button", "a", "input", "textarea", "select"];
    let normalize = |value: &str| {
        value
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase()
    };
    let mut violations = Vec::new();
    let mut offset = 0;
    while let Some(found) = html[offset..].find("aria-label=\"") {
        let at = offset + found;
        offset = at + 1;
        let open = html[..at].rfind('<').expect("tag start");
        let end = at + html[at..].find('>').expect("tag end");
        let opening = &html[open..=end];
        let name_start = at + "aria-label=\"".len();
        let name = &html[name_start..name_start + html[name_start..].find('"').expect("quote")];
        let tag_name = opening[1..]
            .split(|character: char| character.is_whitespace() || character == '>')
            .next()
            .expect("tag name");
        if !INTERACTIVE.contains(&tag_name) {
            continue;
        }
        let visible = if matches!(tag_name, "input" | "textarea" | "select") {
            // A form control shows the text of the label that points at it.
            let Some(id) = opening
                .split("id=\"")
                .nth(1)
                .and_then(|rest| rest.split('"').next())
            else {
                continue;
            };
            let Some(label) = html.find(&format!("for=\"{id}\"")) else {
                continue;
            };
            let start = label + html[label..].find('>').expect("label end") + 1;
            let stop = start + html[start..].find("</label>").expect("label close");
            html[start..stop].to_owned()
        } else {
            let start = end + 1;
            let stop = start
                + html[start..]
                    .find(&format!("</{tag_name}>"))
                    .expect("closing tag");
            let mut plain = String::new();
            let mut in_tag = false;
            for character in html[start..stop].chars() {
                match character {
                    '<' => in_tag = true,
                    '>' => in_tag = false,
                    _ if !in_tag => plain.push(character),
                    _ => {}
                }
            }
            plain
        };
        let visible = normalize(&visible);
        if !visible.is_empty() && !normalize(name).contains(&visible) {
            violations.push(format!("{name:?} hides {visible:?} in {opening}"));
        }
    }
    violations
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) const RECORDING: Phase = Phase::Recording {
    trigger: whispernow_domain::ipc::RecordingTrigger::Toggle,
    session: 1,
};

pub(crate) const PROCESSING: Phase = Phase::Processing { cleaning_up: false };

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn show_history(management: &mut crate::ManagementState, entries: Vec<HistoryEntry>) {
    management.finish_history_refresh(
        1,
        &HistoryFilter::default(),
        HistoryPage {
            entries,
            next_cursor: None,
        },
    );
}

/// Valid saved settings with a hold-to-record shortcut and AI cleanup off.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn editable_settings() -> Settings {
    Settings {
        model_id: ModelId::new("model").expect("test model id"),
        first_run_complete: true,
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
pub(crate) fn history_entry(id: i64, text: &str) -> HistoryEntry {
    HistoryEntry {
        id: EntryId(id),
        created_at: UtcMillis(id),
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
