use std::{any::Any, cell::RefCell, future::Future, pin::Pin, rc::Rc, sync::Once};

pub(crate) use crate::test_fixtures::{history_entry, management_with, settings_view, snapshot};
use dioxus::dioxus_core::{AttributeValue, DynamicNode, Mutation, Mutations, VComponent};
use dioxus::prelude::*;
use dioxus_html::{
    HtmlEventConverter, MountedResult, RenderedElementBacking, SerializedFormData,
    SerializedHtmlEventConverter, SerializedKeyboardData, SerializedMouseData,
};
use whispernow_domain::ipc::EventPayload;
use whispernow_domain::ipc::{AppState, Phase};
use whispernow_domain::{HistoryEntry, ModelId};
use whispernow_domain::{HistoryLimit, HistoryPage, LlmProvider, ProviderSettings, Settings};

use crate::{UiContext, client::fake};

/// Hands a mounted surface's window context to the test after the first render.
#[derive(Clone, Default)]
pub(crate) struct Probe(Rc<RefCell<Option<UiContext>>>);

impl Probe {
    pub(crate) fn provide(
        &self,
        state: Signal<crate::UiState>,
        management: Signal<crate::ManagementState>,
    ) -> UiContext {
        let navigator = crate::Navigator::new(use_signal(|| crate::Route::Home), use_signal(|| 0));
        let context = use_context_provider(|| crate::UiContext {
            state,
            management,
            navigator,
        });
        *self.0.borrow_mut() = Some(context);
        context
    }

    pub(crate) fn context(&self) -> UiContext {
        (*self.0.borrow()).expect("surface rendered")
    }
}

/// A mounted surface and every mutation it has rendered, so a control rendered earlier can
/// still be found.
pub(crate) struct Page {
    pub(crate) dom: VirtualDom,
    rendered: Mutations,
    probe: Probe,
    /// How many desktop calls [`Self::sent`] has already returned.
    seen: usize,
}

impl Page {
    pub(crate) fn host(
        state: crate::UiState,
        management: crate::ManagementState,
        root: fn() -> Element,
    ) -> Self {
        Self::new(|probe| {
            VirtualDom::new_with_props(
                move |()| {
                    let state = use_signal(|| state.clone());
                    let management = use_signal(|| management.clone());
                    probe.provide(state, management);
                    rsx! { {DynamicNode::Component(VComponent::new(root, (), "root"))} }
                },
                (),
            )
        })
    }

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

    fn dispatch(
        &mut self,
        target: (&str, &str),
        name: &'static str,
        data: PlatformEventData,
    ) -> &mut Self {
        let mutations = fire_rendered(&mut self.dom, &self.rendered, target, name, data);
        self.keep(mutations);
        self.settle()
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
        let html = self.html();
        let element = tag(&html, id).unwrap_or_else(|| panic!("rendered element {id}"));
        assert!(
            !element.contains(" disabled="),
            "cannot click disabled target {id}"
        );
        self.dispatch(("id", id), "click", click_event())
    }

    pub(crate) fn click_named(&mut self, name: &str) -> &mut Self {
        self.dispatch(("aria-label", name), "click", click_event())
    }

    /// Opens the select rendered with `id` and clicks its option labelled `label`.
    pub(crate) fn choose(&mut self, id: &str, label: &str) -> &mut Self {
        self.click(id);
        let html = self.html();
        let option = html
            .match_indices(&format!("id=\"{id}-option-"))
            .filter_map(|(at, _)| html[at + "id=\"".len()..].split('"').next())
            .find(|option| text(&html, option) == label)
            .unwrap_or_else(|| panic!("{id} offers {label}"))
            .to_owned();
        self.click(&option)
    }

    pub(crate) fn fire(&mut self, id: &str, name: &'static str, value: &str) -> &mut Self {
        let data = SerializedFormData::new(value.to_owned(), Vec::new());
        self.dispatch(("id", id), name, PlatformEventData::new(Box::new(data)))
    }

    pub(crate) fn input(&mut self, id: &str, value: &str) -> &mut Self {
        self.fire(id, "input", value)
    }

    pub(crate) fn key(&mut self, id: &str, key: Key) -> &mut Self {
        self.key_with_code(id, key, Code::Unidentified, Modifiers::empty())
    }

    pub(crate) fn key_with_code(
        &mut self,
        id: &str,
        key: Key,
        code: Code,
        modifiers: Modifiers,
    ) -> &mut Self {
        let data =
            SerializedKeyboardData::new(key, code, Location::Standard, false, modifiers, false);
        self.dispatch(
            ("id", id),
            "keydown",
            PlatformEventData::new(Box::new(data)),
        )
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

    pub(crate) fn act(&mut self, act: impl FnOnce(UiContext)) -> &mut Self {
        let actions = self.context();
        self.dom.in_scope(ScopeId::ROOT, || act(actions));
        self.settle()
    }

    pub(crate) fn publish(&mut self, event: EventPayload) -> &mut Self {
        self.act(|actions| actions.apply_event(event))
    }
}

pub(crate) fn state_event(phase: Phase, change: impl FnOnce(&mut AppState)) -> EventPayload {
    let mut state = snapshot(phase);
    change(&mut state);
    EventPayload::StateChanged { state }
}

pub(crate) fn tag<'a>(html: &'a str, id: &str) -> Option<&'a str> {
    element(html, &format!("id=\"{id}\""))
}

pub(crate) fn element<'a>(html: &'a str, contains: &str) -> Option<&'a str> {
    let start = html.find(contains)?;
    let open = html[..=start].rfind('<')?;
    let end = start + html[start..].find('>')?;
    Some(&html[open..=end])
}

/// The text inside the element rendered with `id`, without the markup of what it nests.
pub(crate) fn text(html: &str, id: &str) -> String {
    let opening = tag(html, id).unwrap_or_else(|| panic!("element {id}"));
    let start = html.find(opening).expect("opening tag") + opening.len();
    let name = opening[1..]
        .split(|character: char| character.is_whitespace() || character == '>')
        .next()
        .expect("tag name");
    let (open, close) = (format!("<{name}"), format!("</{name}>"));
    let mut depth = 1;
    let mut end = html.len();
    for (index, _) in html[start..].match_indices('<') {
        let rest = &html[start + index..];
        if rest.starts_with(&close) {
            depth -= 1;
            if depth == 0 {
                end = start + index;
                break;
            }
        } else if rest.starts_with(&open)
            && rest[open.len()..].starts_with(|next: char| next.is_whitespace() || next == '>')
        {
            depth += 1;
        }
    }
    strip_tags(&html[start..end])
}

fn strip_tags(html: &str) -> String {
    let mut plain = String::new();
    let mut in_tag = false;
    for character in html.chars() {
        match character {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => plain.push(character),
            _ => {}
        }
    }
    plain
}

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
            strip_tags(&html[start..stop])
        };
        let visible = normalize(&visible);
        if !visible.is_empty() && !normalize(name).contains(&visible) {
            violations.push(format!("{name:?} hides {visible:?} in {opening}"));
        }
    }
    violations
}

pub(crate) const RECORDING: Phase = Phase::Recording {
    trigger: whispernow_domain::ipc::RecordingTrigger::Toggle,
    session: 1,
};

pub(crate) const PROCESSING: Phase = Phase::Processing { cleaning_up: false };

pub(crate) fn show_history(management: &mut crate::ManagementState, entries: Vec<HistoryEntry>) {
    let (request_id, _) = management.history_request();
    management.finish_history_refresh(
        request_id,
        HistoryPage {
            entries,
            has_more: false,
        },
    );
}

/// Valid saved settings with a hold-to-record shortcut and AI cleanup off.
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

pub(crate) fn click_rendered(
    dom: &mut VirtualDom,
    rendered: &Mutations,
    target: &str,
) -> Mutations {
    fire_rendered(dom, rendered, ("id", target), "click", click_event())
}

fn click_event() -> PlatformEventData {
    PlatformEventData::new(Box::<SerializedMouseData>::default())
}

/// Mounts the element rendered with `target` as its id, as a browser does once it is in the
/// page, so a later [`focused`] reports when the page moves focus to it.
pub(crate) fn mount_rendered(
    dom: &mut VirtualDom,
    rendered: &Mutations,
    target: &str,
) -> Mutations {
    fire_rendered(
        dom,
        rendered,
        ("id", target),
        "mounted",
        PlatformEventData::new(Box::new(ProbeElement(target.to_owned()))),
    )
}

thread_local! {
    static FOCUSED: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// The id of the mounted element that took focus last, which this call clears.
pub(crate) fn focused() -> Option<String> {
    FOCUSED.with_borrow_mut(Option::take)
}

/// A mounted element that records when it takes focus.
#[derive(Clone)]
struct ProbeElement(String);

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
struct ProbeConverter;

macro_rules! serialized {
    ($($method:ident -> $data:ty),* $(,)?) => {
        $(fn $method(&self, event: &PlatformEventData) -> $data {
            SerializedHtmlEventConverter.$method(event)
        })*
    };
}

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

fn fire_rendered(
    dom: &mut VirtualDom,
    rendered: &Mutations,
    target: (&str, &str),
    name: &'static str,
    data: PlatformEventData,
) -> Mutations {
    static EVENT_CONVERTER: Once = Once::new();
    EVENT_CONVERTER.call_once(|| {
        set_event_converter(Box::new(ProbeConverter));
    });
    // Use the latest matching attribute: an earlier element may have been removed and its slot reused.
    let element_id = rendered
        .edits
        .iter()
        .rev()
        .find_map(|mutation| match mutation {
            Mutation::SetAttribute {
                name,
                value: AttributeValue::Text(value),
                id,
                ..
            } if *name == target.0 && value == target.1 => Some(*id),
            _ => None,
        })
        .unwrap_or_else(|| panic!("rendered element {target:?}"));
    let event = Event::new(Rc::new(data) as Rc<dyn Any>, true);
    dom.runtime().handle_event(name, event, element_id);
    dom.render_immediate_to_vec()
}
