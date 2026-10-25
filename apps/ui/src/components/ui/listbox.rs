//! The open option list of a `Select` or `Combobox`: where it sits beside its control and how it
//! scrolls to a row.

use std::collections::HashMap;
use std::rc::Rc;

use dioxus::html::geometry::{PixelsRect, PixelsVector2D};
use dioxus::prelude::*;

use super::{Icon, IconName};
use crate::client;

/// Pixels an open list keeps clear of the window edge, its 4px gap to the trigger included.
const WINDOW_MARGIN: f64 = 12.0;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Side {
    Below,
    Above,
}

/// Where an open list sits, and the height it fits in when the window was measured.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Placement {
    side: Side,
    room: Option<f64>,
}

impl Placement {
    const UNMEASURED: Self = Self {
        side: Side::Below,
        room: None,
    };

    /// Below the trigger, unless the list only fits above or there is more room above.
    fn choose(trigger: PixelsRect, height: f64, viewport: f64) -> Self {
        let below = viewport - trigger.max_y() - WINDOW_MARGIN;
        let above = trigger.min_y() - WINDOW_MARGIN;
        let (side, room) = if height <= below || below >= above {
            (Side::Below, below)
        } else {
            (Side::Above, above)
        };
        Self {
            side,
            room: Some(room.max(0.0)),
        }
    }

    /// The list's `data-placement`; the stylesheet hides the list until it is set.
    const fn side_name(self) -> &'static str {
        match self.side {
            Side::Below => "below",
            Side::Above => "above",
        }
    }

    /// Applied as a CSSOM property, never a `style` attribute, which the CSP forbids. 18rem
    /// repeats the stylesheet's own limit.
    fn max_height(self) -> Option<String> {
        self.room.map(|room| format!("min(18rem, {room:.0}px)"))
    }
}

async fn place(trigger: Option<Rc<MountedData>>, list: Rc<MountedData>) -> Placement {
    let (Some(trigger), Some(viewport)) = (trigger, client::viewport_height()) else {
        return Placement::UNMEASURED;
    };
    match (
        trigger.get_client_rect().await,
        list.get_client_rect().await,
    ) {
        (Ok(anchor), Ok(rect)) => Placement::choose(anchor, rect.height(), viewport),
        _ => Placement::UNMEASURED,
    }
}

/// Scrolls the list just enough to show `row`. `scrollIntoView` would scroll the page too.
async fn reveal(list: Rc<MountedData>, row: Rc<MountedData>) {
    let (Ok(frame), Ok(row), Ok(offset)) = (
        list.get_client_rect().await,
        row.get_client_rect().await,
        list.get_scroll_offset().await,
    ) else {
        return;
    };
    let shift = if row.min_y() < frame.min_y() {
        row.min_y() - frame.min_y()
    } else if row.max_y() > frame.max_y() {
        row.max_y() - frame.max_y()
    } else {
        return;
    };
    let top = (offset.y + shift).max(0.0);
    let _ = list
        .scroll(PixelsVector2D::new(offset.x, top), ScrollBehavior::Instant)
        .await;
}

/// Whether the list is open; a disabled control closes it.
pub(super) fn use_open(disabled: bool) -> Signal<bool> {
    let mut open = use_signal(|| false);
    use_effect(use_reactive!(|disabled| {
        if disabled {
            open.set(false);
        }
    }));
    open
}

/// The open list: `{id}-listbox`, with option `n` as `{id}-option-{n}`. It is named by the label
/// element `labelled_by` or by the text `label`, sits beside `anchor`, and scrolls to the `active`
/// option whenever that changes. Focus stays on the control.
#[component]
pub(super) fn OptionList(
    id: String,
    labelled_by: Option<String>,
    label: Option<String>,
    anchor: Signal<Option<Rc<MountedData>>>,
    options: Vec<String>,
    active: Option<usize>,
    selected: Option<usize>,
    #[props(default)] check: bool,
    onhover: EventHandler<usize>,
    onchoose: EventHandler<usize>,
) -> Element {
    let mut placement = use_signal(|| None::<Placement>);
    let mut list = use_signal(|| None::<Rc<MountedData>>);
    // Mounted options by index; mounting is not a change, so nothing subscribes to it.
    let mut rows = use_signal(HashMap::<usize, Rc<MountedData>>::new);
    use_effect(use_reactive!(|active| {
        let Some(index) = active else {
            return;
        };
        let (list, row) = (list.peek().clone(), rows.peek().get(&index).cloned());
        if let (Some(list), Some(row)) = (list, row) {
            spawn(reveal(list, row));
        }
    }));
    rsx! {
        ul {
            id: "{id}-listbox",
            class: "select-listbox",
            role: "listbox",
            aria_labelledby: labelled_by,
            aria_label: label,
            "data-placement": placement().map(Placement::side_name),
            max_height: placement().and_then(Placement::max_height),
            // Keeps focus, and so the highlighted option, on the control.
            onmousedown: move |event| event.prevent_default(),
            onmounted: move |event: MountedEvent| {
                let element = event.data();
                list.set(Some(element.clone()));
                spawn(async move {
                    placement.set(Some(place(anchor.peek().clone(), element.clone()).await));
                    let row = active.and_then(|index| rows.peek().get(&index).cloned());
                    if let Some(row) = row {
                        reveal(element, row).await;
                    }
                });
            },
            for (index, text) in options.into_iter().enumerate() {
                li {
                    key: "{index}",
                    id: "{id}-option-{index}",
                    class: if Some(index) == active { "select-option active" } else { "select-option" },
                    role: "option",
                    aria_selected: if Some(index) == selected { "true" } else { "false" },
                    onmounted: move |event: MountedEvent| {
                        rows.write().insert(index, event.data());
                    },
                    onmousemove: move |_| onhover.call(index),
                    onclick: move |_| onchoose.call(index),
                    span { class: "select-option-label", "{text}" }
                    if check {
                        Icon { name: IconName::Check }
                    }
                }
            }
        }
    }
}
