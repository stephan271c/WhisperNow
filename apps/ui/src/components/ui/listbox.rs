//! Where an open option list sits beside its control, and how it scrolls to a row.

use std::rc::Rc;

use dioxus::html::geometry::{PixelsRect, PixelsVector2D};
use dioxus::prelude::*;

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
pub(super) struct Placement {
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

    /// The list's `data-placement`, which shows it once it was measured.
    pub(super) const fn side_name(self) -> &'static str {
        match self.side {
            Side::Below => "below",
            Side::Above => "above",
        }
    }

    pub(super) fn room_style(self) -> Option<String> {
        self.room.map(|room| format!("--select-room: {room:.0}px"))
    }
}

pub(super) async fn place(trigger: Option<Rc<MountedData>>, list: Rc<MountedData>) -> Placement {
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
pub(super) async fn reveal(list: Rc<MountedData>, row: Rc<MountedData>) {
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
