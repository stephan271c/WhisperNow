use dioxus::prelude::*;

/// Outline icons drawn on a 24-unit grid in the current text colour.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum IconName {
    Home,
    History,
    Sparkles,
    Book,
    Sliders,
    CheckCircle,
    Mic,
    ArrowRight,
    ArrowUp,
    ArrowDown,
    Shield,
    Wave,
    Close,
    Info,
    Warning,
    AlertCircle,
    Stop,
    Gear,
    Copy,
    Trash,
    Folder,
    Play,
    Minimize,
    Maximize,
    Restore,
}

impl IconName {
    const fn path(self) -> &'static str {
        match self {
            Self::Home => "m3 10 9-7 9 7v10a1 1 0 0 1-1 1h-5v-7H9v7H4a1 1 0 0 1-1-1Z",
            Self::History => "M3 11a9 9 0 1 1 2.5 7M3 4v7h7m2-4v5l3 2",
            Self::Sparkles => {
                "m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5ZM20 2v4m-2-2h4"
            }
            Self::Book => "M12 5C8 2 5 3 3 4v15c3-1 6-1 9 1 3-2 6-2 9-1V4c-2-1-5-2-9 1Zm0 0v15",
            Self::Sliders => "M4 6h16M4 12h16M4 18h16M8 3v6m8 0v6m-6 0v6",
            Self::CheckCircle => "m8 12 3 3 5-6M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0",
            Self::Mic => {
                "M9 5a3 3 0 0 1 6 0v7a3 3 0 0 1-6 0Zm-3 6v1a6 6 0 0 0 12 0v-1m-6 7v4m-3 0h6"
            }
            Self::ArrowRight => "M4 12h16m-6-6 6 6-6 6",
            Self::ArrowUp => "M12 20V4m-6 6 6-6 6 6",
            Self::ArrowDown => "M12 4v16m-6-6 6 6 6-6",
            Self::Shield => "m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6Zm-4 9 3 3 5-6",
            Self::Wave => "M3 10v4m4-8v12m5-15v18m5-15v12m4-8v4",
            Self::Close => "M6 6l12 12M18 6 6 18",
            Self::Info => "M12 11v5m0-8h.01M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0",
            Self::Warning => "M12 4 2.5 20h19ZM12 10v4m0 3h.01",
            Self::AlertCircle => "M12 8v5m0 3h.01M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0",
            Self::Stop | Self::Maximize => "M6 6h12v12H6Z",
            Self::Gear => {
                "M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6Zm7.4-3a7.4 7.4 0 0 0-.1-1.3l2-1.6-2-3.4-2.4 1a7.5 7.5 0 0 0-2.2-1.3L14.4 3h-4l-.4 2.4a7.5 7.5 0 0 0-2.2 1.3l-2.4-1-2 3.4 2 1.6a7.4 7.4 0 0 0 0 2.6l-2 1.6 2 3.4 2.4-1a7.5 7.5 0 0 0 2.2 1.3l.4 2.4h4l.4-2.4a7.5 7.5 0 0 0 2.2-1.3l2.4 1 2-3.4-2-1.6c.1-.4.1-.9.1-1.3Z"
            }
            Self::Copy => "M9 9h11v11H9ZM5 15H4V4h11v1",
            Self::Trash => "M4 7h16M10 11v6m4-6v6M6 7l1 13h10l1-13M9 7V4h6v3",
            Self::Folder => {
                "M3 6a1 1 0 0 1 1-1h5l2 2h9a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1Z"
            }
            Self::Play => "M7 4.5v15l12.5-7.5Z",
            Self::Minimize => "M6 12h12",
            Self::Restore => "M6 9h9v9H6Zm3-3h9v9",
        }
    }
}

/// The app's key mark (`brand/logo/whispernow-symbol-reversed.svg`), filled in the
/// current text colour. It is solid rather than outlined, so it is not an `IconName`.
#[component]
pub(crate) fn BrandMark() -> Element {
    rsx! {
        svg { class: "icon", width: "20", height: "20", view_box: "28 28 200 200",
            fill: "currentColor", "aria-hidden": "true", "focusable": "false",
            path {
                fill_rule: "evenodd",
                d: "M72 28H184A44 44 0 0 1 228 72V184A44 44 0 0 1 184 228H72A44 44 0 0 1 28 184V72A44 44 0 0 1 72 28ZM72 41H184A31 31 0 0 1 215 72V170A31 31 0 0 1 184 201H72A31 31 0 0 1 41 170V72A31 31 0 0 1 72 41Z",
            }
            path { d: "M80.5 107A11.5 11.5 0 0 1 103.5 107V135A11.5 11.5 0 0 1 80.5 135ZM116.5 81A11.5 11.5 0 0 1 139.5 81V161A11.5 11.5 0 0 1 116.5 161ZM152.5 97A11.5 11.5 0 0 1 175.5 97V145A11.5 11.5 0 0 1 152.5 145Z" }
        }
    }
}

/// A decorative icon; the control or text beside it carries the accessible name.
#[component]
pub(crate) fn Icon(name: IconName) -> Element {
    rsx! {
        svg { class: "icon", width: "20", height: "20", view_box: "0 0 24 24",
            fill: "none", stroke: "currentColor", stroke_width: "1.7",
            stroke_linecap: "round", stroke_linejoin: "round",
            "aria-hidden": "true", "focusable": "false",
            path { d: name.path() }
        }
    }
}
