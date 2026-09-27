//! The main window's typed destinations.

use whispernow_api::MainWindowSectionDto;

use crate::notice::NoticeScope;

/// One page of the main window.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Route {
    #[default]
    Home,
    History,
    Settings(SettingsRoute),
    /// The first-run setup steps, reached from the Home checklist.
    Setup,
}

/// A Settings sub-page.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SettingsRoute {
    #[default]
    General,
    Audio,
    SpeechModels,
    AiCleanup,
    Vocabulary,
    PrivacyData,
    AboutUpdates,
}

impl SettingsRoute {
    /// Every sub-page in sidebar order.
    pub const ALL: [Self; 7] = [
        Self::General,
        Self::Audio,
        Self::SpeechModels,
        Self::AiCleanup,
        Self::Vocabulary,
        Self::PrivacyData,
        Self::AboutUpdates,
    ];

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::General => "General & Shortcuts",
            Self::Audio => "Audio",
            Self::SpeechModels => "Speech models",
            Self::AiCleanup => "AI cleanup",
            Self::Vocabulary => "Vocabulary",
            Self::PrivacyData => "Privacy & Data",
            Self::AboutUpdates => "About & Updates",
        }
    }

    /// The id of this sub-page's sidebar link.
    #[must_use]
    pub const fn nav_id(self) -> &'static str {
        match self {
            Self::General => "nav-settings-general",
            Self::Audio => "nav-settings-audio",
            Self::SpeechModels => "nav-settings-models",
            Self::AiCleanup => "nav-settings-ai-cleanup",
            Self::Vocabulary => "nav-settings-vocabulary",
            Self::PrivacyData => "nav-settings-privacy",
            Self::AboutUpdates => "nav-settings-about",
        }
    }
}

/// A sidebar entry at the top level.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TopLevel {
    Home,
    History,
    Settings,
}

impl TopLevel {
    pub const ALL: [Self; 3] = [Self::Home, Self::History, Self::Settings];

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Home => "Home",
            Self::History => "History",
            Self::Settings => "Settings",
        }
    }

    #[must_use]
    pub const fn nav_id(self) -> &'static str {
        match self {
            Self::Home => "nav-home",
            Self::History => "nav-history",
            Self::Settings => "nav-settings",
        }
    }

    /// Where selecting this entry goes.
    #[must_use]
    pub const fn route(self) -> Route {
        match self {
            Self::Home => Route::Home,
            Self::History => Route::History,
            Self::Settings => Route::Settings(SettingsRoute::General),
        }
    }
}

impl Route {
    /// The destination an overlay action or tray request names.
    #[must_use]
    pub const fn from_section(section: MainWindowSectionDto) -> Self {
        match section {
            MainWindowSectionDto::History => Self::History,
            MainWindowSectionDto::Settings => Self::Settings(SettingsRoute::General),
        }
    }

    /// The page's one `h1`.
    #[must_use]
    pub const fn title(self) -> &'static str {
        match self {
            Self::Home => "Home",
            Self::History => "History",
            Self::Settings(page) => page.label(),
            Self::Setup => "Set up WhisperNow",
        }
    }

    /// The sidebar entry this page belongs to; setup is reached from Home.
    #[must_use]
    pub const fn top_level(self) -> TopLevel {
        match self {
            Self::Home | Self::Setup => TopLevel::Home,
            Self::History => TopLevel::History,
            Self::Settings(_) => TopLevel::Settings,
        }
    }

    /// Whether this page shows notices of `scope` next to the controls that raised them.
    /// Notices of other scopes go to the window's toast region.
    #[must_use]
    pub const fn hosts(self, scope: NoticeScope) -> bool {
        match scope {
            NoticeScope::Global => false,
            NoticeScope::Home => matches!(self, Self::Home),
            NoticeScope::History => matches!(self, Self::History),
            NoticeScope::Setup => matches!(self, Self::Setup),
            NoticeScope::Settings => matches!(self, Self::Settings(_)),
            NoticeScope::Models => matches!(self, Self::Settings(SettingsRoute::SpeechModels)),
            NoticeScope::Data => matches!(self, Self::Settings(SettingsRoute::PrivacyData)),
            NoticeScope::Updates => matches!(self, Self::Settings(SettingsRoute::AboutUpdates)),
        }
    }
}

/// Where arrow, Home and End keys move focus in a list of `count` navigation links.
#[must_use]
pub fn nav_keyboard_target(key: &str, current: usize, count: usize) -> Option<usize> {
    if count == 0 {
        return None;
    }
    let last = count - 1;
    match key {
        "ArrowDown" => Some(if current >= last { 0 } else { current + 1 }),
        "ArrowUp" => Some(if current == 0 { last } else { current - 1 }),
        "Home" => Some(0),
        "End" => Some(last),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use whispernow_api::MainWindowSectionDto;

    use super::{Route, SettingsRoute, TopLevel, nav_keyboard_target};
    use crate::NoticeScope;

    #[test]
    fn arrow_home_and_end_keys_move_through_the_sidebar_and_wrap() {
        assert_eq!(nav_keyboard_target("ArrowDown", 0, 3), Some(1));
        assert_eq!(nav_keyboard_target("ArrowDown", 2, 3), Some(0));
        assert_eq!(nav_keyboard_target("ArrowUp", 0, 3), Some(2));
        assert_eq!(nav_keyboard_target("ArrowUp", 2, 3), Some(1));
        assert_eq!(nav_keyboard_target("Home", 2, 10), Some(0));
        assert_eq!(nav_keyboard_target("End", 0, 10), Some(9));
        assert_eq!(
            nav_keyboard_target("Enter", 1, 3),
            None,
            "activation stays native"
        );
        assert_eq!(nav_keyboard_target("ArrowDown", 0, 0), None);
    }

    #[test]
    fn overlay_sections_open_their_routes_and_each_page_has_one_title() {
        assert_eq!(
            Route::from_section(MainWindowSectionDto::History),
            Route::History
        );
        assert_eq!(
            Route::from_section(MainWindowSectionDto::Settings),
            Route::Settings(SettingsRoute::General)
        );
        assert_eq!(Route::default(), Route::Home);
        assert_eq!(Route::Setup.top_level(), TopLevel::Home);
        let mut ids = SettingsRoute::ALL.map(SettingsRoute::nav_id).to_vec();
        ids.dedup();
        assert_eq!(ids.len(), SettingsRoute::ALL.len());
        for top in TopLevel::ALL {
            assert_eq!(top.route().top_level(), top);
        }
    }

    #[test]
    fn a_notice_is_hosted_only_by_the_page_with_its_controls() {
        let general = Route::Settings(SettingsRoute::General);
        assert!(general.hosts(NoticeScope::Settings));
        assert!(!Route::Home.hosts(NoticeScope::Settings));
        assert!(!general.hosts(NoticeScope::Models));
        assert!(Route::Settings(SettingsRoute::SpeechModels).hosts(NoticeScope::Models));
        assert!(Route::Home.hosts(NoticeScope::Home));
        assert!(!Route::History.hosts(NoticeScope::Home));
        for route in [Route::Home, Route::History, Route::Setup, general] {
            assert!(!route.hosts(NoticeScope::Global), "{route:?}");
        }
    }
}
