use dioxus::prelude::Key;
use whispernow_domain::ipc::MainWindowSection;

use crate::notice::NoticeScope;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum Route {
    #[default]
    Home,
    History,
    Settings(SettingsRoute),
    Setup,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum SettingsRoute {
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
    pub(crate) const ALL: [Self; 7] = [
        Self::General,
        Self::Audio,
        Self::SpeechModels,
        Self::AiCleanup,
        Self::Vocabulary,
        Self::PrivacyData,
        Self::AboutUpdates,
    ];

    #[must_use]
    pub(crate) const fn label(self) -> &'static str {
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

    #[must_use]
    pub(crate) const fn nav_id(self) -> &'static str {
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TopLevel {
    Home,
    History,
    Settings,
}

impl TopLevel {
    pub(crate) const ALL: [Self; 3] = [Self::Home, Self::History, Self::Settings];

    #[must_use]
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Home => "Home",
            Self::History => "History",
            Self::Settings => "Settings",
        }
    }

    #[must_use]
    pub(crate) const fn nav_id(self) -> &'static str {
        match self {
            Self::Home => "nav-home",
            Self::History => "nav-history",
            Self::Settings => "nav-settings",
        }
    }

    #[must_use]
    pub(crate) const fn route(self) -> Route {
        match self {
            Self::Home => Route::Home,
            Self::History => Route::History,
            Self::Settings => Route::Settings(SettingsRoute::General),
        }
    }
}

impl From<MainWindowSection> for Route {
    fn from(section: MainWindowSection) -> Self {
        match section {
            MainWindowSection::History => Self::History,
            MainWindowSection::Settings => Self::Settings(SettingsRoute::General),
            MainWindowSection::Audio => Self::Settings(SettingsRoute::Audio),
            MainWindowSection::SpeechModels => Self::Settings(SettingsRoute::SpeechModels),
            MainWindowSection::AiCleanup => Self::Settings(SettingsRoute::AiCleanup),
        }
    }
}

impl Route {
    /// The page's one `h1`.
    #[must_use]
    pub(crate) const fn title(self) -> &'static str {
        match self {
            Self::Home => "Home",
            Self::History => "History",
            Self::Settings(page) => page.label(),
            Self::Setup => "Set up WhisperNow",
        }
    }

    #[must_use]
    pub(crate) const fn top_level(self) -> TopLevel {
        match self {
            Self::Home | Self::Setup => TopLevel::Home,
            Self::History => TopLevel::History,
            Self::Settings(_) => TopLevel::Settings,
        }
    }

    #[must_use]
    pub(crate) const fn hosts(self, scope: NoticeScope) -> bool {
        match scope {
            NoticeScope::Global => false,
            NoticeScope::Home => matches!(self, Self::Home),
            NoticeScope::History => matches!(self, Self::History),
            NoticeScope::Setup => matches!(self, Self::Setup),
            NoticeScope::Settings => matches!(self, Self::Settings(_)),
            NoticeScope::Models => matches!(
                self,
                Self::Settings(SettingsRoute::SpeechModels | SettingsRoute::AiCleanup)
            ),
            NoticeScope::Data => matches!(self, Self::Settings(SettingsRoute::PrivacyData)),
        }
    }
}

#[must_use]
pub(crate) fn nav_keyboard_target(key: &Key, current: usize, count: usize) -> Option<usize> {
    if count == 0 {
        return None;
    }
    let last = count - 1;
    match key {
        Key::ArrowDown => Some(if current >= last { 0 } else { current + 1 }),
        Key::ArrowUp => Some(if current == 0 { last } else { current - 1 }),
        Key::Home => Some(0),
        Key::End => Some(last),
        _ => None,
    }
}

#[cfg(test)]
mod tests {

    use super::{Key, Route, SettingsRoute, nav_keyboard_target};
    use crate::notice::NoticeScope;

    #[test]
    fn arrow_home_and_end_keys_move_through_the_sidebar_and_wrap() {
        assert_eq!(nav_keyboard_target(&Key::ArrowDown, 0, 3), Some(1));
        assert_eq!(nav_keyboard_target(&Key::ArrowDown, 2, 3), Some(0));
        assert_eq!(nav_keyboard_target(&Key::ArrowUp, 0, 3), Some(2));
        assert_eq!(nav_keyboard_target(&Key::ArrowUp, 2, 3), Some(1));
        assert_eq!(nav_keyboard_target(&Key::Home, 2, 10), Some(0));
        assert_eq!(nav_keyboard_target(&Key::End, 0, 10), Some(9));
        assert_eq!(
            nav_keyboard_target(&Key::Enter, 1, 3),
            None,
            "activation stays native"
        );
        assert_eq!(nav_keyboard_target(&Key::ArrowDown, 0, 0), None);
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
