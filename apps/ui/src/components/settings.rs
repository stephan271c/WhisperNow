use dioxus::prelude::*;

use super::ui::SettingRow;
use crate::UiContext;
use crate::management::SettingKey;

#[component]
pub(super) fn AutosaveRow(
    id: String,
    title: String,
    description: Option<String>,
    setting: SettingKey,
    children: Element,
) -> Element {
    let context = use_context::<UiContext>();
    let management = context.management;
    let error = management.read().setting_error(setting);
    rsx! {
        SettingRow { id, title, description, error, {children} }
    }
}

#[component]
pub(super) fn SettingsLoading() -> Element {
    rsx! { p { role: "status", "Loading settings…" } }
}
