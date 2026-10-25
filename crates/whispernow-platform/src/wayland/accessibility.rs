use std::time::Duration;

use atspi::{AccessibilityConnection, ObjectRefOwned, State, proxy::accessible::AccessibleProxy};
use futures_util::{StreamExt as _, stream::FuturesUnordered};

use crate::PasteKeys;

/// A slow or missing accessibility bus may delay the paste only this long.
const LOOKUP_TIMEOUT: Duration = Duration::from_millis(300);

/// The paste keystroke the active window expects, found through AT-SPI by
/// reading only each application's top-level windows. When it cannot be found
/// in time, `Ctrl+V` is used.
pub async fn active_paste_keys() -> PasteKeys {
    if let Ok(Some(application)) = tokio::time::timeout(LOOKUP_TIMEOUT, active_application()).await
    {
        PasteKeys::for_application(&application)
    } else {
        tracing::debug!("active window not found; pasting with Ctrl+V");
        PasteKeys::ControlV
    }
}

/// The name of the application whose window is active. Applications are asked
/// concurrently so one that does not answer cannot hide the others.
async fn active_application() -> Option<String> {
    let connection = AccessibilityConnection::new().await.ok()?;
    let registry = connection.root_accessible_on_registry().await.ok()?;
    let applications = registry.get_children().await.ok()?;
    applications
        .iter()
        .filter(|application| !application.is_null())
        .map(|application| active_window_owner(&connection, application))
        .collect::<FuturesUnordered<_>>()
        .filter_map(std::future::ready)
        .next()
        .await
}

async fn active_window_owner(
    connection: &AccessibilityConnection,
    application: &ObjectRefOwned,
) -> Option<String> {
    let application = proxy_for(connection, application).await?;
    for window in application.get_children().await.ok()? {
        if window.is_null() {
            continue;
        }
        let window = proxy_for(connection, &window).await?;
        if window.get_state().await.ok()?.contains(State::Active) {
            return Some(application.name().await.unwrap_or_default());
        }
    }
    None
}

async fn proxy_for<'a>(
    connection: &'a AccessibilityConnection,
    object: &ObjectRefOwned,
) -> Option<AccessibleProxy<'a>> {
    AccessibleProxy::builder(connection.connection())
        .destination(object.name()?.clone())
        .ok()?
        .path(object.path().clone())
        .ok()?
        .build()
        .await
        .ok()
}
