use std::time::{Duration, Instant};

use atspi::{
    ObjectRefOwned, State,
    proxy::{accessible::AccessibleProxy, bus::BusProxy},
    zbus::{self, proxy::CacheProperties},
};
use futures_util::{StreamExt as _, stream::FuturesUnordered};

use crate::PasteKeys;

/// A slow or missing accessibility bus may delay the paste only this long.
const LOOKUP_TIMEOUT: Duration = Duration::from_millis(300);
const REGISTRY: &str = "org.a11y.atspi.Registry";

/// A connection to the accessibility bus, kept between paste-key lookups.
///
/// Not `atspi::AccessibilityConnection`: with atspi's `p2p` feature, which atspi 0.30 always
/// enables, connecting asks every application for its interfaces and opens a peer connection to
/// each, about 19 ms, and fails when any one application refuses.
///
/// The connection's zbus tasks run on the tokio runtime that opened it, so each runtime that
/// looks up keys keeps its own.
#[derive(Default)]
pub struct AccessibilityBus {
    connection: Option<zbus::Connection>,
}

impl AccessibilityBus {
    /// The paste keystroke the active window expects, found through AT-SPI by
    /// reading only each application's top-level windows. When it cannot be found
    /// in time, `Ctrl+V` is used.
    pub async fn active_paste_keys(&mut self) -> PasteKeys {
        let started = Instant::now();
        let application = tokio::time::timeout(LOOKUP_TIMEOUT, self.active_application()).await;
        tracing::debug!(
            duration_ms = started.elapsed().as_secs_f64() * 1_000.0,
            "active window lookup finished"
        );
        match application {
            Ok(Some(application)) => PasteKeys::for_application(&application),
            Ok(None) => {
                tracing::debug!("active window not found; pasting with Ctrl+V");
                PasteKeys::ControlV
            }
            Err(_) => {
                // A connection that stopped answering is replaced on the next paste.
                self.connection = None;
                tracing::debug!("active window lookup timed out; pasting with Ctrl+V");
                PasteKeys::ControlV
            }
        }
    }

    /// Asks over the kept connection; one that fails is replaced once.
    async fn active_application(&mut self) -> Option<String> {
        if let Some(connection) = &self.connection {
            match active_application(connection).await {
                Ok(application) => return application,
                Err(error) => {
                    tracing::debug!(%error, "the accessibility bus connection failed");
                    self.connection = None;
                }
            }
        }
        let connection = connect()
            .await
            .inspect_err(|error| tracing::debug!(%error, "the accessibility bus is unavailable"))
            .ok()?;
        let application = active_application(&connection).await.ok()?;
        self.connection = Some(connection);
        application
    }
}

async fn connect() -> zbus::Result<zbus::Connection> {
    let session = zbus::Connection::session().await?;
    let address = BusProxy::new(&session).await?.get_address().await?;
    zbus::connection::Builder::address(address.as_str())?
        .build()
        .await
}

/// The name of the application whose window is active. Applications are asked
/// concurrently so one that does not answer cannot hide the others. Only a failure
/// to read the registry is an error; one application failing is not.
async fn active_application(connection: &zbus::Connection) -> zbus::Result<Option<String>> {
    let registry = AccessibleProxy::builder(connection)
        .destination(REGISTRY)?
        .cache_properties(CacheProperties::No)
        .build()
        .await?;
    let applications = registry.get_children().await?;
    Ok(applications
        .iter()
        .filter(|application| !application.is_null())
        .map(|application| active_window_owner(connection, application))
        .collect::<FuturesUnordered<_>>()
        .filter_map(std::future::ready)
        .next()
        .await)
}

async fn active_window_owner(
    connection: &zbus::Connection,
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

/// Uncached: a cached proxy adds a match rule and reads every property for the one `name` read.
async fn proxy_for<'a>(
    connection: &'a zbus::Connection,
    object: &ObjectRefOwned,
) -> Option<AccessibleProxy<'a>> {
    AccessibleProxy::builder(connection)
        .destination(object.name()?.clone())
        .ok()?
        .path(object.path().clone())
        .ok()?
        .cache_properties(CacheProperties::No)
        .build()
        .await
        .ok()
}
