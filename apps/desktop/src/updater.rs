//! macOS, Windows, and `AppImage` installations update in place. Debian and RPM installations
//! belong to the system package manager, so a newer release is only offered through its
//! release page.

use std::{
    sync::{Arc, Mutex, MutexGuard},
    time::{Duration, Instant},
};
use whispernow_domain::{AppError, ErrorCategory};

use semver::Version;
use tauri::{AppHandle, Manager as _};
use tauri_plugin_updater::{Update, UpdaterExt as _};
use whispernow_domain::ipc::{
    ManualUpdatePackage, UpdateError, UpdatePhase, UpdateProgress, UpdateState,
};
use whispernow_platform::open_url;

use crate::services::{Dictation, NativeServices};

pub(crate) const PERIODIC_CHECK_INTERVAL: Duration = Duration::from_hours(6);
const MAX_RELEASE_NOTES_BYTES: usize = 8 * 1024;
/// How long a download may go without receiving a byte. The builder's `timeout` bounds only
/// the check, and a total limit would cut off a slow large download.
const DOWNLOAD_READ_TIMEOUT: Duration = Duration::from_mins(1);

pub(crate) fn is_newer_stable(current: &Version, release: &Version) -> bool {
    release.pre.is_empty() && release > current
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum UpdateDelivery {
    InPlace,
    Manual(ManualUpdatePackage),
}

struct ManualUpdate {
    version: String,
    notes: Option<String>,
    package: ManualUpdatePackage,
}

impl ManualUpdate {
    fn publish(&self, state: &mut UpdateState) {
        state.phase = UpdatePhase::ManualInstallAvailable {
            version: self.version.clone(),
            release_notes: self.notes.clone(),
            package: self.package,
            error: None,
        };
    }
}

/// Which download chunks publish progress: one per `INTERVAL`, and one whenever the whole
/// percent changes.
#[derive(Default)]
struct ProgressThrottle {
    published_at: Option<Instant>,
    percent: Option<u64>,
}

impl ProgressThrottle {
    const INTERVAL: Duration = Duration::from_millis(100);

    fn poll_due(&mut self, now: Instant, downloaded: u64, total: Option<u64>) -> bool {
        let percent = total
            .filter(|total| *total > 0)
            .map(|total| downloaded.saturating_mul(100) / total);
        let waited = self
            .published_at
            .is_none_or(|at| now.saturating_duration_since(at) >= Self::INTERVAL);
        if !waited && percent == self.percent {
            return false;
        }
        self.published_at = Some(now);
        self.percent = percent;
        true
    }
}

#[derive(Debug, PartialEq)]
struct ReadyUpdate<U> {
    update: U,
    bytes: Vec<u8>,
}

/// The published update state and what it holds. `U` is the downloaded update, so the
/// transitions are tested without the plugin.
struct Updates<U> {
    state: UpdateState,
    ready: Option<Arc<ReadyUpdate<U>>>,
    manual: Option<ManualUpdate>,
    /// Whether the user asked for the running check, which then reports its result.
    check_is_manual: bool,
}

impl<U> Updates<U> {
    fn new(enabled: bool) -> Self {
        Self {
            state: UpdateState {
                phase: if enabled {
                    UpdatePhase::Idle
                } else {
                    UpdatePhase::Disabled
                },
                notification_dismissed: false,
                restart_deferred: false,
            },
            ready: None,
            manual: None,
            check_is_manual: false,
        }
    }

    /// Whether a check starts. A check asked while one runs joins it.
    fn begin_check(&mut self, manual: bool) -> bool {
        match self.state.phase {
            UpdatePhase::Disabled
            | UpdatePhase::ReadyToRestart { .. }
            | UpdatePhase::Installing { .. } => false,
            UpdatePhase::Checking | UpdatePhase::Downloading { .. } => {
                self.check_is_manual |= manual;
                false
            }
            _ => {
                self.check_is_manual = manual;
                self.state.phase = UpdatePhase::Checking;
                self.state.restart_deferred = false;
                if manual {
                    self.state.notification_dismissed = false;
                }
                true
            }
        }
    }

    fn up_to_date(&mut self) {
        self.state.phase = if self.check_is_manual {
            UpdatePhase::UpToDate
        } else {
            UpdatePhase::Idle
        };
        self.manual = None;
    }

    fn found_manual(&mut self, update: ManualUpdate) {
        if self
            .manual
            .as_ref()
            .is_none_or(|previous| previous.version != update.version)
        {
            self.state.notification_dismissed = false;
        }
        update.publish(&mut self.state);
        self.manual = Some(update);
    }

    fn downloading(&mut self, version: String, notes: Option<String>) {
        self.state.phase = UpdatePhase::Downloading {
            version,
            release_notes: notes,
            progress: UpdateProgress {
                downloaded_bytes: 0,
                total_bytes: None,
            },
        };
    }

    fn progress(&mut self, downloaded_bytes: u64, total_bytes: Option<u64>) {
        if let UpdatePhase::Downloading { progress, .. } = &mut self.state.phase {
            *progress = UpdateProgress {
                downloaded_bytes,
                total_bytes,
            };
        }
    }

    fn downloaded(&mut self, update: U, bytes: Vec<u8>) {
        let UpdatePhase::Downloading {
            version,
            release_notes,
            ..
        } = &self.state.phase
        else {
            panic!("an update finishes downloading only from the downloading phase");
        };
        self.state.phase = UpdatePhase::ReadyToRestart {
            version: version.clone(),
            release_notes: release_notes.clone(),
            error: None,
        };
        self.ready = Some(Arc::new(ReadyUpdate { update, bytes }));
        self.state.notification_dismissed = false;
    }

    /// A scheduled check fails quietly and keeps offering a manual release it found before.
    fn failed(&mut self, code: UpdateError) {
        let manual = self.check_is_manual;
        self.state.phase = if manual {
            UpdatePhase::Error { error: code }
        } else {
            UpdatePhase::Idle
        };
        if let (false, Some(update)) = (manual, &self.manual) {
            update.publish(&mut self.state);
        }
        if !manual {
            tracing::warn!(?code, "scheduled update check failed");
        }
    }

    fn dismiss(&mut self) {
        if matches!(
            self.state.phase,
            UpdatePhase::ReadyToRestart { .. } | UpdatePhase::ManualInstallAvailable { .. }
        ) {
            self.state.notification_dismissed = true;
        }
    }

    fn release_page(&self) -> Option<String> {
        if let UpdatePhase::ManualInstallAvailable { version, .. } = &self.state.phase {
            Some(format!(
                "{}/releases/tag/v{version}",
                env!("CARGO_PKG_REPOSITORY")
            ))
        } else {
            None
        }
    }

    /// Takes the downloaded update for installing. `dictating` tells the user the restart
    /// waits for the dictation in flight.
    fn take_ready(&mut self, dictating: bool) -> Option<Arc<ReadyUpdate<U>>> {
        let UpdatePhase::ReadyToRestart { error, .. } = &mut self.state.phase else {
            return None;
        };
        let ready = self.ready.take()?;
        self.state.restart_deferred = dictating;
        *error = None;
        Some(ready)
    }

    fn installing(&mut self) {
        let UpdatePhase::ReadyToRestart {
            version,
            release_notes,
            ..
        } = &self.state.phase
        else {
            panic!("an update installs only from the ready phase");
        };
        self.state.phase = UpdatePhase::Installing {
            version: version.clone(),
            release_notes: release_notes.clone(),
        };
        self.state.restart_deferred = false;
    }

    fn restore_ready(&mut self, ready: Arc<ReadyUpdate<U>>, error: Option<UpdateError>) {
        let (UpdatePhase::ReadyToRestart {
            version,
            release_notes,
            ..
        }
        | UpdatePhase::Installing {
            version,
            release_notes,
        }) = &self.state.phase
        else {
            panic!("an update is restored only from the ready or installing phase");
        };
        self.state.phase = UpdatePhase::ReadyToRestart {
            version: version.clone(),
            release_notes: release_notes.clone(),
            error,
        };
        self.ready = Some(ready);
        self.state.restart_deferred = false;
    }

    fn finish_install_worker(
        &mut self,
        result: Result<Result<(), AppError>, tokio::task::JoinError>,
        ready: Arc<ReadyUpdate<U>>,
        dictation: &Dictation,
    ) -> Result<(), AppError> {
        result.unwrap_or_else(|error| {
            tracing::error!(
                panicked = error.is_panic(),
                cancelled = error.is_cancelled(),
                "update installer worker stopped"
            );
            dictation.end_update();
            self.restore_ready(ready, Some(UpdateError::InstallFailed));
            Err(AppError::new(
                ErrorCategory::General,
                "worker_stopped",
                "The update installer stopped before installation completed",
            ))
        })
    }
}

pub(crate) struct UpdateController {
    app: AppHandle,
    services: Arc<NativeServices>,
    updates: Mutex<Updates<Update>>,
}

impl UpdateController {
    pub(crate) fn new(app: AppHandle, services: Arc<NativeServices>, enabled: bool) -> Arc<Self> {
        Arc::new(Self {
            app,
            services,
            updates: Mutex::new(Updates::new(enabled)),
        })
    }

    pub(crate) fn snapshot(&self) -> UpdateState {
        self.lock().state.clone()
    }

    fn lock(&self) -> MutexGuard<'_, Updates<Update>> {
        self.updates.lock().expect("updater state lock poisoned")
    }

    /// Publishes under the lock, on the channel the application state uses, so events arrive in
    /// the order things happened.
    fn update<T>(&self, change: impl FnOnce(&mut Updates<Update>) -> T) -> T {
        let mut updates = self.lock();
        let result = change(&mut updates);
        self.services
            .state
            .events()
            .publish(whispernow_domain::ipc::EventPayload::UpdateChanged {
                state: updates.state.clone(),
            });
        result
    }

    /// A check asked while one runs joins it and returns the current state.
    pub(crate) async fn check(&self, manual: bool) -> UpdateState {
        if self.update(|updates| updates.begin_check(manual))
            && let Err(code) = self.check_and_download().await
        {
            self.update(|updates| updates.failed(code));
        }
        self.snapshot()
    }

    pub(crate) async fn check_manual(&self) -> Result<UpdateState, AppError> {
        self.services.state.refuse_during_maintenance()?;
        Ok(self.check(true).await)
    }

    async fn check_and_download(&self) -> Result<(), UpdateError> {
        let route = update_route(
            std::env::consts::OS,
            std::env::consts::ARCH,
            tauri::utils::platform::bundle_type().as_ref(),
        )?;
        let exit_app = self.app.clone();
        let mut builder = self
            .app
            .updater_builder()
            .timeout(Duration::from_secs(30))
            .configure_client(|client| client.read_timeout(DOWNLOAD_READ_TIMEOUT))
            // The Windows installer exits the process itself.
            .on_before_exit(move || {
                shut_down_for_update(&exit_app);
                exit_app.cleanup_before_exit();
            });
        if let Some(target) = route.target {
            builder = builder.target(target);
        }
        let update = builder
            .build()
            .map_err(|error| updater_error(error, UpdaterPhase::Check))?
            .check()
            .await
            .map_err(|error| updater_error(error, UpdaterPhase::Check))?;
        let Some(update) = update else {
            self.update(Updates::up_to_date);
            return Ok(());
        };
        let notes = sanitize_release_notes(update.body.as_deref());
        if let UpdateDelivery::Manual(package) = route.delivery {
            let manual = ManualUpdate {
                version: update.version.clone(),
                notes,
                package,
            };
            self.update(|updates| updates.found_manual(manual));
            return Ok(());
        }
        let version = update.version.clone();
        self.update(|updates| updates.downloading(version, notes));
        let mut downloaded = 0_u64;
        let mut throttle = ProgressThrottle::default();
        let bytes = update
            .download(
                |chunk, total| {
                    downloaded = downloaded.saturating_add(chunk as u64);
                    if throttle.poll_due(Instant::now(), downloaded, total) {
                        self.update(|updates| updates.progress(downloaded, total));
                    }
                },
                || {},
            )
            .await
            .map_err(|error| updater_error(error, UpdaterPhase::Download))?;
        self.update(|updates| updates.downloaded(update, bytes));
        Ok(())
    }

    pub(crate) fn dismiss(&self) -> UpdateState {
        self.update(Updates::dismiss);
        self.snapshot()
    }

    /// The page address never crosses the webview boundary; it is derived here from the
    /// release's version.
    pub(crate) fn open_release_page(&self) -> UpdateState {
        let Some(release_page) = self.lock().release_page() else {
            return self.snapshot();
        };
        let error = open_url(&release_page)
            .inspect_err(|error| tracing::warn!(%error, "could not open the release page"))
            .err()
            .map(|_| UpdateError::ReleasePageUnavailable);
        self.update(|updates| {
            if let UpdatePhase::ManualInstallAvailable { error: current, .. } =
                &mut updates.state.phase
            {
                *current = error;
            }
        });
        self.snapshot()
    }

    pub(crate) async fn restart_to_update(self: &Arc<Self>) -> Result<UpdateState, AppError> {
        let dictating = self.services.state.snapshot().phase.is_busy();
        let Some(ready) = self.update(|updates| updates.take_ready(dictating)) else {
            return Ok(self.snapshot());
        };
        let controller = Arc::clone(self);
        let worker_ready = Arc::clone(&ready);
        let result = tokio::task::spawn_blocking(move || controller.install(worker_ready)).await;
        self.update(|updates| {
            updates.finish_install_worker(result, ready, &self.services.dictation)
        })?;
        Ok(self.snapshot())
    }

    /// Runs on a blocking worker: waiting for the dictation, installing and shutting down all
    /// block.
    fn install(&self, ready: Arc<ReadyUpdate<Update>>) -> Result<(), AppError> {
        let dictation = &self.services.dictation;
        if let Err(error) = dictation.begin_update() {
            self.update(|updates| updates.restore_ready(ready, None));
            return Err(error);
        }
        self.update(Updates::installing);
        match ready.update.install(&ready.bytes) {
            Ok(()) => {
                shut_down_for_update(&self.app);
                self.app.request_restart();
            }
            Err(error) => {
                dictation.end_update();
                let code = updater_error(error, UpdaterPhase::Install);
                self.update(|updates| updates.restore_ready(ready, Some(code)));
            }
        }
        Ok(())
    }
}

fn sanitize_release_notes(notes: Option<&str>) -> Option<String> {
    let notes = notes?.trim();
    if notes.is_empty() {
        return None;
    }
    let mut output = String::with_capacity(notes.len().min(MAX_RELEASE_NOTES_BYTES));
    for character in notes.chars() {
        if output.len() + character.len_utf8() > MAX_RELEASE_NOTES_BYTES {
            break;
        }
        if !character.is_control() || matches!(character, '\n' | '\r' | '\t') {
            output.push(character);
        }
    }
    Some(output)
}

#[derive(Debug, Eq, PartialEq)]
struct UpdateRoute {
    /// Explicit updater target; `None` keeps the plugin's own `{os}-{arch}` selection.
    target: Option<String>,
    delivery: UpdateDelivery,
}

/// Routes each installation format. Linux formats read their own signed target, because a
/// generic Linux entry could hand a Debian installation an RPM or an `AppImage`.
fn update_route(
    os: &str,
    architecture: &str,
    bundle: Option<&tauri::utils::config::BundleType>,
) -> Result<UpdateRoute, UpdateError> {
    use tauri::utils::config::BundleType;

    if os != "linux" {
        return Ok(UpdateRoute {
            target: None,
            delivery: UpdateDelivery::InPlace,
        });
    }
    // The release matrix builds Linux packages for x86-64 only.
    if architecture != "x86_64" {
        return Err(UpdateError::UnsupportedTarget);
    }
    let (format, delivery) = match bundle {
        Some(BundleType::AppImage) => ("appimage", UpdateDelivery::InPlace),
        Some(BundleType::Deb) => ("deb", UpdateDelivery::Manual(ManualUpdatePackage::Deb)),
        Some(BundleType::Rpm) => ("rpm", UpdateDelivery::Manual(ManualUpdatePackage::Rpm)),
        _ => return Err(UpdateError::UnsupportedTarget),
    };
    Ok(UpdateRoute {
        target: Some(format!("linux-{architecture}-{format}")),
        delivery,
    })
}

#[tauri::command]
#[allow(
    clippy::needless_pass_by_value,
    reason = "Tauri command state injection requires an owned State wrapper"
)]
pub(crate) fn get_update_state(controller: tauri::State<'_, Arc<UpdateController>>) -> UpdateState {
    controller.snapshot()
}

#[tauri::command]
pub(crate) async fn check_for_updates(
    controller: tauri::State<'_, Arc<UpdateController>>,
) -> Result<UpdateState, AppError> {
    controller.check_manual().await
}

#[tauri::command]
#[allow(
    clippy::needless_pass_by_value,
    reason = "Tauri command state injection requires an owned State wrapper"
)]
pub(crate) fn dismiss_update(controller: tauri::State<'_, Arc<UpdateController>>) -> UpdateState {
    controller.dismiss()
}

#[tauri::command]
pub(crate) async fn restart_to_update(
    controller: tauri::State<'_, Arc<UpdateController>>,
) -> Result<UpdateState, AppError> {
    Arc::clone(controller.inner()).restart_to_update().await
}

/// Only the first call shuts down.
fn shut_down_for_update(app: &AppHandle) {
    let state = app.state::<crate::gateway::Gateway>();
    if state.begin_shutdown() {
        crate::hide_recording_bar(app);
        crate::hide_settings_window(app);
        state.finish_shutdown();
    }
}

#[tauri::command]
#[allow(
    clippy::needless_pass_by_value,
    reason = "Tauri command state injection requires an owned State wrapper"
)]
pub(crate) fn open_update_release_page(
    controller: tauri::State<'_, Arc<UpdateController>>,
) -> UpdateState {
    controller.open_release_page()
}

#[derive(Clone, Copy, Debug)]
enum UpdaterPhase {
    Check,
    Download,
    Install,
}

fn updater_error(error: tauri_plugin_updater::Error, phase: UpdaterPhase) -> UpdateError {
    use tauri_plugin_updater::Error;

    tracing::warn!(%error, ?phase, "update failed");
    match error {
        Error::Minisign(_) | Error::Base64(_) | Error::SignatureUtf8(_) => {
            UpdateError::InvalidSignature
        }
        Error::TargetNotFound(_)
        | Error::TargetsNotFound(_)
        | Error::UnsupportedArch
        | Error::UnsupportedOs => UpdateError::UnsupportedTarget,
        Error::Serialization(_) | Error::Semver(_) | Error::UrlParse(_) => {
            UpdateError::MalformedMetadata
        }
        Error::ReleaseNotFound => UpdateError::ServerUnavailable,
        Error::Reqwest(error) if error.is_connect() => UpdateError::Offline,
        Error::Reqwest(_) | Error::Network(_) => match phase {
            UpdaterPhase::Check => UpdateError::ServerUnavailable,
            UpdaterPhase::Download => UpdateError::DownloadInterrupted,
            UpdaterPhase::Install => UpdateError::InstallFailed,
        },
        Error::Io(error) => match phase {
            UpdaterPhase::Install => match error.kind() {
                std::io::ErrorKind::PermissionDenied => UpdateError::InsufficientPrivileges,
                std::io::ErrorKind::ReadOnlyFilesystem => UpdateError::ReadOnlyFilesystem,
                _ => UpdateError::InstallFailed,
            },
            UpdaterPhase::Download => UpdateError::DownloadInterrupted,
            UpdaterPhase::Check => UpdateError::Internal,
        },
        Error::DebInstallFailed | Error::PackageInstallFailed | Error::InvalidUpdaterFormat => {
            UpdateError::InstallFailed
        }
        _ => match phase {
            UpdaterPhase::Install => UpdateError::InstallFailed,
            UpdaterPhase::Check | UpdaterPhase::Download => UpdateError::Internal,
        },
    }
}

pub(crate) fn spawn_periodic_checks(controller: Arc<UpdateController>) {
    tauri::async_runtime::spawn(async move {
        tokio::time::sleep(Duration::from_secs(20)).await;
        let _ = controller.check(false).await;
        let mut interval = tokio::time::interval(PERIODIC_CHECK_INTERVAL);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        interval.tick().await;
        loop {
            interval.tick().await;
            let _ = controller.check(false).await;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Update states holding a fake downloaded update.
    fn updates() -> Updates<&'static str> {
        Updates::new(true)
    }

    fn manual(version: &str) -> ManualUpdate {
        ManualUpdate {
            version: version.to_owned(),
            notes: Some("notes".to_owned()),
            package: ManualUpdatePackage::Deb,
        }
    }

    fn version(value: &str) -> Version {
        Version::parse(value).expect("a version")
    }

    #[test]
    fn only_a_newer_stable_release_is_offered() {
        assert!(is_newer_stable(&version("0.2.0"), &version("0.2.1")));
        for candidate in ["0.2.0", "0.1.9", "0.2.1-rc.1"] {
            assert!(
                !is_newer_stable(&version("0.2.0"), &version(candidate)),
                "{candidate}"
            );
        }
    }

    #[test]
    fn installation_formats_choose_delivery_and_unsupported_linux_targets_are_refused() {
        use tauri::utils::config::BundleType;

        let route = |target: &str, delivery| {
            Ok(UpdateRoute {
                target: Some(target.to_owned()),
                delivery,
            })
        };
        assert_eq!(
            update_route("linux", "x86_64", Some(&BundleType::Deb)),
            route(
                "linux-x86_64-deb",
                UpdateDelivery::Manual(ManualUpdatePackage::Deb)
            )
        );
        assert_eq!(
            update_route("linux", "x86_64", Some(&BundleType::Rpm)),
            route(
                "linux-x86_64-rpm",
                UpdateDelivery::Manual(ManualUpdatePackage::Rpm)
            )
        );
        assert_eq!(
            update_route("linux", "x86_64", Some(&BundleType::AppImage)),
            route("linux-x86_64-appimage", UpdateDelivery::InPlace)
        );
        assert_eq!(
            update_route("macos", "x86_64", Some(&BundleType::App)),
            Ok(UpdateRoute {
                target: None,
                delivery: UpdateDelivery::InPlace,
            })
        );
        for (architecture, bundle) in [("x86_64", None), ("aarch64", Some(BundleType::AppImage))] {
            assert_eq!(
                update_route("linux", architecture, bundle.as_ref()),
                Err(UpdateError::UnsupportedTarget),
                "{architecture} {bundle:?}"
            );
        }
    }

    #[test]
    fn a_package_release_is_offered_through_its_release_page() {
        let mut updates = updates();
        assert!(updates.begin_check(true));
        updates.found_manual(manual("0.2.1"));
        assert_eq!(
            updates.state.phase,
            UpdatePhase::ManualInstallAvailable {
                version: "0.2.1".to_owned(),
                release_notes: Some("notes".to_owned()),
                package: ManualUpdatePackage::Deb,
                error: None,
            }
        );
        assert_eq!(
            updates.release_page().as_deref(),
            Some("https://github.com/stephan271c/WhisperNow/releases/tag/v0.2.1")
        );
        assert!(updates.take_ready(false).is_none(), "nothing is installed");

        updates.dismiss();
        assert!(updates.begin_check(false));
        updates.found_manual(manual("0.2.1"));
        assert!(
            updates.state.notification_dismissed,
            "the same release stays dismissed"
        );
        assert!(updates.begin_check(false));
        updates.found_manual(manual("0.2.2"));
        assert!(
            !updates.state.notification_dismissed,
            "a newer release is announced again"
        );

        assert!(updates.begin_check(true));
        updates.up_to_date();
        assert_eq!(updates.state.phase, UpdatePhase::UpToDate);
        assert_eq!(updates.release_page(), None);
    }

    #[test]
    fn a_scheduled_failure_is_quiet_and_keeps_the_offer_but_a_manual_one_is_visible() {
        let mut updates = updates();
        updates.begin_check(false);
        updates.failed(UpdateError::Offline);
        assert_eq!(updates.state.phase, UpdatePhase::Idle);

        updates.begin_check(false);
        updates.found_manual(manual("0.2.1"));
        updates.begin_check(false);
        updates.failed(UpdateError::ServerUnavailable);
        assert_eq!(
            updates.state.phase,
            UpdatePhase::ManualInstallAvailable {
                version: "0.2.1".to_owned(),
                release_notes: Some("notes".to_owned()),
                package: ManualUpdatePackage::Deb,
                error: None,
            }
        );

        updates.begin_check(true);
        updates.failed(UpdateError::InvalidSignature);
        assert_eq!(
            updates.state.phase,
            UpdatePhase::Error {
                error: UpdateError::InvalidSignature,
            }
        );
        let UpdatePhase::Error { error } = updates.state.phase else {
            panic!("manual failure");
        };
        assert!(!error.is_recoverable());
        assert_eq!(updates.release_page(), None);
    }

    #[test]
    fn a_download_that_stalls_fails_and_a_new_check_can_start() {
        let mut updates = updates();
        assert!(updates.begin_check(false));
        updates.downloading("0.2.1".to_owned(), None);
        updates.progress(2, Some(4));
        assert!(!updates.begin_check(true), "the running download is joined");

        updates.failed(UpdateError::DownloadInterrupted);
        assert_eq!(
            updates.state.phase,
            UpdatePhase::Error {
                error: UpdateError::DownloadInterrupted,
            }
        );
        assert!(updates.begin_check(true));
    }

    #[test]
    fn progress_is_published_every_hundred_milliseconds_and_at_each_whole_percent() {
        let start = Instant::now();
        let at = |millis| start + Duration::from_millis(millis);
        let mut throttle = ProgressThrottle::default();
        assert!(throttle.poll_due(at(0), 1, Some(1_000)), "the first chunk");
        assert!(!throttle.poll_due(at(10), 5, Some(1_000)), "too soon");
        assert!(
            throttle.poll_due(at(20), 10, Some(1_000)),
            "a whole percent"
        );
        assert!(!throttle.poll_due(at(119), 15, Some(1_000)), "99 ms later");
        assert!(throttle.poll_due(at(120), 16, Some(1_000)), "100 ms later");

        let mut unknown = ProgressThrottle::default();
        assert!(unknown.poll_due(at(0), 1, None));
        assert!(
            !unknown.poll_due(at(99), 1_000_000, None),
            "no total, no percent"
        );
        assert!(unknown.poll_due(at(100), 1_000_001, None));
        assert!(
            !unknown.poll_due(at(150), 1_000_002, Some(0)),
            "an empty total is no percent"
        );
    }

    #[test]
    fn many_tiny_chunks_publish_a_bounded_number_of_states() {
        let start = Instant::now();
        let (chunk, total) = (256_u64, 1024 * 1024_u64);
        let mut throttle = ProgressThrottle::default();
        let published = (1..=total / chunk)
            .filter(|index| {
                let now = start + Duration::from_micros(index * 100);
                throttle.poll_due(now, index * chunk, Some(total))
            })
            .count();
        // 4096 chunks in 410 ms: the first, 100 whole percents, and one per 100 ms between them.
        assert!((101..=105).contains(&published), "{published}");
    }

    #[test]
    fn a_manual_check_joining_a_scheduled_one_reports_its_result() {
        let mut updates = updates();
        assert!(updates.begin_check(false));
        assert!(!updates.begin_check(true), "the running check is joined");
        updates.failed(UpdateError::Offline);
        assert_eq!(
            updates.state.phase,
            UpdatePhase::Error {
                error: UpdateError::Offline,
            }
        );
    }

    #[test]
    fn a_downloaded_update_survives_dismissal_and_a_failed_install() {
        let mut updates = updates();
        updates.begin_check(true);
        updates.downloading("0.2.1".to_owned(), None);
        updates.progress(2, Some(4));
        assert_eq!(
            updates.state.phase,
            UpdatePhase::Downloading {
                version: "0.2.1".to_owned(),
                release_notes: None,
                progress: UpdateProgress {
                    downloaded_bytes: 2,
                    total_bytes: Some(4),
                },
            }
        );
        updates.downloaded("update", vec![1, 2, 3, 4]);
        assert_eq!(
            updates.state.phase,
            UpdatePhase::ReadyToRestart {
                version: "0.2.1".to_owned(),
                release_notes: None,
                error: None,
            }
        );
        updates.dismiss();
        assert!(updates.state.notification_dismissed);
        assert!(
            !updates.begin_check(true),
            "no check replaces a ready update"
        );

        let ready = updates.take_ready(false).expect("ready update");
        updates.installing();
        assert_eq!(
            updates.state.phase,
            UpdatePhase::Installing {
                version: "0.2.1".to_owned(),
                release_notes: None,
            }
        );
        assert!(!updates.begin_check(true), "no check runs while installing");
        updates.restore_ready(ready, Some(UpdateError::InstallFailed));
        assert_eq!(
            updates.state.phase,
            UpdatePhase::ReadyToRestart {
                version: "0.2.1".to_owned(),
                release_notes: None,
                error: Some(UpdateError::InstallFailed),
            }
        );
        assert_eq!(
            updates.take_ready(false),
            Some(Arc::new(ReadyUpdate {
                update: "update",
                bytes: vec![1, 2, 3, 4]
            })),
            "the install can be retried"
        );
    }

    #[test]
    fn a_failed_installer_worker_restores_the_download_and_dictation_admission() {
        let fixture = crate::services::Fixture::new();
        fixture.flush();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("runtime");
        let mut updates = updates();
        updates.begin_check(true);
        updates.downloading("0.2.1".to_owned(), None);
        updates.downloaded("update", vec![1, 2, 3, 4]);
        let ready = updates.take_ready(true).expect("ready update");
        assert!(updates.state.restart_deferred);
        fixture.dictation.begin_update().expect("begin update");
        updates.installing();
        assert_eq!(
            fixture.state.snapshot().phase,
            crate::services::Phase::Updating
        );
        assert_eq!(
            updates.state.phase,
            UpdatePhase::Installing {
                version: "0.2.1".to_owned(),
                release_notes: None,
            }
        );
        let worker_ready = Arc::clone(&ready);

        let result = runtime.block_on(async move {
            tokio::task::spawn_blocking(move || -> Result<(), AppError> {
                drop(worker_ready);
                panic!("installer worker failed");
            })
            .await
        });
        assert!(result.as_ref().expect_err("worker failed").is_panic());
        let error = updates
            .finish_install_worker(result, ready, &fixture.dictation)
            .expect_err("worker failure reaches the caller");

        assert_eq!(error.code, "worker_stopped");
        assert_eq!(error.category, ErrorCategory::General);
        assert_eq!(fixture.state.snapshot().phase, crate::services::Phase::Idle);
        assert_eq!(
            updates.state.phase,
            UpdatePhase::ReadyToRestart {
                version: "0.2.1".to_owned(),
                release_notes: None,
                error: Some(UpdateError::InstallFailed),
            }
        );
        assert!(!updates.state.restart_deferred);
        assert!(!updates.begin_check(true), "the download remains ready");
        let retry = updates.take_ready(false).expect("retry the installation");
        assert_eq!(retry.bytes, [1, 2, 3, 4]);
        assert_eq!(retry.update, "update");
        assert!(matches!(
            updates.state.phase,
            UpdatePhase::ReadyToRestart { error: None, .. }
        ));
        fixture
            .dictation
            .begin_update()
            .expect("retry the update phase");
        fixture.dictation.end_update();
        fixture
            .dictation
            .start(crate::services::RecordingTrigger::Toggle)
            .expect("dictation is admitted after recovery");
        assert!(fixture.state.snapshot().phase.is_busy());
        fixture.dictation.cancel().expect("cancel test dictation");
        fixture.flush();
    }

    #[test]
    fn install_io_errors_have_install_specific_codes() {
        for (kind, expected) in [
            (
                std::io::ErrorKind::PermissionDenied,
                UpdateError::InsufficientPrivileges,
            ),
            (
                std::io::ErrorKind::ReadOnlyFilesystem,
                UpdateError::ReadOnlyFilesystem,
            ),
            (std::io::ErrorKind::Other, UpdateError::InstallFailed),
        ] {
            let error = tauri_plugin_updater::Error::Io(std::io::Error::from(kind));
            assert_eq!(updater_error(error, UpdaterPhase::Install), expected);
        }
        assert_eq!(
            updater_error(
                tauri_plugin_updater::Error::Io(std::io::Error::other("download")),
                UpdaterPhase::Download,
            ),
            UpdateError::DownloadInterrupted
        );
    }

    #[test]
    fn release_notes_are_plain_bounded_text() {
        let notes = format!(
            "<script>alert(1)</script>{}",
            "x".repeat(MAX_RELEASE_NOTES_BYTES)
        );
        let safe = sanitize_release_notes(Some(&notes)).expect("notes");
        assert!(safe.len() <= MAX_RELEASE_NOTES_BYTES);
        assert!(safe.contains("<script>"));
        assert_eq!(
            sanitize_release_notes(Some("Fix\0es\u{7}\nNext\tstep\r\nDone\u{7f}")),
            Some("Fixes\nNext\tstep\r\nDone".to_owned())
        );
    }
}
