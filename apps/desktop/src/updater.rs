//! Signed updates through `tauri-plugin-updater`.
//!
//! macOS, Windows, and `AppImage` installations update in place. Debian and RPM installations
//! belong to the system package manager, so a newer release is only offered through its
//! release page.

use std::{
    sync::{Arc, Mutex, MutexGuard},
    time::Duration,
};

use semver::Version;
use tauri::{AppHandle, Manager as _};
use tauri_plugin_updater::{Update, UpdaterExt as _};
use whispernow_api::{
    ApiErrorDto, ManualUpdatePackageDto, UpdateErrorCodeDto, UpdateErrorDto, UpdatePhaseDto,
    UpdateProgressDto, UpdateStateDto,
};

pub(crate) const PERIODIC_CHECK_INTERVAL: Duration = Duration::from_hours(6);
const MAX_RELEASE_NOTES_BYTES: usize = 8 * 1024;

type StateCallback = Arc<dyn Fn(UpdateStateDto) + Send + Sync>;

/// The plugin's version comparator: only a stable release newer than the running one.
pub(crate) fn is_newer_stable(current: &Version, release: &Version) -> bool {
    release.pre.is_empty() && release > current
}

/// How a newer release reaches the user.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum UpdateDelivery {
    /// The plugin replaces the running bundle after the user restarts.
    InPlace,
    /// The user installs the release's package with the system package manager.
    Manual(ManualUpdatePackageDto),
}

/// A newer release the user installs by hand from its release page.
struct ManualUpdate {
    version: String,
    notes: Option<String>,
    package: ManualUpdatePackageDto,
}

impl ManualUpdate {
    fn publish(&self, state: &mut UpdateStateDto) {
        state.phase = UpdatePhaseDto::ManualInstallAvailable;
        state.available_version = Some(self.version.clone());
        state.release_notes.clone_from(&self.notes);
        state.manual_package = Some(self.package);
        state.progress = None;
    }

    /// The GitHub release page, which lists the packages.
    fn release_page(&self) -> String {
        format!(
            "{}/releases/tag/v{}",
            env!("CARGO_PKG_REPOSITORY"),
            self.version
        )
    }
}

fn error_dto(code: UpdateErrorCodeDto) -> UpdateErrorDto {
    UpdateErrorDto {
        code,
        recoverable: !matches!(
            code,
            UpdateErrorCodeDto::InvalidSignature | UpdateErrorCodeDto::MalformedMetadata
        ),
    }
}

/// The published update state and what it holds. `U` is the downloaded update, so the
/// transitions are tested without the plugin.
struct Updates<U> {
    state: UpdateStateDto,
    /// The downloaded update and its signed bytes.
    ready: Option<(U, Vec<u8>)>,
    manual: Option<ManualUpdate>,
    /// Whether the user asked for the running check, which then reports its result.
    check_is_manual: bool,
}

impl<U> Updates<U> {
    fn new(enabled: bool) -> Self {
        Self {
            state: UpdateStateDto {
                phase: if enabled {
                    UpdatePhaseDto::Idle
                } else {
                    UpdatePhaseDto::Disabled
                },
                available_version: None,
                release_notes: None,
                progress: None,
                error: None,
                notification_dismissed: false,
                restart_deferred: false,
                manual_package: None,
            },
            ready: None,
            manual: None,
            check_is_manual: false,
        }
    }

    /// Whether a check starts. A check asked while one runs joins it.
    fn begin_check(&mut self, manual: bool) -> bool {
        match self.state.phase {
            UpdatePhaseDto::Disabled
            | UpdatePhaseDto::ReadyToRestart
            | UpdatePhaseDto::Installing => false,
            UpdatePhaseDto::Checking | UpdatePhaseDto::Downloading => {
                self.check_is_manual |= manual;
                false
            }
            _ => {
                self.check_is_manual = manual;
                self.state.phase = UpdatePhaseDto::Checking;
                self.state.progress = None;
                self.state.error = None;
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
            UpdatePhaseDto::UpToDate
        } else {
            UpdatePhaseDto::Idle
        };
        self.state.available_version = None;
        self.state.release_notes = None;
        self.state.manual_package = None;
        self.state.progress = None;
        self.state.error = None;
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
        self.state.error = None;
        self.manual = Some(update);
    }

    fn downloading(&mut self, version: String, notes: Option<String>) {
        self.state.phase = UpdatePhaseDto::Downloading;
        self.state.available_version = Some(version);
        self.state.release_notes = notes;
        self.state.progress = Some(UpdateProgressDto {
            downloaded_bytes: 0,
            total_bytes: None,
        });
    }

    fn progress(&mut self, downloaded_bytes: u64, total_bytes: Option<u64>) {
        if self.state.phase == UpdatePhaseDto::Downloading {
            self.state.progress = Some(UpdateProgressDto {
                downloaded_bytes,
                total_bytes,
            });
        }
    }

    fn downloaded(&mut self, update: U, bytes: Vec<u8>) {
        self.ready = Some((update, bytes));
        self.state.phase = UpdatePhaseDto::ReadyToRestart;
        self.state.progress = None;
        self.state.error = None;
        self.state.notification_dismissed = false;
    }

    /// A scheduled check fails quietly and keeps offering a manual release it found before.
    fn failed(&mut self, code: UpdateErrorCodeDto) {
        let manual = self.check_is_manual;
        self.state.phase = if manual {
            UpdatePhaseDto::Error
        } else {
            UpdatePhaseDto::Idle
        };
        self.state.manual_package = None;
        self.state.progress = None;
        self.state.error = manual.then(|| error_dto(code));
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
            UpdatePhaseDto::ReadyToRestart | UpdatePhaseDto::ManualInstallAvailable
        ) {
            self.state.notification_dismissed = true;
        }
    }

    fn release_page(&self) -> Option<String> {
        (self.state.phase == UpdatePhaseDto::ManualInstallAvailable)
            .then(|| self.manual.as_ref().map(ManualUpdate::release_page))
            .flatten()
    }

    /// Takes the downloaded update for installing. `dictating` tells the user the restart
    /// waits for the dictation in flight.
    fn take_ready(&mut self, dictating: bool) -> Option<(U, Vec<u8>)> {
        if self.state.phase != UpdatePhaseDto::ReadyToRestart {
            return None;
        }
        let ready = self.ready.take()?;
        self.state.restart_deferred = dictating;
        self.state.error = None;
        Some(ready)
    }

    fn installing(&mut self) {
        self.state.phase = UpdatePhaseDto::Installing;
        self.state.restart_deferred = false;
    }

    fn restore_ready(&mut self, ready: (U, Vec<u8>), error: Option<UpdateErrorCodeDto>) {
        self.ready = Some(ready);
        self.state.phase = UpdatePhaseDto::ReadyToRestart;
        self.state.error = error.map(error_dto);
        self.state.restart_deferred = false;
    }
}

pub(crate) struct UpdateController {
    app: AppHandle,
    on_state: StateCallback,
    updates: Mutex<Updates<Update>>,
}

impl UpdateController {
    pub(crate) fn new(app: AppHandle, enabled: bool, on_state: StateCallback) -> Arc<Self> {
        Arc::new(Self {
            app,
            on_state,
            updates: Mutex::new(Updates::new(enabled)),
        })
    }

    pub(crate) fn snapshot(&self) -> UpdateStateDto {
        self.lock().state.clone()
    }

    /// A poisoned lock means another thread panicked mid-update, which is a bug; it is never
    /// reported as an update error.
    fn lock(&self) -> MutexGuard<'_, Updates<Update>> {
        self.updates.lock().expect("updater state lock poisoned")
    }

    /// Changes the state and publishes the result.
    fn update<T>(&self, change: impl FnOnce(&mut Updates<Update>) -> T) -> T {
        let (result, snapshot) = {
            let mut updates = self.lock();
            let result = change(&mut updates);
            (result, updates.state.clone())
        };
        (self.on_state)(snapshot);
        result
    }

    /// Checks for a newer release and downloads it. Only one check runs at a time: a check
    /// asked while one runs joins it and returns the current state.
    pub(crate) async fn check(&self, manual: bool) -> UpdateStateDto {
        if self.update(|updates| updates.begin_check(manual))
            && let Err(code) = self.check_and_download().await
        {
            self.update(|updates| updates.failed(code));
        }
        self.snapshot()
    }

    async fn check_and_download(&self) -> Result<(), UpdateErrorCodeDto> {
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
        let bytes = update
            .download(
                |chunk, total| {
                    downloaded = downloaded.saturating_add(chunk as u64);
                    self.update(|updates| updates.progress(downloaded, total));
                },
                || {},
            )
            .await
            .map_err(|error| updater_error(error, UpdaterPhase::Download))?;
        self.update(|updates| updates.downloaded(update, bytes));
        Ok(())
    }

    pub(crate) fn dismiss(&self) -> UpdateStateDto {
        self.update(Updates::dismiss);
        self.snapshot()
    }

    /// Opens the release page of the offered manual update. The page address never crosses the
    /// webview boundary; it is derived here from the release's version.
    pub(crate) fn open_release_page(&self) -> UpdateStateDto {
        let Some(release_page) = self.lock().release_page() else {
            return self.snapshot();
        };
        let error = open_in_browser(&release_page).err();
        self.update(|updates| updates.state.error = error.map(error_dto));
        self.snapshot()
    }

    /// Installs the downloaded update and restarts.
    pub(crate) async fn restart_to_update(self: &Arc<Self>) -> Result<UpdateStateDto, ApiErrorDto> {
        let dictating = self
            .app
            .state::<crate::gateway::Gateway>()
            .services_state()
            .phase
            .is_busy();
        let Some(ready) = self.update(|updates| updates.take_ready(dictating)) else {
            return Ok(self.snapshot());
        };
        let controller = Arc::clone(self);
        tokio::task::spawn_blocking(move || controller.install(ready))
            .await
            .unwrap_or_else(|_| {
                tracing::error!("update installer worker stopped");
                Ok(())
            })?;
        Ok(self.snapshot())
    }

    /// Runs on a blocking worker: waiting for the dictation in flight, installing, and shutting
    /// down all block on native, filesystem, and process work.
    fn install(&self, (update, bytes): (Update, Vec<u8>)) -> Result<(), ApiErrorDto> {
        let state = self.app.state::<crate::gateway::Gateway>();
        if let Err(error) = state.services.dictation.begin_update() {
            self.update(|updates| updates.restore_ready((update, bytes), None));
            return Err(error);
        }
        self.update(Updates::installing);
        match update.install(&bytes) {
            Ok(()) => {
                shut_down_for_update(&self.app);
                self.app.request_restart();
            }
            Err(error) => {
                state.services.dictation.end_update();
                let code = updater_error(error, UpdaterPhase::Install);
                self.update(|updates| updates.restore_ready((update, bytes), Some(code)));
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

/// The `latest.json` entry an installation reads and how its update is delivered.
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
) -> Result<UpdateRoute, UpdateErrorCodeDto> {
    use tauri::utils::config::BundleType;

    if os != "linux" {
        return Ok(UpdateRoute {
            target: None,
            delivery: UpdateDelivery::InPlace,
        });
    }
    // The release matrix builds Linux packages for x86-64 only.
    if architecture != "x86_64" {
        return Err(UpdateErrorCodeDto::UnsupportedTarget);
    }
    let (format, delivery) = match bundle {
        Some(BundleType::AppImage) => ("appimage", UpdateDelivery::InPlace),
        Some(BundleType::Deb) => ("deb", UpdateDelivery::Manual(ManualUpdatePackageDto::Deb)),
        Some(BundleType::Rpm) => ("rpm", UpdateDelivery::Manual(ManualUpdatePackageDto::Rpm)),
        _ => return Err(UpdateErrorCodeDto::UnsupportedTarget),
    };
    Ok(UpdateRoute {
        target: Some(format!("linux-{architecture}-{format}")),
        delivery,
    })
}

/// Hands the release page to the desktop's default browser. Manual delivery exists only for
/// Linux packages, so other platforms have no opener.
fn open_in_browser(url: &str) -> Result<(), UpdateErrorCodeDto> {
    #[cfg(target_os = "linux")]
    {
        let mut opener = std::process::Command::new("xdg-open")
            .arg(url)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .map_err(|_| UpdateErrorCodeDto::ReleasePageUnavailable)?;
        std::thread::spawn(move || {
            if !opener.wait().is_ok_and(|status| status.success()) {
                tracing::warn!("release page opener failed");
            }
        });
        Ok(())
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = url;
        Err(UpdateErrorCodeDto::ReleasePageUnavailable)
    }
}

#[tauri::command]
#[allow(
    clippy::needless_pass_by_value,
    reason = "Tauri command state injection requires an owned State wrapper"
)]
pub(crate) fn get_update_state(
    controller: tauri::State<'_, Arc<UpdateController>>,
) -> UpdateStateDto {
    controller.snapshot()
}

#[tauri::command]
pub(crate) async fn check_for_updates(
    app: AppHandle,
    controller: tauri::State<'_, Arc<UpdateController>>,
) -> Result<UpdateStateDto, ApiErrorDto> {
    app.state::<crate::gateway::Gateway>()
        .refuse_while_updating()?;
    Ok(controller.inner().check(true).await)
}

#[tauri::command]
#[allow(
    clippy::needless_pass_by_value,
    reason = "Tauri command state injection requires an owned State wrapper"
)]
pub(crate) fn dismiss_update(
    controller: tauri::State<'_, Arc<UpdateController>>,
) -> UpdateStateDto {
    controller.dismiss()
}

#[tauri::command]
pub(crate) async fn restart_to_update(
    controller: tauri::State<'_, Arc<UpdateController>>,
) -> Result<UpdateStateDto, ApiErrorDto> {
    Arc::clone(controller.inner()).restart_to_update().await
}

/// Closes the application before the installer replaces it. Only the first call shuts down.
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
) -> UpdateStateDto {
    controller.open_release_page()
}

#[derive(Clone, Copy)]
enum UpdaterPhase {
    Check,
    Download,
    Install,
}

fn updater_error(error: tauri_plugin_updater::Error, phase: UpdaterPhase) -> UpdateErrorCodeDto {
    use tauri_plugin_updater::Error;

    match error {
        Error::Minisign(_) | Error::Base64(_) | Error::SignatureUtf8(_) => {
            UpdateErrorCodeDto::InvalidSignature
        }
        Error::TargetNotFound(_)
        | Error::TargetsNotFound(_)
        | Error::UnsupportedArch
        | Error::UnsupportedOs => UpdateErrorCodeDto::UnsupportedTarget,
        Error::Serialization(_) | Error::Semver(_) | Error::UrlParse(_) => {
            UpdateErrorCodeDto::MalformedMetadata
        }
        Error::ReleaseNotFound => UpdateErrorCodeDto::ServerUnavailable,
        Error::Reqwest(error) if error.is_connect() => UpdateErrorCodeDto::Offline,
        Error::Reqwest(_) | Error::Network(_) => match phase {
            UpdaterPhase::Check => UpdateErrorCodeDto::ServerUnavailable,
            UpdaterPhase::Download => UpdateErrorCodeDto::DownloadInterrupted,
            UpdaterPhase::Install => UpdateErrorCodeDto::InstallFailed,
        },
        Error::Io(error) => match phase {
            UpdaterPhase::Install => match error.kind() {
                std::io::ErrorKind::PermissionDenied => UpdateErrorCodeDto::InsufficientPrivileges,
                std::io::ErrorKind::ReadOnlyFilesystem => UpdateErrorCodeDto::ReadOnlyFilesystem,
                _ => UpdateErrorCodeDto::InstallFailed,
            },
            UpdaterPhase::Download => UpdateErrorCodeDto::DownloadInterrupted,
            UpdaterPhase::Check => UpdateErrorCodeDto::Internal,
        },
        Error::DebInstallFailed | Error::PackageInstallFailed | Error::InvalidUpdaterFormat => {
            UpdateErrorCodeDto::InstallFailed
        }
        _ => match phase {
            UpdaterPhase::Install => UpdateErrorCodeDto::InstallFailed,
            UpdaterPhase::Check | UpdaterPhase::Download => UpdateErrorCodeDto::Internal,
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
            package: ManualUpdatePackageDto::Deb,
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
    fn deb_and_rpm_notify_while_appimage_macos_and_windows_update_in_place() {
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
                UpdateDelivery::Manual(ManualUpdatePackageDto::Deb)
            )
        );
        assert_eq!(
            update_route("linux", "x86_64", Some(&BundleType::Rpm)),
            route(
                "linux-x86_64-rpm",
                UpdateDelivery::Manual(ManualUpdatePackageDto::Rpm)
            )
        );
        assert_eq!(
            update_route("linux", "x86_64", Some(&BundleType::AppImage)),
            route("linux-x86_64-appimage", UpdateDelivery::InPlace)
        );
        for (os, bundle) in [
            ("macos", Some(BundleType::App)),
            ("windows", Some(BundleType::Nsis)),
        ] {
            assert_eq!(
                update_route(os, "x86_64", bundle.as_ref()),
                Ok(UpdateRoute {
                    target: None,
                    delivery: UpdateDelivery::InPlace,
                }),
                "{os}"
            );
        }
        for (architecture, bundle) in [
            ("x86_64", None),
            ("x86_64", Some(BundleType::Nsis)),
            ("aarch64", Some(BundleType::AppImage)),
            ("riscv64", Some(BundleType::Deb)),
        ] {
            assert_eq!(
                update_route("linux", architecture, bundle.as_ref()),
                Err(UpdateErrorCodeDto::UnsupportedTarget),
                "{architecture} {bundle:?}"
            );
        }
    }

    #[test]
    fn a_package_release_is_offered_through_its_release_page() {
        let mut updates = updates();
        assert!(updates.begin_check(true));
        updates.found_manual(manual("0.2.1"));
        assert_eq!(updates.state.phase, UpdatePhaseDto::ManualInstallAvailable);
        assert_eq!(updates.state.available_version.as_deref(), Some("0.2.1"));
        assert_eq!(updates.state.release_notes.as_deref(), Some("notes"));
        assert_eq!(
            updates.state.manual_package,
            Some(ManualUpdatePackageDto::Deb)
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
        assert_eq!(updates.state.phase, UpdatePhaseDto::UpToDate);
        assert_eq!(updates.state.manual_package, None);
        assert_eq!(updates.release_page(), None);
    }

    #[test]
    fn a_scheduled_failure_is_quiet_and_keeps_the_offer_but_a_manual_one_is_visible() {
        let mut updates = updates();
        updates.begin_check(false);
        updates.failed(UpdateErrorCodeDto::Offline);
        assert_eq!(updates.state.phase, UpdatePhaseDto::Idle);
        assert_eq!(updates.state.error, None);

        updates.begin_check(false);
        updates.found_manual(manual("0.2.1"));
        updates.begin_check(false);
        updates.failed(UpdateErrorCodeDto::ServerUnavailable);
        assert_eq!(updates.state.phase, UpdatePhaseDto::ManualInstallAvailable);
        assert_eq!(updates.state.available_version.as_deref(), Some("0.2.1"));

        updates.begin_check(true);
        updates.failed(UpdateErrorCodeDto::InvalidSignature);
        assert_eq!(updates.state.phase, UpdatePhaseDto::Error);
        assert_eq!(updates.state.manual_package, None);
        assert_eq!(
            updates.state.error,
            Some(UpdateErrorDto {
                code: UpdateErrorCodeDto::InvalidSignature,
                recoverable: false,
            })
        );
        assert_eq!(updates.release_page(), None);
    }

    #[test]
    fn a_manual_check_joining_a_scheduled_one_reports_its_result() {
        let mut updates = updates();
        assert!(updates.begin_check(false));
        assert!(!updates.begin_check(true), "the running check is joined");
        updates.failed(UpdateErrorCodeDto::Offline);
        assert_eq!(updates.state.phase, UpdatePhaseDto::Error);
        assert_eq!(
            updates.state.error.map(|error| error.code),
            Some(UpdateErrorCodeDto::Offline)
        );
    }

    #[test]
    fn a_downloaded_update_survives_dismissal_and_a_failed_install() {
        let mut updates = updates();
        updates.begin_check(true);
        updates.downloading("0.2.1".to_owned(), None);
        updates.progress(2, Some(4));
        assert_eq!(
            updates.state.progress,
            Some(UpdateProgressDto {
                downloaded_bytes: 2,
                total_bytes: Some(4),
            })
        );
        updates.downloaded("update", vec![1, 2, 3, 4]);
        assert_eq!(updates.state.phase, UpdatePhaseDto::ReadyToRestart);
        updates.dismiss();
        assert!(updates.state.notification_dismissed);
        assert!(
            !updates.begin_check(true),
            "no check replaces a ready update"
        );

        let ready = updates.take_ready(false).expect("ready update");
        updates.installing();
        assert_eq!(updates.state.phase, UpdatePhaseDto::Installing);
        assert!(!updates.begin_check(true), "no check runs while installing");
        updates.restore_ready(ready, Some(UpdateErrorCodeDto::InstallFailed));
        assert_eq!(updates.state.phase, UpdatePhaseDto::ReadyToRestart);
        assert_eq!(
            updates.state.error.map(|error| error.code),
            Some(UpdateErrorCodeDto::InstallFailed)
        );
        assert_eq!(
            updates.take_ready(false),
            Some(("update", vec![1, 2, 3, 4])),
            "the install can be retried"
        );
    }

    #[test]
    fn a_restart_while_dictating_tells_the_user_it_waits() {
        let mut updates = updates();
        updates.begin_check(true);
        updates.downloaded("update", vec![1]);
        assert!(updates.take_ready(true).is_some());
        assert!(updates.state.restart_deferred);
        updates.installing();
        assert!(!updates.state.restart_deferred);
    }

    #[test]
    fn install_io_errors_have_install_specific_recovery_codes() {
        for (kind, expected) in [
            (
                std::io::ErrorKind::PermissionDenied,
                UpdateErrorCodeDto::InsufficientPrivileges,
            ),
            (
                std::io::ErrorKind::ReadOnlyFilesystem,
                UpdateErrorCodeDto::ReadOnlyFilesystem,
            ),
            (std::io::ErrorKind::Other, UpdateErrorCodeDto::InstallFailed),
        ] {
            let error = tauri_plugin_updater::Error::Io(std::io::Error::from(kind));
            assert_eq!(updater_error(error, UpdaterPhase::Install), expected);
        }
        assert_eq!(
            updater_error(
                tauri_plugin_updater::Error::Io(std::io::Error::other("download")),
                UpdaterPhase::Download,
            ),
            UpdateErrorCodeDto::DownloadInterrupted
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
    }
}
