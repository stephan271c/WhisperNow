mod activation;
mod autostart;
mod gateway;
#[cfg(target_os = "linux")]
mod input_helper;
mod overlay;
#[cfg(target_os = "linux")]
mod portal_identity;
mod services;
mod shortcuts;
mod tray;
mod updater;
#[cfg(target_os = "linux")]
mod wayland_shortcuts;

use std::{io, sync::Arc, time::Duration};

use gateway::{Gateway, commands, forward_events};
use services::{Backend, Composition};
use tauri::Manager as _;
use tauri_plugin_global_shortcut::ShortcutState;
use whispernow_domain::WindowGeometry;

use crate::shortcuts::{
    KeyTransition, ShortcutEvent, ShortcutId, Shortcuts, StartedRegistrar, TauriShortcutRegistrar,
    cancel_binding, recording_binding,
};
use crate::{
    activation::ActivationGate,
    autostart::{AutostartRegistration, TauriAutostart},
    overlay::RecordingOverlay,
    tray::{TrayAction, TrayController},
};

pub(crate) const SETTINGS_WINDOW: &str = "settings";
pub(crate) const RECORDING_BAR_WINDOW: &str = "recording-bar";

const SHUTDOWN_WATCHDOG_TIMEOUT: Duration = Duration::from_secs(20);

/// The command catalog test reads the paths from the same tokens as the handler.
macro_rules! registered_commands {
    ($($($segment:ident)::+),+ $(,)?) => {
        #[cfg(test)]
        const REGISTERED_COMMANDS: &[&str] = &[$(stringify!($($segment)::+)),+];

        fn invoke_handler() -> impl Fn(tauri::ipc::Invoke) -> bool + Send + Sync + 'static {
            tauri::generate_handler![$($($segment)::+),+]
        }
    };
}

registered_commands![
    commands::get_settings,
    commands::update_settings,
    commands::set_start_on_login,
    commands::set_api_key,
    commands::clear_api_key,
    commands::list_provider_models,
    commands::list_audio_devices,
    commands::test_microphone,
    commands::list_models,
    commands::download_model,
    commands::cancel_download,
    commands::select_model,
    commands::delete_model,
    commands::save_setup_checkpoint,
    commands::complete_setup,
    commands::restart_setup,
    commands::get_permissions,
    commands::request_permission,
    commands::open_permission_settings,
    commands::start_test_dictation,
    commands::stop_recording,
    commands::cancel_recording,
    commands::dismiss_recording_overlay,
    commands::copy_text,
    commands::open_main_window,
    commands::get_state,
    commands::get_history,
    commands::get_history_audio,
    commands::open_recordings_folder,
    commands::delete_history_entry,
    commands::clear_history,
    commands::data_locations,
    commands::clear_data,
    commands::suspend_shortcuts,
    commands::resume_shortcuts,
    updater::get_update_state,
    updater::check_for_updates,
    updater::dismiss_update,
    updater::restart_to_update,
    updater::open_update_release_page,
];

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    whispernow_platform::init_logging();
    tracing::info!(version = env!("CARGO_PKG_VERSION"), "starting WhisperNow");
    let (shortcut_events, shortcut_receiver) = tokio::sync::mpsc::unbounded_channel();
    let plugin_shortcut_events = shortcut_events.clone();
    let native_wayland = native_wayland_session();
    let floating_windows = !native_wayland || use_xwayland_windows();
    let activation = Arc::new(ActivationGate::default());
    let callback_activation = Arc::clone(&activation);
    let public_key = option_env!("WHISPERNOW_UPDATER_PUBLIC_KEY")
        .unwrap_or_default()
        .trim();
    let updates_enabled = !cfg!(debug_assertions) && !public_key.is_empty();
    let mut updater_plugin =
        tauri_plugin_updater::Builder::new().default_version_comparator(|current, release| {
            updater::is_newer_stable(&current, &release.version)
        });
    if !public_key.is_empty() {
        updater_plugin = updater_plugin.pubkey(public_key);
    }
    let builder = tauri::Builder::default()
        // The single-instance plugin must be registered first so no other plugin performs
        // duplicate side effects in a secondary process.
        .plugin(tauri_plugin_single_instance::init(
            move |app, _args, _cwd| {
                let activate = callback_activation.request_activation(|| {
                    app.try_state::<Gateway>()
                        .map(|state| state.backend.state.snapshot().phase)
                });
                if activate {
                    show_settings_window(app);
                }
            },
        ))
        .plugin(tauri_plugin_autostart::Builder::new().build())
        .plugin(tauri_plugin_process::init())
        .plugin(updater_plugin.build());
    let builder = if native_wayland {
        builder
    } else {
        builder.plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(move |_app, shortcut, event| {
                    let id = ShortcutId(shortcut.id());
                    let transition = match event.state() {
                        ShortcutState::Pressed => KeyTransition::Pressed,
                        ShortcutState::Released => KeyTransition::Released,
                    };
                    let _queued =
                        plugin_shortcut_events.send(ShortcutEvent::Transition { id, transition });
                })
                .build(),
        )
    };
    let startup = Startup {
        native_wayland,
        floating_windows,
        updates_enabled,
        activation,
        shortcut_events,
        shortcut_receiver,
    };
    let application = builder
        .setup(move |app| setup(app, startup))
        .on_window_event(on_window_event)
        .invoke_handler(invoke_handler())
        .build(tauri::generate_context!())
        .unwrap_or_else(|error| panic!("Tauri setup failed: {error}"));
    application.run(on_run_event);
}

struct Startup {
    native_wayland: bool,
    floating_windows: bool,
    updates_enabled: bool,
    activation: Arc<ActivationGate>,
    shortcut_events: tokio::sync::mpsc::UnboundedSender<ShortcutEvent>,
    shortcut_receiver: tokio::sync::mpsc::UnboundedReceiver<ShortcutEvent>,
}

fn setup(app: &mut tauri::App, startup: Startup) -> Result<(), Box<dyn std::error::Error>> {
    let Startup {
        native_wayland,
        floating_windows,
        updates_enabled,
        activation,
        shortcut_events,
        shortcut_receiver,
    } = startup;
    let overlay = RecordingOverlay::new(floating_windows);
    let (events, event_receiver) = services::Events::channel();
    let resource_dir = app.path().resource_dir()?;
    let Composition {
        backend,
        enable_autostart,
    } = Backend::compose(events, native_wayland, &resource_dir)
        .map_err(|error| io::Error::other(format!("desktop setup failed ({})", error.code)))?;
    let tray = TrayController::new(app, backend.output.can_copy())?;
    let settings = backend.settings.snapshot();
    let start_minimized = settings.start_minimized;
    let autostart: Arc<dyn AutostartRegistration> =
        Arc::new(TauriAutostart::new(app.handle().clone()));
    if enable_autostart && let Err(error) = autostart.set_enabled(true) {
        backend.state.publish_error(error);
    }
    if let Some(geometry) = settings.window_geometry {
        restore_settings_geometry(app, geometry);
    }
    // The page draws its own titlebar when the window is undecorated. Windows is
    // undecorated by tauri.windows.conf.json; Linux here, before the event loop can
    // answer a page request.
    #[cfg(target_os = "linux")]
    if let Some(window) = app.get_webview_window(SETTINGS_WINDOW)
        && let Err(error) = window.set_decorations(false)
    {
        tracing::warn!(%error, "settings window decorations could not be removed");
    }
    let persisted_binding = recording_binding(&settings);
    let StartedRegistrar { registrar, failure } =
        shortcut_registrar(app.handle().clone(), native_wayland, shortcut_events)?;
    apply_initial_window_policy(app, start_minimized);
    let backend = Arc::new(backend);
    let probe_app = app.handle().clone();
    let shortcuts = Arc::new(Shortcuts::new(
        registrar,
        persisted_binding,
        Arc::new(move || {
            probe_app
                .try_state::<Gateway>()
                .is_some_and(|state| state.shortcuts_enabled())
        }),
        failure,
    ));
    if let Some(error) = shortcuts.failure() {
        tracing::warn!("configured push-to-talk shortcut is unavailable");
        backend.state.publish_error(gateway::shortcut_error(error));
    }
    shortcuts.set_cancel_binding(cancel_binding(&settings));
    app.manage(Gateway::new(
        backend,
        shortcuts,
        autostart,
        Arc::clone(&activation),
        tray,
    ));
    Backend::spawn_retention_checks(Arc::clone(&app.state::<Gateway>().backend));
    let update_controller = updater::UpdateController::new(
        app.handle().clone(),
        Arc::clone(&app.state::<Gateway>().backend),
        updates_enabled,
    );
    app.manage(Arc::clone(&update_controller));
    if updates_enabled {
        updater::spawn_periodic_checks(update_controller);
    }
    // A second launch during setup asked for the window before there was a state.
    if activation
        .release_deferred_activation(&app.state::<Gateway>().backend.state.snapshot().phase)
    {
        show_settings_window(app.handle());
    }
    forward_events(app.handle().clone(), overlay, event_receiver)?;
    app.state::<Gateway>().backend.state.publish();
    spawn_shortcut_dispatch(app.handle().clone(), shortcut_receiver);
    Ok(())
}

fn spawn_shortcut_dispatch(
    app: tauri::AppHandle,
    mut receiver: tokio::sync::mpsc::UnboundedReceiver<ShortcutEvent>,
) {
    tauri::async_runtime::spawn(async move {
        while let Some(event) = receiver.recv().await {
            match event {
                ShortcutEvent::Transition { id, transition } => {
                    let state = app.state::<Gateway>();
                    if let Some(action) = state.shortcuts.input(id, transition) {
                        state.shortcut_action(action);
                    }
                }
                ShortcutEvent::CapabilityLost(error) => {
                    tracing::warn!(?error, "Wayland shortcut portal capability was lost");
                    // Invalidation waits for any in-flight native registration.
                    let lost_app = app.clone();
                    if let Err(join_error) = tauri::async_runtime::spawn_blocking(move || {
                        lost_app.state::<Gateway>().shortcut_capability_lost(error);
                    })
                    .await
                    {
                        tracing::error!(
                            error = %join_error,
                            "the lost shortcut capability was not handled"
                        );
                    }
                }
            }
        }
    });
}

fn on_window_event(window: &tauri::Window, event: &tauri::WindowEvent) {
    let tauri::WindowEvent::CloseRequested { api, .. } = event else {
        return;
    };
    let label = window.label();
    if ![SETTINGS_WINDOW, RECORDING_BAR_WINDOW].contains(&label) {
        return;
    }
    api.prevent_close();
    if label == SETTINGS_WINDOW
        && let (Ok(position), Ok(size)) = (window.outer_position(), window.inner_size())
    {
        let geometry = WindowGeometry {
            x: position.x,
            y: position.y,
            width: size.width,
            height: size.height,
        };
        // The save takes the settings lock and writes a file, so it
        // runs off the event loop; the window hides immediately.
        let app = window.app_handle().clone();
        tauri::async_runtime::spawn_blocking(move || {
            if let Some(state) = app.try_state::<Gateway>()
                && let Err(error) = state.backend.settings.save_window_geometry(geometry)
            {
                tracing::warn!(code = error.code, "window geometry could not be saved");
            }
        });
    }
    if let Err(error) = window.hide() {
        tracing::warn!(%error, label, "window could not be hidden");
    }
    // A hidden window cannot finish a shortcut capture, so global
    // shortcuts must not stay suspended on its behalf.
    let app = window.app_handle().clone();
    tauri::async_runtime::spawn(async move {
        let Some(state) = app.try_state::<Gateway>() else {
            return;
        };
        if let Err(error) = state.set_shortcut_capture(false).await {
            tracing::warn!(code = error.code, "shortcuts could not be resumed");
        }
    });
}

fn on_run_event(app: &tauri::AppHandle, event: tauri::RunEvent) {
    let tauri::RunEvent::ExitRequested { api, code, .. } = event else {
        return;
    };
    let state = app.state::<Gateway>();
    if state.shutdown_complete() {
        return;
    }
    api.prevent_exit();
    if !state.begin_shutdown() {
        return;
    }
    hide_recording_bar(app);
    hide_settings_window(app);
    let cleanup_app = app.clone();
    let watchdog_app = app.clone();
    let exit_code = code.unwrap_or(0);
    let _watchdog = std::thread::Builder::new()
        .name("whispernow-shutdown-watchdog".to_owned())
        .spawn(move || {
            std::thread::sleep(SHUTDOWN_WATCHDOG_TIMEOUT);
            if !watchdog_app.state::<Gateway>().shutdown_complete() {
                tracing::error!("shutdown deadline exceeded; forcing process exit");
                std::process::exit(1);
            }
        });
    tauri::async_runtime::spawn_blocking(move || {
        cleanup_app.state::<Gateway>().finish_shutdown();
        cleanup_app.exit(exit_code);
    });
}

#[cfg_attr(not(target_os = "linux"), expect(clippy::unnecessary_wraps))]
fn shortcut_registrar(
    app: tauri::AppHandle,
    native_wayland: bool,
    events: tokio::sync::mpsc::UnboundedSender<ShortcutEvent>,
) -> tauri::Result<StartedRegistrar> {
    #[cfg(target_os = "linux")]
    if native_wayland {
        let identity = portal_identity::PortalIdentity::new(
            app.config().identifier.clone(),
            app.package_info().name.clone(),
            &app.path().data_dir()?,
        );
        return Ok(wayland_shortcuts::WaylandShortcutRegistrar::spawn(
            identity, events,
        ));
    }
    let _ = (native_wayland, events);
    Ok(StartedRegistrar {
        registrar: Arc::new(TauriShortcutRegistrar::new(app)),
        failure: None,
    })
}

#[cfg(target_os = "linux")]
fn native_wayland_session() -> bool {
    wayland_shortcuts::native_wayland_session()
}

#[cfg(not(target_os = "linux"))]
const fn native_wayland_session() -> bool {
    false
}

/// GNOME gives a native Wayland window no way to stay above other apps or choose its position,
/// so the recording bar could not float. Windows use `XWayland` instead, chosen before GTK
/// starts; shortcuts and paste still go through the portals. An explicit `GDK_BACKEND` wins.
#[cfg(target_os = "linux")]
fn use_xwayland_windows() -> bool {
    if let Some(backend) = std::env::var_os("GDK_BACKEND") {
        return backend
            .to_str()
            .is_some_and(|backend| backend.starts_with("x11"));
    }
    if std::env::var_os("DISPLAY").is_none() {
        return false;
    }
    gdk::set_allowed_backends("x11");
    true
}

#[cfg(not(target_os = "linux"))]
const fn use_xwayland_windows() -> bool {
    false
}

/// Runs on the event-loop thread, so actions that can wait for processing or native shortcut
/// round trips run on a worker and report through state and error events.
fn handle_tray_action(app: &tauri::AppHandle, action: TrayAction) {
    match action {
        TrayAction::Settings => {
            show_settings_window(app);
        }
        TrayAction::CheckForUpdates => {
            let controller = Arc::clone(app.state::<Arc<updater::UpdateController>>().inner());
            tauri::async_runtime::spawn(async move {
                if let Err(error) = controller.check_manual().await {
                    tracing::debug!(code = error.code, "update check was not admitted");
                }
            });
        }
        TrayAction::CopyLastTranscript => {
            let app = app.clone();
            tauri::async_runtime::spawn(async move {
                app.state::<Gateway>()
                    .copy_last_transcript_from_tray()
                    .await;
            });
        }
        TrayAction::CancelRecording => {
            let app = app.clone();
            tauri::async_runtime::spawn(async move {
                app.state::<Gateway>().cancel_dictation().await;
            });
        }
        TrayAction::TogglePause => {
            let app = app.clone();
            tauri::async_runtime::spawn_blocking(move || toggle_pause(&app.state::<Gateway>()));
        }
        // The exit handler owns shutdown admission; a quit during shutdown is ignored there.
        TrayAction::Quit => app.exit(0),
    }
}

fn toggle_pause(state: &Gateway) {
    let paused = matches!(state.backend.state.lock().phase, services::Phase::Paused);
    let dictation = &state.backend.dictation;
    let changed = if paused {
        dictation.resume()
    } else {
        dictation.pause()
    };
    if let Err(error) = changed {
        tracing::warn!(
            code = error.code,
            paused,
            "the tray could not change dictation"
        );
        state.publish_error(&error);
    }
}

fn apply_initial_window_policy(app: &tauri::App, start_minimized: bool) {
    if start_minimized {
        hide_settings_window(app.handle());
    } else {
        show_settings_window(app.handle());
    }
    hide_recording_bar(app.handle());
}

/// Window changes are queued to the event-loop thread and never awaited. State
/// publication runs on worker threads, and blocking native window getters there
/// would wait on the event loop while it may itself wait on application state.
pub(crate) fn on_window(
    app: &tauri::AppHandle,
    label: &'static str,
    operation: impl FnOnce(&tauri::WebviewWindow) + Send + 'static,
) {
    let main_thread_app = app.clone();
    let scheduled = app.run_on_main_thread(move || {
        if let Some(window) = main_thread_app.get_webview_window(label) {
            operation(&window);
        }
    });
    if let Err(error) = scheduled {
        tracing::warn!(%error, label, "window change could not be scheduled");
    }
}

pub(crate) fn show_settings_window(app: &tauri::AppHandle) {
    on_window(app, SETTINGS_WINDOW, |window| {
        if let Err(error) = window.show() {
            tracing::warn!(%error, "the settings window could not be shown");
        } else if let Err(error) = window.set_focus() {
            tracing::warn!(%error, "the settings window could not be focused");
        }
    });
}

pub(crate) fn hide_settings_window(app: &tauri::AppHandle) {
    hide_window(app, SETTINGS_WINDOW);
}

pub(crate) fn hide_recording_bar(app: &tauri::AppHandle) {
    hide_window(app, RECORDING_BAR_WINDOW);
}

fn hide_window(app: &tauri::AppHandle, label: &'static str) {
    on_window(app, label, move |window| {
        if let Err(error) = window.hide() {
            tracing::warn!(%error, label, "window could not be hidden");
        }
    });
}

fn restore_settings_geometry(app: &tauri::App, geometry: WindowGeometry) {
    let Some(window) = app.get_webview_window(SETTINGS_WINDOW) else {
        return;
    };
    let monitors: Vec<_> = window
        .available_monitors()
        .unwrap_or_default()
        .iter()
        .map(overlay::monitor_area)
        .collect();
    if monitors.is_empty() {
        return;
    }
    let geometry = clamp_geometry_to_monitors(geometry, &monitors);
    if let Err(error) = window.set_size(tauri::PhysicalSize::new(geometry.width, geometry.height)) {
        tracing::warn!(%error, "the settings window size could not be restored");
    }
    if let Err(error) = window.set_position(tauri::PhysicalPosition::new(geometry.x, geometry.y)) {
        tracing::warn!(%error, "the settings window position could not be restored");
    }
}

fn clamp_geometry_to_monitors(
    mut geometry: WindowGeometry,
    monitors: &[overlay::MonitorArea],
) -> WindowGeometry {
    geometry.width = geometry.width.clamp(720, 3_840);
    geometry.height = geometry.height.clamp(520, 2_160);
    let Some(monitor) = monitors
        .iter()
        .find(|monitor| overlay::overlaps(&monitor.work_area, &geometry))
        .or_else(|| monitors.first())
    else {
        return geometry;
    };
    let work = monitor.work_area;
    geometry.width = geometry.width.min(work.width);
    geometry.height = geometry.height.min(work.height);
    let max_x = i64::from(work.x) + i64::from(work.width - geometry.width);
    let max_y = i64::from(work.y) + i64::from(work.height - geometry.height);
    geometry.x = i64::from(geometry.x)
        .clamp(i64::from(work.x), max_x)
        .try_into()
        .unwrap_or(work.x);
    geometry.y = i64::from(geometry.y)
        .clamp(i64::from(work.y), max_y)
        .try_into()
        .unwrap_or(work.y);
    geometry
}

#[cfg(test)]
mod tests;
