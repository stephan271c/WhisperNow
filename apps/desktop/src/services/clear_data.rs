use whispernow_domain::ClearDataCategory;
use whispernow_domain::ipc::{ClearDataItem, ClearDataResult, DataLocation, DataPlace};
use whispernow_domain::{AppError, ErrorCategory};

use super::{Backend, models::hold_cleanup_unloaded};

impl Backend {
    /// The existing data a clear would remove, including the credential store entry.
    pub(crate) fn data_locations(&self) -> Result<Vec<DataLocation>, AppError> {
        let locations = self
            .paths
            .data_locations()
            .map_err(|error| inspect_error(&error))?;
        Ok(locations
            .into_iter()
            .chain([credential_store_location()])
            .collect())
    }

    pub(crate) fn clear_data(&self) -> Result<ClearDataResult, AppError> {
        let _model_transition = self.models.transition();
        let _quiescent = self.dictation.begin_data_clear()?;
        let _cleanup_unloaded = hold_cleanup_unloaded(&self.local_cleanup)?;
        let mut saved = self.settings.lock();
        let plan = self
            .paths
            .clear_plan()
            .map_err(|error| inspect_error(&error))?;
        let history_item = self.history.clear_data(&plan);
        self.state.events().history_changed();
        let entries = plan.remove(&[ClearDataCategory::Settings, ClearDataCategory::Models]);
        let settings_file = DataPlace::Path(self.paths.settings_file());
        let reset_model_id = entries
            .iter()
            .filter(|entry| entry.location.place == settings_file)
            .all(|entry| entry.deleted)
            .then(|| {
                *saved = whispernow_domain::Settings::default();
                saved.selected_model_id().clone()
            });
        let restart_required = entries.iter().any(|entry| entry.deleted);
        drop(saved);
        let credential_item = ClearDataItem {
            location: credential_store_location(),
            deleted: self.settings.delete_api_keys(),
        };
        let credential_warning = (!credential_item.deleted).then(|| {
            AppError::new(
                ErrorCategory::General,
                "credential_store_unavailable",
                "The OS credential store is unavailable, so saved API keys were not removed from it",
            )
        });
        let result_items = entries
            .into_iter()
            .chain([history_item, credential_item])
            .collect();
        self.state.apply(|state| {
            if let Some(model_id) = reset_model_id {
                state.selected_model_id = model_id;
            }
            state.last_transcript = None;
            state.last_error = credential_warning;
            state.failed_download = None;
        });
        Ok(ClearDataResult {
            items: result_items,
            restart_required,
        })
    }
}

pub(super) fn credential_store_location() -> DataLocation {
    DataLocation {
        category: ClearDataCategory::Settings,
        place: DataPlace::CredentialStore,
    }
}

fn inspect_error(error: &std::io::Error) -> AppError {
    tracing::warn!(%error, "application data could not be inspected");
    AppError::new(
        ErrorCategory::General,
        "clear_data_inspect",
        "Application data could not be inspected",
    )
}
