use whispernow_domain::ClearDataCategory;
use whispernow_domain::ipc::{ClearDataItem, ClearDataResult, DataLocation, DataPlace};
use whispernow_domain::{AppError, ErrorCategory};

use super::{Backend, models::hold_cleanup_unloaded};

impl Backend {
    /// The existing data each category would remove, including the credential store entry.
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

    pub(crate) fn clear_data(
        &self,
        categories: &[ClearDataCategory],
    ) -> Result<ClearDataResult, AppError> {
        let _model_transition = self.models.transition();
        let _quiescent = self.dictation.begin_data_clear()?;
        let clears_models = categories.contains(&ClearDataCategory::Models);
        let _cleanup_unloaded = if clears_models {
            Some(hold_cleanup_unloaded(&self.local_cleanup)?)
        } else {
            None
        };
        let clears_settings = categories.contains(&ClearDataCategory::Settings);
        let mut settings_lock = clears_settings.then(|| self.settings.lock());
        let plan = self
            .paths
            .clear_plan(categories)
            .map_err(|error| inspect_error(&error))?;
        let history_item = categories.contains(&ClearDataCategory::History).then(|| {
            let item = self.history.clear_data(&plan);
            self.state.events().history_changed();
            item
        });
        let other_categories: Vec<_> = categories
            .iter()
            .copied()
            .filter(|category| *category != ClearDataCategory::History)
            .collect();
        let entries = plan.remove(&other_categories);
        let mut reset_model_id = None;
        if let Some(saved) = &mut settings_lock
            && entries
                .iter()
                .filter(|entry| entry.location.place == DataPlace::Path(self.paths.settings_file()))
                .all(|entry| entry.deleted)
        {
            **saved = whispernow_domain::Settings::default();
            reset_model_id = Some(saved.selected_model_id().clone());
        }
        let restart_required = entries.iter().any(|entry| {
            matches!(
                entry.location.category,
                ClearDataCategory::Settings | ClearDataCategory::Models
            ) && entry.deleted
        });
        drop(settings_lock);
        let credential_item = clears_settings.then(|| ClearDataItem {
            location: credential_store_location(),
            deleted: self.settings.delete_api_keys(),
        });
        let credential_warning = credential_item
            .as_ref()
            .filter(|item| !item.deleted)
            .map(|_| {
                AppError::new(ErrorCategory::General,
                    "credential_store_unavailable",
                    "The OS credential store is unavailable, so saved API keys were not removed from it",
                )
            });
        let result_items = entries
            .into_iter()
            .chain(history_item)
            .chain(credential_item)
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
