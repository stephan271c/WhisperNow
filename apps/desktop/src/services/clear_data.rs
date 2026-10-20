use whispernow_domain::AppError;
use whispernow_domain::ClearDataCategory;
use whispernow_domain::ipc::{ClearDataItem, ClearDataResult, DataLocation, ErrorCategory};
use whispernow_platform::{CudaPackRemovalOutcome, ModelStoreError};

use super::{NativeServices, models::hold_cleanup_unloaded};

impl NativeServices {
    /// The existing data each category would remove, including the credential store entry.
    pub(crate) fn data_locations(&self) -> Result<Vec<DataLocation>, AppError> {
        let locations = self.paths.data_locations().map_err(inspect_error)?;
        Ok(locations
            .into_iter()
            .chain([DataLocation {
                category: ClearDataCategory::Settings,
                path: CREDENTIAL_STORE_CLEAR_TARGET.into(),
            }])
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
        let cuda_path = self.paths.cuda_pack_dir();
        let plan = self.paths.clear_plan(categories).map_err(inspect_error)?;
        let history_item = categories.contains(&ClearDataCategory::History).then(|| {
            let item = self.history.clear_data(&plan);
            self.publish_history_changed();
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
                .filter(|entry| {
                    entry.category == ClearDataCategory::Settings
                        && entry.path == self.paths.settings_file().to_string_lossy()
                })
                .all(|entry| entry.deleted)
        {
            **saved = whispernow_domain::Settings::default();
            reset_model_id = Some(saved.model_id.clone());
        }
        let mut restart_required = entries.iter().any(|entry| {
            matches!(
                entry.category,
                ClearDataCategory::Settings | ClearDataCategory::Models
            ) && entry.deleted
        });
        let cuda_item = if clears_models {
            let removal = self.models.model_store().remove_cuda_pack();
            if removal == Err(ModelStoreError::NotInstalled) {
                None
            } else {
                restart_required |= removal.is_ok();
                Some(ClearDataItem {
                    category: ClearDataCategory::Models,
                    path: cuda_path.to_string_lossy().into_owned(),
                    deleted: removal == Ok(CudaPackRemovalOutcome::Removed),
                })
            }
        } else {
            None
        };
        drop(settings_lock);
        let credential_item = clears_settings.then(|| ClearDataItem {
            category: ClearDataCategory::Settings,
            path: CREDENTIAL_STORE_CLEAR_TARGET.to_owned(),
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
            .chain(cuda_item)
            .chain(credential_item)
            .collect();
        self.state.apply(|state| {
            if let Some(model_id) = reset_model_id {
                state.selected_model_id = model_id;
            }
            state.last_transcript = None;
            state.last_error = credential_warning;
            state.download_failures = 0;
        });
        Ok(ClearDataResult {
            items: result_items,
            restart_required,
        })
    }
}

pub(super) const CREDENTIAL_STORE_CLEAR_TARGET: &str = "OS credential store: saved LLM API keys";

fn inspect_error(_: std::io::Error) -> AppError {
    AppError::new(
        ErrorCategory::General,
        "clear_data_inspect",
        "Application data could not be inspected",
    )
}
