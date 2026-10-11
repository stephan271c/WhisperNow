use whispernow_domain::AppError;
use whispernow_domain::ClearDataCategory;
use whispernow_domain::ModelId;
use whispernow_domain::ipc::{ClearDataItem, ClearDataResult, DataLocation, ErrorCategory};
use whispernow_platform::{ClearDataError, CudaPackRemovalOutcome, Llama, ModelStoreError};

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
        let settings_lock = clears_settings.then(|| self.settings.transaction());
        let cuda_path = self.paths.cuda_pack_dir();
        let clear_files = || self.paths.clear_data(categories).map_err(inspect_error);
        let entries = if categories.contains(&ClearDataCategory::History) {
            let result = self.history.with_repository_closed(clear_files);
            self.publish_history_changed();
            result.and_then(|result| result)?
        } else {
            clear_files()?
        };
        let mut restart_required = entries.iter().any(|entry| {
            matches!(
                entry.category,
                ClearDataCategory::Settings | ClearDataCategory::Models
            ) && entry.outcome.is_ok()
        });
        let cuda_item = if clears_models {
            let removal = self
                .models
                .model_store()
                .remove_cuda_pack(Llama::cuda_loaded());
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
        // Keys are deleted after the files so a slow or unavailable store never
        // holds history closed or blocks them.
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
            .map(|entry| ClearDataItem {
                category: entry.category,
                path: entry.path.to_string_lossy().into_owned(),
                deleted: entry.outcome.is_ok(),
            })
            .chain(cuda_item)
            .chain(credential_item)
            .collect();
        self.state.apply(|state| {
            if restart_required {
                state.selected_model_id = ModelId::default();
            }
            state.last_transcript = None;
            state.last_error = credential_warning;
            state.download_failures = 0;
        });
        drop(settings_lock);
        Ok(ClearDataResult {
            items: result_items,
            restart_required,
        })
    }
}

pub(super) const CREDENTIAL_STORE_CLEAR_TARGET: &str = "OS credential store: saved LLM API keys";

fn inspect_error(_: ClearDataError) -> AppError {
    AppError::new(
        ErrorCategory::General,
        "clear_data_inspect",
        "Application data could not be inspected",
    )
}
