//! Clearing application data: the storage targets, the OS credential store entry, and the
//! application-state repair that follows a clear.

use whispernow_api::{
    ApiErrorDto, ClearDataCategory, ClearDataRequest, ClearDataResult, ClearDataResultItemDto,
    DataLocationDto, ErrorCategoryDto,
};
use whispernow_domain::{ModelId, OverlayPosition, OverlayStyle};
use whispernow_platform::ClearDataError;

use super::NativeServices;

impl NativeServices {
    /// The existing data each category would remove, including the credential store entry.
    pub(crate) fn data_locations(&self) -> Result<Vec<DataLocationDto>, ApiErrorDto> {
        let locations = self.clear_data.locations().map_err(inspect_error)?;
        Ok(locations
            .into_iter()
            .map(|location| DataLocationDto {
                category: location.category,
                path: location.path.to_string_lossy().into_owned(),
            })
            .chain([DataLocationDto {
                category: ClearDataCategory::Settings,
                path: CREDENTIAL_STORE_CLEAR_TARGET.to_owned(),
            }])
            .collect())
    }

    /// Removes the chosen categories while nothing is dictating or downloading. The settings
    /// lock is held throughout, so no settings write lands between the deletion and the state
    /// repair that follows it.
    pub(crate) fn clear_data(
        &self,
        request: &ClearDataRequest,
    ) -> Result<ClearDataResult, ApiErrorDto> {
        let _model_transition = self.models.transition();
        let settings_lock = self.settings.transaction();
        {
            let state = self.state.lock();
            state.phase.require_idle()?;
            if !state.active_downloads.is_empty() {
                return Err(ApiErrorDto::new(
                    ErrorCategoryDto::General,
                    "application_busy",
                    "Wait for model downloads to finish before clearing data",
                ));
            }
        }
        let categories = &request.categories;
        let clears_settings = categories.contains(&ClearDataCategory::Settings);
        // The recognizer holds the model files open.
        self.dictation.unload_model()?;
        let clear_result = self
            .history
            .with_repository_closed(|| self.clear_data.clear(categories).map_err(inspect_error));
        self.publish_history_changed();
        let entries = clear_result.and_then(|result| result)?;
        let restart_required = entries.iter().any(|entry| {
            matches!(
                entry.category,
                ClearDataCategory::Settings | ClearDataCategory::Models
            ) && entry.outcome.is_ok()
        });
        // Keys are deleted after the files so a slow or unavailable store never
        // holds history closed or blocks them.
        let credential_item = clears_settings.then(|| ClearDataResultItemDto {
            category: ClearDataCategory::Settings,
            path: CREDENTIAL_STORE_CLEAR_TARGET.to_owned(),
            deleted: self.settings.delete_api_keys(),
        });
        let credential_warning = credential_item
            .as_ref()
            .filter(|item| !item.deleted)
            .map(|_| {
                ApiErrorDto::new(ErrorCategoryDto::General,
                    "credential_store_unavailable",
                    "The OS credential store is unavailable, so saved API keys were not removed from it",
                )
            });
        let result_items = entries
            .into_iter()
            .map(|entry| ClearDataResultItemDto {
                category: entry.category,
                path: entry.path.to_string_lossy().into_owned(),
                deleted: entry.outcome.is_ok(),
            })
            .chain(credential_item)
            .collect();
        self.state.apply(|state| {
            if restart_required {
                state.selected_model_id = ModelId::default();
            }
            if clears_settings {
                state.overlay_style = OverlayStyle::default();
                state.overlay_position = OverlayPosition::default();
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

/// Clear-data entry for the LLM API keys held outside the settings file.
pub(super) const CREDENTIAL_STORE_CLEAR_TARGET: &str = "OS credential store: saved LLM API keys";

fn inspect_error(_: ClearDataError) -> ApiErrorDto {
    ApiErrorDto::new(
        ErrorCategoryDto::General,
        "clear_data_inspect",
        "Application data could not be inspected",
    )
}
