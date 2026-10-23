use serde::{Deserialize, Deserializer, Serialize, de::Error as _};

/// Opaque native input-device identity, never blank. Callers must not parse this value.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize)]
#[serde(transparent)]
pub struct AudioDeviceId(String);

impl AudioDeviceId {
    /// `None` for a blank identifier.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Option<Self> {
        let value = value.into();
        (!value.trim().is_empty()).then_some(Self(value))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for AudioDeviceId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::new(String::deserialize(deserializer)?)
            .ok_or_else(|| D::Error::custom("audio device identifier must not be blank"))
    }
}

#[cfg(test)]
mod tests {
    use super::AudioDeviceId;

    #[test]
    fn a_blank_device_id_is_refused_when_read() {
        for blank in [r#""""#, r#"" ""#, r#""\t\n""#] {
            assert!(
                serde_json::from_str::<AudioDeviceId>(blank).is_err(),
                "{blank}"
            );
        }
        let id: AudioDeviceId = serde_json::from_str(r#""usb-mic""#).expect("device id");
        assert_eq!(id.as_str(), "usb-mic");
    }
}
