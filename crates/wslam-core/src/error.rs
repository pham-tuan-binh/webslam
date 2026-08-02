//! Error type shared across the workspace.

/// Everything that can go wrong in web-slam.
///
/// One enum rather than per-crate errors: the wasm boundary has to flatten them
/// into strings anyway, and a caller in JavaScript cannot match on a nested
/// error hierarchy. Variants stay coarse and their messages stay specific.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A configuration value is impossible or self-contradictory.
    #[error("invalid configuration: {0}")]
    Config(String),

    /// A required sensor tier is unavailable.
    ///
    /// Raised, for example, by `ScaleSource::inertial()` when L0 is not
    /// compiled in — spec.md §3 specifies it *"throws if L0 unavailable"*
    /// rather than silently degrading.
    #[error("sensor tier {required} unavailable: {reason}")]
    SensorTier {
        /// Tier the caller asked for.
        required: u8,
        /// Why it cannot be provided.
        reason: String,
    },

    /// The GPU could not be acquired or a kernel failed.
    #[error("gpu: {0}")]
    Gpu(String),

    /// Not enough data to answer yet. Distinct from a hard failure: the caller
    /// should keep feeding frames.
    #[error("insufficient data: {0}")]
    Insufficient(String),

    /// A numerical solve failed to converge or hit a singular system.
    #[error("numerical failure in {stage}: {detail}")]
    Numerical {
        /// Which solve.
        stage: &'static str,
        /// What went wrong.
        detail: String,
    },

    /// A serialised map could not be read.
    #[error("map format: {0}")]
    MapFormat(String),

    /// A serialised map is a version this build does not understand.
    #[error("map format version {found} unsupported (this build reads {supported})")]
    MapVersion {
        /// Version found in the header.
        found: u16,
        /// Version this build supports.
        supported: u16,
    },

    /// A dataset or asset could not be read.
    #[error("io: {0}")]
    Io(String),

    /// Tracking is lost and no map is available to recover into.
    #[error("tracking lost: {0}")]
    TrackingLost(String),
}

impl Error {
    /// Convenience for the common numerical case.
    #[must_use]
    pub fn numerical(stage: &'static str, detail: impl Into<String>) -> Self {
        Error::Numerical {
            stage,
            detail: detail.into(),
        }
    }

    /// Convenience for the common insufficient-data case.
    #[must_use]
    pub fn insufficient(detail: impl Into<String>) -> Self {
        Error::Insufficient(detail.into())
    }

    /// Whether the caller should retry with more data rather than give up.
    #[must_use]
    pub fn is_transient(&self) -> bool {
        matches!(self, Error::Insufficient(_) | Error::TrackingLost(_))
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e.to_string())
    }
}

/// Workspace result alias.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transient_errors_are_distinguishable() {
        assert!(Error::insufficient("need 10 more frames").is_transient());
        assert!(!Error::numerical("pnp", "singular").is_transient());
        assert!(!Error::Config("bad".into()).is_transient());
    }

    #[test]
    fn messages_name_the_stage() {
        let e = Error::numerical("homography-decomposition", "rank deficient");
        assert!(e.to_string().contains("homography-decomposition"));
        assert!(e.to_string().contains("rank deficient"));
    }

    #[test]
    fn map_version_error_reports_both_versions() {
        let e = Error::MapVersion {
            found: 9,
            supported: crate::MAP_FORMAT_VERSION,
        };
        assert!(e.to_string().contains('9'));
    }
}
