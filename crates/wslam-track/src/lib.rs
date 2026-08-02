//! # wslam-track — L3
//!
//! Sparse feature tracking (KLT) and PnP against the active local map
//! (spec.md §4 L3). Runs every frame, on the critical path, and produces pose
//! **up to scale** — L3 makes no metric claim, which is why spec.md §6 L3
//! measures it as "ATE after Sim(3) alignment".
//!
//! The frontend is a pipeline of six stages, one module each:
//!
//! | Module | Stage |
//! |---|---|
//! | [`pyramid`] | Gaussian/box pyramid with per-level intrinsics |
//! | [`corners`] | Shi-Tomasi response, grid-bucketed selection, sub-pixel |
//! | [`klt`] | Pyramidal inverse-compositional flow, forward-backward checked |
//! | [`pnp`] | P3P/EPnP inside seeded RANSAC |
//! | [`motion_ba`] | Motion-only bundle adjustment, Huber |
//! | [`triangulate`] | DLT plus cheirality, for new landmarks |
//! | [`init`] | Two-view homography/essential bootstrap |
//!
//! [`tracker`] composes them and owns the [`wslam_core::TrackingState`] machine.
//!
//! ## Determinism
//!
//! Every RNG here is a [`wslam_core::DeterministicRng`] built from
//! [`TrackConfig::seed`], and no estimation path reads a wall clock. The
//! optional [`wslam_core::HostClock`] a [`Tracker`] may hold feeds
//! [`StageTimings`] and nothing else, so the same frames and the same seed
//! produce a bit-identical trajectory whether or not one is installed
//! (spec.md §6).

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod corners;
pub mod init;
pub mod klt;
pub mod local_ba;
pub mod motion_ba;
pub mod pnp;
pub mod pyramid;
pub mod tracker;
pub mod triangulate;

pub use tracker::{
    Backend, Feature, FeatureState, LocalLandmark, LocalMap, StageTimings, TrackConfig,
    FailureCounts, TrackOutcome, Tracker,
};
#[cfg(feature = "gpu")]
pub use wslam_gpu::GpuContext;
