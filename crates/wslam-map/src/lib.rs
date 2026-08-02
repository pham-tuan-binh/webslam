//! # wslam-map — L4: keyframe map, place recognition, pose graph
//!
//! spec.md §4 L4 stages this layer by value-to-cost:
//!
//! - **(a) Keyframe map + relocalization — mandatory.** *"Tracking loss is not
//!   an edge case on a phone; it is a routine event from occlusion, pocketing,
//!   or fast motion. Without relocalization the session and the user's anchor
//!   are destroyed."* [`descriptor`], [`vocabulary`], [`keyframe`], [`db`] and
//!   [`reloc`].
//! - **(b) Loop closure + pose graph.** [`posegraph`].
//! - **(c) Map persistence.** [`serialize`] — *"this is what makes the map a
//!   ScaleSource"*, and the reason the scale anchor lives in the file header
//!   rather than somewhere in the payload.
//!
//! Two rules run through the whole crate.
//!
//! **No pose escapes without geometric verification.** [`Relocalizer::query`]
//! returns [`Candidate`]s that carry no pose at all; the only type carrying one
//! is [`Verified`], which no other crate can construct. spec.md §5: *"a
//! false-positive loop closure corrupts the map irrecoverably and is worse than
//! no loop closure at all."*
//!
//! **Nothing here reads a wall clock and no RNG is unseeded.** Vocabulary
//! training, RANSAC and the BRIEF sampling pattern all draw from
//! [`wslam_core::DeterministicRng`]; time enters only as
//! [`wslam_core::Timestamp`] on a keyframe. That is what makes a map
//! bit-reproducible across a replay (spec.md §6).

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod db;
pub mod descriptor;
pub mod keyframe;
pub mod posegraph;
pub mod reloc;
pub mod serialize;
pub mod vocabulary;

pub use db::{CullPolicy, MapDb};
pub use descriptor::{describe, fast_keypoints, BinaryDescriptor};
pub use keyframe::{Keyframe, KeyframeId, Landmark, LandmarkId};
pub use posegraph::{Edge, PoseGraph, SolverConfig, SolverReport};
pub use reloc::{Candidate, Rejection, RelocConfig, Relocalizer, Verified};
pub use serialize::{deserialize_map, serialize_map};
pub use vocabulary::{BowVector, Vocabulary};

/// Synthetic scenes shared by the test modules.
///
/// A wall of landmarks at a known depth, a camera sweeping past it, and a fixed
/// descriptor per landmark. Everything is a deterministic function of a seed,
/// so a failing assertion is reproducible from the test name alone.
///
/// Descriptor *appearance* noise is deliberately absent: `descriptor.rs`
/// exercises that directly, and letting it leak in here would turn a
/// relocalization failure into a question about the vocabulary's quantisation
/// margin. What these scenes do model is pixel noise, viewpoint change, and —
/// in [`Scene::shuffled_appearance`] — the adversarial case of a *different*
/// place that looks identical.
#[cfg(test)]
pub(crate) mod synth {
    use crate::db::MapDb;
    use crate::descriptor::BinaryDescriptor;
    use crate::keyframe::{Keyframe, KeyframeId, Landmark, LandmarkId};
    use crate::vocabulary::Vocabulary;
    use std::sync::Arc;
    use wslam_core::{CameraIntrinsics, DeterministicRng, Se3, So3, Timestamp, Vec2, Vec3};

    /// A wall of landmarks and a trajectory past it.
    pub(crate) struct Scene {
        /// True landmark positions in world coordinates.
        pub landmarks: Vec<Vec3>,
        /// One descriptor per landmark, index-parallel with `landmarks`.
        pub descriptors: Vec<BinaryDescriptor>,
        /// Ground-truth `T_world_camera` per keyframe.
        pub poses: Vec<Se3>,
        /// Camera model.
        pub intrinsics: CameraIntrinsics,
        /// Vocabulary trained on `descriptors`.
        pub vocabulary: Arc<Vocabulary>,
    }

    /// What a camera sees from one pose.
    pub(crate) struct Observation {
        pub keypoints: Vec<Vec2>,
        pub descriptors: Vec<BinaryDescriptor>,
        /// Index into [`Scene::landmarks`] for each observed feature.
        pub landmark_index: Vec<usize>,
    }

    fn random_descriptor(rng: &mut DeterministicRng) -> BinaryDescriptor {
        let mut d = [0u8; 32];
        for b in d.iter_mut() {
            *b = rng.below(256) as u8;
        }
        BinaryDescriptor(d)
    }

    impl Scene {
        /// A wall of landmarks with `n_poses` camera stations sweeping past it.
        pub(crate) fn new(seed: u64, n_poses: usize) -> Self {
            let mut rng = DeterministicRng::new("synth-scene", seed);
            let intrinsics = CameraIntrinsics::from_focal(500.0, 640, 480);

            // Depth varies with x and y so the point cloud is not planar — a
            // plane is a degenerate configuration for pose estimation and would
            // make these tests easier than reality.
            let mut landmarks = Vec::new();
            let mut descriptors = Vec::new();
            for ix in 0..49 {
                let x = -6.0 + ix as f64 * 0.25;
                for iy in 0..5 {
                    let y = -1.0 + iy as f64 * 0.5;
                    let z = 3.0 + 0.6 * (3.0 * x).sin() + 0.3 * (5.0 * y).cos();
                    landmarks.push(Vec3::new(x, y, z));
                    descriptors.push(random_descriptor(&mut rng));
                }
            }

            let poses = (0..n_poses)
                .map(|i| {
                    let t = if n_poses > 1 {
                        -4.0 + 8.0 * i as f64 / (n_poses - 1) as f64
                    } else {
                        0.0
                    };
                    // Yaw and bob, so consecutive views are not pure translation.
                    let yaw = 0.06 * (t * 0.7).sin();
                    let pitch = 0.03 * (t * 1.1).cos();
                    Se3::new(
                        So3::exp(&Vec3::new(pitch, yaw, 0.0)),
                        Vec3::new(t, 0.05 * (t * 1.3).sin(), 0.0),
                    )
                })
                .collect();

            let vocabulary = Arc::new(Vocabulary::train(&descriptors, 6, 4, &mut rng));
            Scene {
                landmarks,
                descriptors,
                poses,
                intrinsics,
                vocabulary,
            }
        }

        /// The same geometry with the descriptors permuted between landmarks.
        ///
        /// The adversarial false positive: a place that produces a *high*
        /// bag-of-words score and descriptor matches that all succeed, but whose
        /// 2D-3D correspondences are geometric nonsense. Only PnP can reject it.
        pub(crate) fn shuffled_appearance(&self, seed: u64) -> Scene {
            let mut rng = DeterministicRng::new("synth-shuffle", seed);
            let mut descriptors = self.descriptors.clone();
            rng.shuffle(&mut descriptors);
            Scene {
                landmarks: self.landmarks.clone(),
                descriptors,
                poses: self.poses.clone(),
                intrinsics: self.intrinsics,
                vocabulary: Arc::clone(&self.vocabulary),
            }
        }
    }

    fn observe_inner(
        scene: &Scene,
        pose: Se3,
        noise_px: f64,
        rng: Option<&mut DeterministicRng>,
    ) -> Observation {
        let inv = pose.inverse();
        let mut obs = Observation {
            keypoints: Vec::new(),
            descriptors: Vec::new(),
            landmark_index: Vec::new(),
        };
        let mut rng = rng;
        for (i, p) in scene.landmarks.iter().enumerate() {
            let Some(px) = scene.intrinsics.project(&inv.act(p)) else {
                continue;
            };
            if !scene.intrinsics.contains(px, 2.0) {
                continue;
            }
            let px = match (noise_px > 0.0, rng.as_deref_mut()) {
                (true, Some(r)) => px + Vec2::new(r.normal() * noise_px, r.normal() * noise_px),
                _ => px,
            };
            obs.keypoints.push(px);
            obs.descriptors.push(scene.descriptors[i]);
            obs.landmark_index.push(i);
        }
        obs
    }

    /// Observe with sub-pixel measurement noise — the query side.
    pub(crate) fn observe(scene: &Scene, pose: Se3, rng: &mut DeterministicRng) -> Observation {
        observe_inner(scene, pose, 0.3, Some(rng))
    }

    /// Observe exactly — the map side, standing in for a converged bundle
    /// adjustment.
    pub(crate) fn observe_noiseless(scene: &Scene, pose: Se3) -> Observation {
        observe_inner(scene, pose, 0.0, None)
    }

    /// Insert one keyframe observing `pose`, creating landmarks as needed.
    pub(crate) fn add_keyframe(db: &mut MapDb, scene: &Scene, pose: Se3) -> KeyframeId {
        let obs = observe_noiseless(scene, pose);
        let index = db.next_keyframe_id();
        let kf = Keyframe::new(
            KeyframeId::UNSET,
            Timestamp::from_seconds(index as f64 * 0.2),
            pose,
            obs.keypoints,
            obs.descriptors,
            scene.intrinsics,
            &scene.vocabulary,
        );
        let id = db.insert_keyframe(kf);
        for (feature, &li) in obs.landmark_index.iter().enumerate() {
            let lid = LandmarkId(li as u64);
            if db.landmark(lid).is_none() {
                db.insert_landmark(Landmark {
                    id: lid,
                    position: scene.landmarks[li],
                    descriptor: scene.descriptors[li],
                    observations: Vec::new(),
                });
            }
            db.link_observation(id, feature, lid);
        }
        id
    }

    /// One pass along the trajectory, with a small pose perturbation so a
    /// revisit is a genuine revisit rather than a replay of identical poses.
    pub(crate) fn append_lap(db: &mut MapDb, scene: &Scene, rng: &mut DeterministicRng) {
        for pose in &scene.poses {
            let jitter = wslam_core::Vec6::from_iterator(
                (0..6).map(|i| rng.normal() * if i < 3 { 0.02 } else { 0.005 }),
            );
            add_keyframe(db, scene, pose.plus(&jitter));
        }
    }

    /// A single sweep: `n_keyframes` stations, each observing what it can see.
    pub(crate) fn corridor_map(n_keyframes: usize, seed: u64) -> (MapDb, Scene) {
        let scene = Scene::new(seed, n_keyframes);
        let mut db = MapDb::new(Arc::clone(&scene.vocabulary));
        for pose in &scene.poses {
            add_keyframe(&mut db, &scene, *pose);
        }
        (db, scene)
    }

    /// `laps` passes over the same `per_lap` stations — the redundancy culling
    /// is meant to exploit.
    pub(crate) fn revisit_map(laps: usize, per_lap: usize, seed: u64) -> (MapDb, Scene) {
        let scene = Scene::new(seed, per_lap);
        let mut db = MapDb::new(Arc::clone(&scene.vocabulary));
        let mut rng = DeterministicRng::new("synth-revisit", seed ^ 0x5EED);
        for _ in 0..laps {
            append_lap(&mut db, &scene, &mut rng);
        }
        (db, scene)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn the_fixture_produces_a_map_worth_testing_against() {
            let (db, scene) = corridor_map(12, 1);
            assert_eq!(db.keyframe_count(), 12);
            assert!(db.landmark_count() > 100, "{}", db.landmark_count());
            assert!(scene.vocabulary.word_count() > 50);
            for kf in db.keyframes() {
                assert!(kf.is_consistent());
                assert!(kf.len() > 40, "{} sees only {} features", kf.id, kf.len());
                assert_eq!(kf.matched_count(), kf.len());
                assert!(!kf.bow.is_empty());
            }
            // Consecutive keyframes must overlap but not coincide, or the
            // relocalization tests would be trivial.
            let a = db.keyframe(KeyframeId(4)).unwrap();
            let b = db.keyframe(KeyframeId(5)).unwrap();
            let s = a.bow.score(&b.bow);
            assert!(s > 0.2 && s < 0.95, "neighbour score {s}");
            let far = db.keyframe(KeyframeId(11)).unwrap();
            assert!(a.bow.score(&far.bow) < 0.05);
        }

        #[test]
        fn a_shuffled_scene_looks_the_same_but_is_not() {
            let base = Scene::new(2, 4);
            let other = base.shuffled_appearance(9);
            assert_eq!(base.landmarks, other.landmarks);
            assert_ne!(base.descriptors, other.descriptors);
            let a = observe_noiseless(&base, base.poses[0]);
            let b = observe_noiseless(&other, other.poses[0]);
            assert_eq!(a.keypoints.len(), b.keypoints.len());
        }

        #[test]
        fn observation_noise_is_small_and_seeded() {
            let scene = Scene::new(3, 2);
            let clean = observe_noiseless(&scene, scene.poses[0]);
            let noisy = observe(&scene, scene.poses[0], &mut DeterministicRng::new("t", 1));
            let again = observe(&scene, scene.poses[0], &mut DeterministicRng::new("t", 1));
            assert_eq!(clean.landmark_index, noisy.landmark_index);
            assert_eq!(noisy.keypoints, again.keypoints);
            let max = clean
                .keypoints
                .iter()
                .zip(noisy.keypoints.iter())
                .map(|(a, b)| (a - b).norm())
                .fold(0.0f64, f64::max);
            assert!(max > 0.0 && max < 3.0, "noise magnitude {max}");
        }
    }
}
