//! Keyframes and landmarks — the map's storage unit.
//!
//! spec.md §1 is explicit that the map is *"sparse keyframes and landmarks,
//! nothing renderable"*. A keyframe is therefore a pose, the 2-D features
//! observed at it, their descriptors, the landmark each feature was matched to,
//! and the bag-of-words vector used to find the keyframe again.
//!
//! `keypoints`, `descriptors` and `landmarks` are **index-parallel**. Every
//! producer must keep them the same length; [`Keyframe::is_consistent`] is the
//! assertion, and the deserialiser enforces it on load.

use crate::descriptor::BinaryDescriptor;
use crate::vocabulary::{BowVector, Vocabulary};
use wslam_core::{CameraIntrinsics, Se3, Timestamp, Vec2, Vec3};

/// Stable identifier for a keyframe within one map.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct KeyframeId(pub u64);

impl KeyframeId {
    /// Sentinel meaning "the database should assign an id". Chosen as
    /// `u64::MAX` so it can never collide with a real, monotonically assigned
    /// id.
    pub const UNSET: KeyframeId = KeyframeId(u64::MAX);

    /// Whether this id still needs assigning.
    #[must_use]
    pub fn is_unset(self) -> bool {
        self == KeyframeId::UNSET
    }
}

impl std::fmt::Display for KeyframeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "kf{}", self.0)
    }
}

/// Stable identifier for a landmark within one map.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct LandmarkId(pub u64);

impl LandmarkId {
    /// Sentinel meaning "the database should assign an id".
    pub const UNSET: LandmarkId = LandmarkId(u64::MAX);

    /// Whether this id still needs assigning.
    #[must_use]
    pub fn is_unset(self) -> bool {
        self == LandmarkId::UNSET
    }
}

impl std::fmt::Display for LandmarkId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "lm{}", self.0)
    }
}

/// A keyframe: one anchored view of the scene.
#[derive(Debug, Clone, PartialEq)]
pub struct Keyframe {
    /// Identifier, unique within the map.
    pub id: KeyframeId,
    /// Capture time in the unified timebase.
    pub timestamp: Timestamp,
    /// `T_world_camera`. Up to scale unless the map carries a metric anchor.
    pub pose: Se3,
    /// Feature locations in pixels, undistorted or not according to the
    /// producer's convention — [`Keyframe::intrinsics`] says which model
    /// applies.
    pub keypoints: Vec<Vec2>,
    /// One descriptor per keypoint.
    pub descriptors: Vec<BinaryDescriptor>,
    /// The landmark each keypoint was matched to, if any.
    pub landmarks: Vec<Option<LandmarkId>>,
    /// Bag-of-words vector over [`Keyframe::descriptors`].
    pub bow: BowVector,
    /// Intrinsics in force when this keyframe was captured. Per-keyframe rather
    /// than per-map because L2 refines focal length online, so early and late
    /// keyframes in a session genuinely differ.
    pub intrinsics: CameraIntrinsics,
}

impl Keyframe {
    /// Build a keyframe, computing its bag-of-words vector from `vocabulary`
    /// and leaving every feature unmatched.
    ///
    /// `id` may be [`KeyframeId::UNSET`], in which case
    /// [`crate::MapDb::insert_keyframe`] assigns one.
    #[must_use]
    pub fn new(
        id: KeyframeId,
        timestamp: Timestamp,
        pose: Se3,
        keypoints: Vec<Vec2>,
        descriptors: Vec<BinaryDescriptor>,
        intrinsics: CameraIntrinsics,
        vocabulary: &Vocabulary,
    ) -> Self {
        let bow = vocabulary.transform(&descriptors);
        let n = keypoints.len().min(descriptors.len());
        Keyframe {
            id,
            timestamp,
            pose,
            keypoints,
            descriptors,
            landmarks: vec![None; n],
            bow,
            intrinsics,
        }
    }

    /// Number of features.
    #[must_use]
    pub fn len(&self) -> usize {
        self.keypoints.len()
    }

    /// Whether the keyframe has no features at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keypoints.is_empty()
    }

    /// Whether the three parallel arrays agree in length.
    #[must_use]
    pub fn is_consistent(&self) -> bool {
        self.keypoints.len() == self.descriptors.len()
            && self.keypoints.len() == self.landmarks.len()
    }

    /// The distinct landmarks this keyframe observes, ascending.
    #[must_use]
    pub fn observed_landmarks(&self) -> Vec<LandmarkId> {
        let mut ids: Vec<LandmarkId> = self.landmarks.iter().flatten().copied().collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// How many features are matched to a landmark.
    #[must_use]
    pub fn matched_count(&self) -> usize {
        self.landmarks.iter().filter(|l| l.is_some()).count()
    }

    /// Camera centre in world coordinates.
    #[must_use]
    pub fn position(&self) -> Vec3 {
        self.pose.translation()
    }

    /// Approximate heap footprint in bytes.
    ///
    /// Computed from lengths rather than allocator capacities so that the
    /// number is a deterministic function of the map contents — spec.md §6 L4
    /// wants MB/min *measured*, and a metric that jitters with `Vec` growth
    /// policy cannot be regression-tested.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.keypoints.len() * std::mem::size_of::<Vec2>()
            + self.descriptors.len() * std::mem::size_of::<BinaryDescriptor>()
            + self.landmarks.len() * std::mem::size_of::<Option<LandmarkId>>()
            + self.bow.memory_bytes()
    }
}

/// A landmark: one 3-D point, with the keyframes that see it.
#[derive(Debug, Clone, PartialEq)]
pub struct Landmark {
    /// Identifier, unique within the map.
    pub id: LandmarkId,
    /// Position in world coordinates. Metres iff the map's scale anchor is
    /// metric.
    pub position: Vec3,
    /// Representative descriptor — the Hamming median over its observations,
    /// which is what [`Landmark::update_descriptor`] maintains.
    pub descriptor: BinaryDescriptor,
    /// Keyframes observing this landmark, ascending and deduplicated.
    pub observations: Vec<KeyframeId>,
}

impl Landmark {
    /// A landmark with a single observation.
    #[must_use]
    pub fn new(
        id: LandmarkId,
        position: Vec3,
        descriptor: BinaryDescriptor,
        observed_by: KeyframeId,
    ) -> Self {
        Landmark {
            id,
            position,
            descriptor,
            observations: vec![observed_by],
        }
    }

    /// Record an observation. Idempotent, and keeps the list sorted so that map
    /// contents are independent of insertion order.
    pub fn observe(&mut self, keyframe: KeyframeId) {
        if let Err(pos) = self.observations.binary_search(&keyframe) {
            self.observations.insert(pos, keyframe);
        }
    }

    /// Drop an observation. Returns whether one was removed.
    pub fn forget(&mut self, keyframe: KeyframeId) -> bool {
        match self.observations.binary_search(&keyframe) {
            Ok(pos) => {
                self.observations.remove(pos);
                true
            }
            Err(_) => false,
        }
    }

    /// How many keyframes see this landmark.
    #[must_use]
    pub fn observation_count(&self) -> usize {
        self.observations.len()
    }

    /// Whether no keyframe sees this landmark any more — it should be culled.
    #[must_use]
    pub fn is_orphaned(&self) -> bool {
        self.observations.is_empty()
    }

    /// Recompute the representative descriptor as the Hamming median of the
    /// supplied per-observation descriptors.
    ///
    /// ORB-SLAM2 picks the observation minimising summed distance to the
    /// others; the bitwise majority is the exact minimiser over the whole
    /// space, costs the same, and is what the vocabulary clusters with.
    pub fn update_descriptor(&mut self, observed: &[BinaryDescriptor]) {
        if !observed.is_empty() {
            self.descriptor = BinaryDescriptor::majority(observed);
        }
    }

    /// Approximate heap footprint in bytes.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.observations.len() * std::mem::size_of::<KeyframeId>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wslam_core::{DeterministicRng, So3};

    fn descriptor(seed: u8) -> BinaryDescriptor {
        BinaryDescriptor([seed; 32])
    }

    #[test]
    fn unset_ids_are_distinguishable_from_real_ones() {
        assert!(KeyframeId::UNSET.is_unset());
        assert!(!KeyframeId(0).is_unset());
        assert!(!KeyframeId(1_000_000).is_unset());
        assert!(LandmarkId::UNSET.is_unset());
        assert!(!LandmarkId(0).is_unset());
        assert_eq!(KeyframeId(7).to_string(), "kf7");
        assert_eq!(LandmarkId(7).to_string(), "lm7");
    }

    #[test]
    fn new_keyframe_is_consistent_and_unmatched() {
        let mut rng = DeterministicRng::new("t", 1);
        let descs: Vec<BinaryDescriptor> = (0..8).map(|i| descriptor(i * 17)).collect();
        let vocab = Vocabulary::train(&descs, 2, 3, &mut rng);
        let kps: Vec<Vec2> = (0..8)
            .map(|i| Vec2::new(i as f64, 2.0 * i as f64))
            .collect();
        let kf = Keyframe::new(
            KeyframeId::UNSET,
            Timestamp::from_seconds(1.5),
            Se3::identity(),
            kps,
            descs.clone(),
            CameraIntrinsics::from_focal(500.0, 640, 480),
            &vocab,
        );
        assert!(kf.is_consistent());
        assert_eq!(kf.len(), 8);
        assert!(!kf.is_empty());
        assert_eq!(kf.matched_count(), 0);
        assert!(kf.observed_landmarks().is_empty());
        assert_eq!(kf.bow, vocab.transform(&descs));
    }

    #[test]
    fn observed_landmarks_are_sorted_and_deduplicated() {
        let mut kf = Keyframe::new(
            KeyframeId(3),
            Timestamp::ZERO,
            Se3::identity(),
            vec![Vec2::zeros(); 5],
            vec![descriptor(1); 5],
            CameraIntrinsics::from_focal(500.0, 640, 480),
            &Vocabulary::empty(),
        );
        kf.landmarks = vec![
            Some(LandmarkId(9)),
            None,
            Some(LandmarkId(2)),
            Some(LandmarkId(9)),
            Some(LandmarkId(5)),
        ];
        assert_eq!(
            kf.observed_landmarks(),
            vec![LandmarkId(2), LandmarkId(5), LandmarkId(9)]
        );
        assert_eq!(kf.matched_count(), 4);
    }

    #[test]
    fn keyframe_position_is_the_camera_centre() {
        let pose = Se3::new(
            So3::exp(&Vec3::new(0.2, -0.1, 0.4)),
            Vec3::new(1.0, 2.0, 3.0),
        );
        let kf = Keyframe::new(
            KeyframeId(0),
            Timestamp::ZERO,
            pose,
            vec![],
            vec![],
            CameraIntrinsics::from_focal(500.0, 640, 480),
            &Vocabulary::empty(),
        );
        assert_eq!(kf.position(), Vec3::new(1.0, 2.0, 3.0));
        assert!(kf.is_empty());
        assert!(kf.is_consistent());
    }

    #[test]
    fn landmark_observations_stay_sorted_and_unique() {
        let mut lm = Landmark::new(
            LandmarkId(1),
            Vec3::new(0.0, 0.0, 1.0),
            descriptor(3),
            KeyframeId(5),
        );
        lm.observe(KeyframeId(2));
        lm.observe(KeyframeId(9));
        lm.observe(KeyframeId(5)); // duplicate
        assert_eq!(
            lm.observations,
            vec![KeyframeId(2), KeyframeId(5), KeyframeId(9)]
        );
        assert_eq!(lm.observation_count(), 3);
        assert!(lm.forget(KeyframeId(5)));
        assert!(!lm.forget(KeyframeId(5)));
        assert_eq!(lm.observations, vec![KeyframeId(2), KeyframeId(9)]);
        assert!(!lm.is_orphaned());
        lm.forget(KeyframeId(2));
        lm.forget(KeyframeId(9));
        assert!(lm.is_orphaned());
    }

    #[test]
    fn update_descriptor_takes_the_hamming_median() {
        let mut lm = Landmark::new(
            LandmarkId(1),
            Vec3::zeros(),
            BinaryDescriptor::ZERO,
            KeyframeId(0),
        );
        let a = BinaryDescriptor([0b0000_0111; 32]);
        let b = BinaryDescriptor([0b0000_0011; 32]);
        let c = BinaryDescriptor([0b0000_0001; 32]);
        lm.update_descriptor(&[a, b, c]);
        assert_eq!(lm.descriptor.0[0], 0b0000_0011);
        // An empty update leaves the descriptor alone rather than zeroing it.
        lm.update_descriptor(&[]);
        assert_eq!(lm.descriptor.0[0], 0b0000_0011);
    }

    #[test]
    fn memory_bytes_scales_with_contents() {
        let small = Keyframe::new(
            KeyframeId(0),
            Timestamp::ZERO,
            Se3::identity(),
            vec![Vec2::zeros(); 10],
            vec![descriptor(1); 10],
            CameraIntrinsics::from_focal(500.0, 640, 480),
            &Vocabulary::empty(),
        );
        let big = Keyframe::new(
            KeyframeId(0),
            Timestamp::ZERO,
            Se3::identity(),
            vec![Vec2::zeros(); 100],
            vec![descriptor(1); 100],
            CameraIntrinsics::from_focal(500.0, 640, 480),
            &Vocabulary::empty(),
        );
        assert!(big.memory_bytes() > small.memory_bytes());
        // 90 extra features cost at least 90 descriptors' worth of bytes.
        assert!(big.memory_bytes() - small.memory_bytes() >= 90 * 32);
    }
}
