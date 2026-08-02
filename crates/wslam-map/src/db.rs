//! The keyframe database: storage, the inverted index, and culling.
//!
//! Two things here are load-bearing beyond "a container of keyframes".
//!
//! **The inverted index** is what makes place recognition cheap. Without it a
//! query scores against every keyframe in the map and relocalization cost grows
//! linearly with session length; with it, only keyframes sharing a word are
//! touched (Gálvez-López & Tardós 2012, §IV).
//!
//! **Culling** is the answer to spec.md §9's *"Unbounded map memory — tab killed
//! on long sessions"*, whose stated mitigation is *"keyframe culling; measure
//! MB/min from M4a"*. [`MapDb::memory_bytes`] is the measurement and
//! [`MapDb::cull`] is the mitigation; they belong together.

use crate::descriptor::BinaryDescriptor;
use crate::keyframe::{Keyframe, KeyframeId, Landmark, LandmarkId};
use crate::vocabulary::{BowVector, Vocabulary};
use std::collections::BTreeMap;
use std::sync::Arc;
use wslam_core::{ScaleEstimate, Se3};

/// When a keyframe is redundant enough to drop.
///
/// The default is ORB-SLAM2's rule (Mur-Artal & Tardós 2017, §VI-E): discard a
/// keyframe when 90% of the landmarks it sees are also seen by at least three
/// other keyframes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CullPolicy {
    /// Fraction of a keyframe's landmarks that must be redundantly observed
    /// before the keyframe itself is redundant.
    pub redundancy_ratio: f64,
    /// How many *other* keyframes must see a landmark for it to count as
    /// redundantly observed.
    pub min_other_observers: usize,
    /// Never cull the oldest `protect_oldest` keyframes. The first keyframe
    /// defines the map frame the scale anchor was measured in, so losing it
    /// costs more than it saves.
    pub protect_oldest: usize,
    /// Never cull the newest `protect_newest` keyframes: the tracker's local
    /// map is built around them and they have not had time to become
    /// redundant.
    pub protect_newest: usize,
    /// Refuse to cull a keyframe that is the sole observer of some landmark.
    ///
    /// With this set, culling is guaranteed not to change the landmark count —
    /// coverage is preserved exactly, which is what makes "cull shrinks memory
    /// without shrinking the map" a checkable claim rather than a hope.
    pub protect_sole_observers: bool,
    /// Hard cap on keyframe count; ignored when zero. Above the cap, keyframes
    /// are dropped even if they are not redundant (still respecting the
    /// protections above) — a bounded map beats a perfect one that gets the tab
    /// killed.
    pub max_keyframes: usize,
}

impl Default for CullPolicy {
    fn default() -> Self {
        CullPolicy {
            redundancy_ratio: 0.9,
            min_other_observers: 3,
            protect_oldest: 1,
            protect_newest: 2,
            protect_sole_observers: true,
            max_keyframes: 0,
        }
    }
}

/// Keyframes, landmarks, the inverted index and the map's scale anchor.
#[derive(Debug, Clone)]
pub struct MapDb {
    vocabulary: Arc<Vocabulary>,
    keyframes: BTreeMap<u64, Keyframe>,
    landmarks: BTreeMap<u64, Landmark>,
    /// word -> keyframes containing it, ascending.
    inverted: BTreeMap<u32, Vec<KeyframeId>>,
    next_keyframe: u64,
    next_landmark: u64,
    scale_anchor: ScaleEstimate,
}

impl MapDb {
    /// An empty map over a shared vocabulary.
    ///
    /// The anchor starts at [`ScaleEstimate::unscaled`] — spec.md §3's *"no
    /// silent assumptions"*: an unanchored map reports infinite scale variance
    /// rather than pretending its units are metres.
    #[must_use]
    pub fn new(vocabulary: Arc<Vocabulary>) -> Self {
        MapDb {
            vocabulary,
            keyframes: BTreeMap::new(),
            landmarks: BTreeMap::new(),
            inverted: BTreeMap::new(),
            next_keyframe: 0,
            next_landmark: 0,
            scale_anchor: ScaleEstimate::unscaled(),
        }
    }

    /// The vocabulary this map's bag-of-words vectors were built against.
    #[must_use]
    pub fn vocabulary(&self) -> &Arc<Vocabulary> {
        &self.vocabulary
    }

    /// Insert a keyframe and index its words.
    ///
    /// If `kf.id` is [`KeyframeId::UNSET`] a fresh id is assigned; otherwise the
    /// supplied id is honoured, which is how deserialisation restores a map
    /// with its cross-references intact. Either way an existing keyframe with
    /// the same id is replaced and its index entries rebuilt.
    pub fn insert_keyframe(&mut self, mut kf: Keyframe) -> KeyframeId {
        let id = if kf.id.is_unset() {
            KeyframeId(self.next_keyframe)
        } else {
            kf.id
        };
        kf.id = id;
        // Keep the parallel arrays honest even if a caller got them wrong;
        // everything downstream indexes all three with one loop counter.
        let n = kf.keypoints.len().min(kf.descriptors.len());
        kf.keypoints.truncate(n);
        kf.descriptors.truncate(n);
        kf.landmarks.resize(n, None);

        if self.keyframes.contains_key(&id.0) {
            self.deindex(id);
        }
        self.next_keyframe = self.next_keyframe.max(id.0.saturating_add(1));
        for word in kf.bow.words() {
            let slot = self.inverted.entry(word).or_default();
            if let Err(pos) = slot.binary_search(&id) {
                slot.insert(pos, id);
            }
        }
        self.keyframes.insert(id.0, kf);
        id
    }

    /// Insert a landmark, assigning an id if it is [`LandmarkId::UNSET`].
    pub fn insert_landmark(&mut self, mut lm: Landmark) -> LandmarkId {
        let id = if lm.id.is_unset() {
            LandmarkId(self.next_landmark)
        } else {
            lm.id
        };
        lm.id = id;
        lm.observations.sort_unstable();
        lm.observations.dedup();
        self.next_landmark = self.next_landmark.max(id.0.saturating_add(1));
        self.landmarks.insert(id.0, lm);
        id
    }

    /// Look up a keyframe.
    #[must_use]
    pub fn keyframe(&self, id: KeyframeId) -> Option<&Keyframe> {
        self.keyframes.get(&id.0)
    }

    /// Mutable keyframe access, for attaching landmark matches after insertion.
    pub fn keyframe_mut(&mut self, id: KeyframeId) -> Option<&mut Keyframe> {
        self.keyframes.get_mut(&id.0)
    }

    /// Look up a landmark.
    #[must_use]
    pub fn landmark(&self, id: LandmarkId) -> Option<&Landmark> {
        self.landmarks.get(&id.0)
    }

    /// Mutable landmark access.
    pub fn landmark_mut(&mut self, id: LandmarkId) -> Option<&mut Landmark> {
        self.landmarks.get_mut(&id.0)
    }

    /// All keyframes, ascending by id.
    pub fn keyframes(&self) -> impl Iterator<Item = &Keyframe> {
        self.keyframes.values()
    }

    /// All landmarks, ascending by id.
    pub fn landmarks(&self) -> impl Iterator<Item = &Landmark> {
        self.landmarks.values()
    }

    /// Number of keyframes.
    #[must_use]
    pub fn keyframe_count(&self) -> usize {
        self.keyframes.len()
    }

    /// Number of landmarks.
    #[must_use]
    pub fn landmark_count(&self) -> usize {
        self.landmarks.len()
    }

    /// Whether the map holds nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keyframes.is_empty() && self.landmarks.is_empty()
    }

    /// Next id the database would assign to a keyframe. Serialised so that ids
    /// keep increasing across a save/load cycle.
    #[must_use]
    pub fn next_keyframe_id(&self) -> u64 {
        self.next_keyframe
    }

    /// Next id the database would assign to a landmark.
    #[must_use]
    pub fn next_landmark_id(&self) -> u64 {
        self.next_landmark
    }

    /// Restore the id counters after deserialisation.
    pub fn set_next_ids(&mut self, keyframe: u64, landmark: u64) {
        self.next_keyframe = self.next_keyframe.max(keyframe);
        self.next_landmark = self.next_landmark.max(landmark);
    }

    /// Keyframes whose bag-of-words vector contains `word`.
    #[must_use]
    pub fn keyframes_with_word(&self, word: u32) -> &[KeyframeId] {
        self.inverted.get(&word).map_or(&[], |v| v.as_slice())
    }

    /// Number of indexed words.
    #[must_use]
    pub fn indexed_word_count(&self) -> usize {
        self.inverted.len()
    }

    /// Attach a keypoint of a keyframe to a landmark, updating both sides.
    ///
    /// Returns `false` if either id or the feature index is unknown, rather
    /// than panicking: the caller is usually acting on a match that a
    /// concurrent cull may already have invalidated.
    pub fn link_observation(
        &mut self,
        keyframe: KeyframeId,
        feature: usize,
        landmark: LandmarkId,
    ) -> bool {
        if !self.landmarks.contains_key(&landmark.0) {
            return false;
        }
        let Some(kf) = self.keyframes.get_mut(&keyframe.0) else {
            return false;
        };
        let Some(slot) = kf.landmarks.get_mut(feature) else {
            return false;
        };
        *slot = Some(landmark);
        if let Some(lm) = self.landmarks.get_mut(&landmark.0) {
            lm.observe(keyframe);
        }
        true
    }

    /// The map's metric anchor.
    ///
    /// spec.md §2: *"Anchor scale once ... and every subsequent session recovers
    /// metric by relocalizing"*. This value, and its variance, is the entire
    /// point of persisting a map.
    #[must_use]
    pub fn scale_anchor(&self) -> ScaleEstimate {
        self.scale_anchor
    }

    /// Set the metric anchor.
    pub fn set_scale_anchor(&mut self, s: ScaleEstimate) {
        self.scale_anchor = s;
    }

    /// Rescale the whole map into metres and record the anchor.
    ///
    /// Landmark positions and keyframe centres are multiplied by
    /// `anchor.value`; rotations are scale-invariant and untouched. Applying an
    /// anchor twice would double-scale the map, so this is a one-shot operation
    /// guarded by the caller — [`MapDb::scale_anchor`] reports what is already
    /// applied.
    pub fn apply_scale(&mut self, anchor: ScaleEstimate) {
        let s = anchor.value;
        if s.is_finite() && s > 0.0 && s != 1.0 {
            for kf in self.keyframes.values_mut() {
                kf.pose = kf.pose.scaled(s);
            }
            for lm in self.landmarks.values_mut() {
                lm.position *= s;
            }
        }
        self.scale_anchor = anchor;
    }

    /// Replace a keyframe's pose, as a pose-graph optimisation would.
    pub fn set_keyframe_pose(&mut self, id: KeyframeId, pose: Se3) -> bool {
        match self.keyframes.get_mut(&id.0) {
            Some(kf) => {
                kf.pose = pose;
                true
            }
            None => false,
        }
    }

    /// Remove a keyframe, its index entries and its observations.
    ///
    /// Landmarks left with no observers are removed too; the count of removed
    /// landmarks is returned alongside so a caller can report coverage loss.
    pub fn remove_keyframe(&mut self, id: KeyframeId) -> (bool, usize) {
        let Some(kf) = self.keyframes.remove(&id.0) else {
            return (false, 0);
        };
        for word in kf.bow.words() {
            if let Some(slot) = self.inverted.get_mut(&word) {
                if let Ok(pos) = slot.binary_search(&id) {
                    slot.remove(pos);
                }
            }
        }
        self.inverted.retain(|_, v| !v.is_empty());

        let mut orphans = Vec::new();
        for lm_id in kf.landmarks.iter().flatten() {
            if let Some(lm) = self.landmarks.get_mut(&lm_id.0) {
                lm.forget(id);
                if lm.is_orphaned() {
                    orphans.push(lm_id.0);
                }
            }
        }
        let dropped = orphans.len();
        for o in orphans {
            self.landmarks.remove(&o);
        }
        (true, dropped)
    }

    fn deindex(&mut self, id: KeyframeId) {
        if let Some(kf) = self.keyframes.get(&id.0) {
            let words: Vec<u32> = kf.bow.words().collect();
            for w in words {
                if let Some(slot) = self.inverted.get_mut(&w) {
                    if let Ok(pos) = slot.binary_search(&id) {
                        slot.remove(pos);
                    }
                }
            }
            self.inverted.retain(|_, v| !v.is_empty());
        }
    }

    /// Fraction of a keyframe's landmarks that are also seen by at least
    /// `min_other` other keyframes.
    ///
    /// A keyframe that observes no landmarks scores **0.0** — unevaluable, not
    /// redundant.
    ///
    /// The opposite reading is tempting: a keyframe contributing no geometry
    /// costs memory and earns nothing. But the consequence of scoring it 1.0 is
    /// that `cull` deletes it, and if the *producer* is not populating landmark
    /// links then every keyframe scores 1.0 and the map deletes itself the
    /// moment culling runs. That is not hypothetical — it is what made the
    /// keyframe count fall from 184 at 600 frames to 56 at 1200 on EuRoC, and
    /// it takes the relocalization database with it.
    ///
    /// ORB-SLAM's rule is `nRedundantObservations > 0.9 * nMPs`, which is false
    /// when `nMPs == 0` for exactly this reason. Refusing to judge is the safe
    /// default when the alternative is destroying the map.
    #[must_use]
    pub fn redundancy(&self, id: KeyframeId, min_other: usize) -> f64 {
        let Some(kf) = self.keyframes.get(&id.0) else {
            return 0.0;
        };
        let observed = kf.observed_landmarks();
        if observed.is_empty() {
            return 0.0;
        }
        let redundant = observed
            .iter()
            .filter(|l| {
                self.landmarks
                    .get(&l.0)
                    .is_some_and(|lm| lm.observation_count().saturating_sub(1) >= min_other)
            })
            .count();
        redundant as f64 / observed.len() as f64
    }

    /// Whether this keyframe is the only observer of some landmark.
    #[must_use]
    pub fn is_sole_observer(&self, id: KeyframeId) -> bool {
        let Some(kf) = self.keyframes.get(&id.0) else {
            return false;
        };
        kf.observed_landmarks().iter().any(|l| {
            self.landmarks
                .get(&l.0)
                .is_some_and(|lm| lm.observations.as_slice() == [id])
        })
    }

    /// Drop redundant keyframes. Returns how many were removed.
    ///
    /// Decisions are made one at a time against the *surviving* set, exactly as
    /// ORB-SLAM2's local mapping thread does: dropping a keyframe reduces its
    /// landmarks' observation counts, which can make the next candidate
    /// non-redundant. Evaluating all candidates against the original counts and
    /// then deleting them together is the classic way to gut a map in one pass.
    pub fn cull(&mut self, policy: &CullPolicy) -> usize {
        let order: Vec<u64> = self.keyframes.keys().copied().collect();
        let n = order.len();
        let protect_lo = policy.protect_oldest.min(n);
        let protect_hi = n.saturating_sub(policy.protect_newest);
        let mut removed = 0;

        for (rank, &raw) in order.iter().enumerate() {
            if rank < protect_lo || rank >= protect_hi {
                continue;
            }
            let id = KeyframeId(raw);
            if !self.keyframes.contains_key(&raw) {
                continue;
            }
            if policy.protect_sole_observers && self.is_sole_observer(id) {
                continue;
            }
            let redundant =
                self.redundancy(id, policy.min_other_observers) >= policy.redundancy_ratio;
            let over_cap = policy.max_keyframes > 0 && self.keyframes.len() > policy.max_keyframes;
            if redundant || over_cap {
                self.remove_keyframe(id);
                removed += 1;
            }
        }
        if removed > 0 {
            log::debug!(
                "culled {removed} keyframes, {} remain, {} bytes",
                self.keyframes.len(),
                self.memory_bytes()
            );
        }
        removed
    }

    /// Approximate resident size of the map in bytes.
    ///
    /// spec.md §6 L4 asks for *"map memory growth vs session duration (MB/min)"*
    /// — this is the numerator. Computed from element counts rather than
    /// allocator capacity so it is a deterministic function of map contents and
    /// can therefore be asserted on in a regression test.
    ///
    /// The vocabulary is excluded: it is shared (`Arc`) and typically loaded
    /// from a fixed artifact, so charging it to the map would make a
    /// per-session growth figure meaningless. Use
    /// [`Vocabulary::memory_bytes`] for that side of the budget.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        let node = 48; // rough BTreeMap per-entry overhead, counted uniformly
        let kf: usize = self
            .keyframes
            .values()
            .map(|k| k.memory_bytes() + node)
            .sum();
        let lm: usize = self
            .landmarks
            .values()
            .map(|l| l.memory_bytes() + node)
            .sum();
        let index: usize = self
            .inverted
            .values()
            .map(|v| node + v.len() * std::mem::size_of::<KeyframeId>())
            .sum();
        std::mem::size_of::<Self>() + kf + lm + index
    }

    /// Compute a bag-of-words vector with this map's vocabulary.
    #[must_use]
    pub fn transform(&self, descriptors: &[BinaryDescriptor]) -> BowVector {
        self.vocabulary.transform(descriptors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth;
    use wslam_core::{CameraIntrinsics, DeterministicRng, ScaleKind, Timestamp, Vec2, Vec3};

    fn empty_db() -> MapDb {
        MapDb::new(Arc::new(Vocabulary::empty()))
    }

    #[test]
    fn insert_assigns_ids_and_honours_explicit_ones() {
        let mut db = empty_db();
        let vocab = Vocabulary::empty();
        let mk = |id| {
            Keyframe::new(
                id,
                Timestamp::ZERO,
                Se3::identity(),
                vec![Vec2::zeros(); 2],
                vec![BinaryDescriptor::ZERO; 2],
                CameraIntrinsics::from_focal(500.0, 640, 480),
                &vocab,
            )
        };
        assert_eq!(db.insert_keyframe(mk(KeyframeId::UNSET)), KeyframeId(0));
        assert_eq!(db.insert_keyframe(mk(KeyframeId::UNSET)), KeyframeId(1));
        assert_eq!(db.insert_keyframe(mk(KeyframeId(40))), KeyframeId(40));
        // The counter jumps past an explicit id so it cannot be reissued.
        assert_eq!(db.insert_keyframe(mk(KeyframeId::UNSET)), KeyframeId(41));
        assert_eq!(db.keyframe_count(), 4);
        let ids: Vec<u64> = db.keyframes().map(|k| k.id.0).collect();
        assert_eq!(ids, vec![0, 1, 40, 41], "iteration must be id-ordered");
    }

    #[test]
    fn insert_repairs_mismatched_parallel_arrays() {
        let mut db = empty_db();
        let mut kf = Keyframe::new(
            KeyframeId::UNSET,
            Timestamp::ZERO,
            Se3::identity(),
            vec![Vec2::zeros(); 5],
            vec![BinaryDescriptor::ZERO; 3],
            CameraIntrinsics::from_focal(500.0, 640, 480),
            &Vocabulary::empty(),
        );
        kf.landmarks.clear();
        let id = db.insert_keyframe(kf);
        let stored = db.keyframe(id).unwrap();
        assert!(stored.is_consistent());
        assert_eq!(stored.len(), 3);
    }

    #[test]
    fn inverted_index_lists_only_keyframes_holding_the_word() {
        let (db, _) = synth::corridor_map(6, 20260801);
        assert!(db.indexed_word_count() > 0);
        for kf in db.keyframes() {
            for w in kf.bow.words() {
                assert!(
                    db.keyframes_with_word(w).contains(&kf.id),
                    "{} missing from the index for word {w}",
                    kf.id
                );
            }
        }
        for (word, ids) in db.inverted.iter() {
            for id in ids {
                assert!(db.keyframe(*id).unwrap().bow.weight(*word) > 0.0);
            }
        }
        assert!(db.keyframes_with_word(u32::MAX).is_empty());
    }

    #[test]
    fn inverted_index_is_pruned_when_a_keyframe_is_removed() {
        let (mut db, _) = synth::corridor_map(6, 42);
        let victim = db.keyframes().next().unwrap().id;
        let words: Vec<u32> = db.keyframe(victim).unwrap().bow.words().collect();
        assert!(!words.is_empty());
        db.remove_keyframe(victim);
        for w in words {
            assert!(!db.keyframes_with_word(w).contains(&victim));
        }
        assert!(db.inverted.values().all(|v| !v.is_empty()));
    }

    #[test]
    fn link_observation_updates_both_sides_and_fails_softly() {
        let mut db = empty_db();
        let kf = db.insert_keyframe(Keyframe::new(
            KeyframeId::UNSET,
            Timestamp::ZERO,
            Se3::identity(),
            vec![Vec2::zeros(); 3],
            vec![BinaryDescriptor::ZERO; 3],
            CameraIntrinsics::from_focal(500.0, 640, 480),
            &Vocabulary::empty(),
        ));
        let lm = db.insert_landmark(Landmark::new(
            LandmarkId::UNSET,
            Vec3::new(0.0, 0.0, 2.0),
            BinaryDescriptor::ZERO,
            kf,
        ));
        assert!(db.link_observation(kf, 1, lm));
        assert_eq!(db.keyframe(kf).unwrap().landmarks[1], Some(lm));
        assert_eq!(db.landmark(lm).unwrap().observations, vec![kf]);
        // Unknown ids and out-of-range features are reported, not panicked on.
        assert!(!db.link_observation(kf, 99, lm));
        assert!(!db.link_observation(KeyframeId(77), 0, lm));
        assert!(!db.link_observation(kf, 0, LandmarkId(77)));
    }

    #[test]
    fn scale_anchor_defaults_to_unscaled_and_roundtrips() {
        let mut db = empty_db();
        assert_eq!(db.scale_anchor().source, ScaleKind::None);
        assert!(db.scale_anchor().variance.is_infinite());
        let anchor = ScaleEstimate::metric(ScaleKind::Fiducial, 1.37, 2.5e-4);
        db.set_scale_anchor(anchor);
        assert_eq!(db.scale_anchor(), anchor);
    }

    #[test]
    fn apply_scale_rescales_geometry_but_not_rotation() {
        let (mut db, _) = synth::corridor_map(4, 7);
        let before: Vec<Vec3> = db.landmarks().map(|l| l.position).collect();
        let rot_before: Vec<_> = db.keyframes().map(|k| k.pose.rotation().matrix()).collect();
        let anchor = ScaleEstimate::metric(ScaleKind::Declared, 2.0, 1e-6);
        db.apply_scale(anchor);
        for (a, b) in db.landmarks().zip(before.iter()) {
            assert!((a.position - b * 2.0).norm() < 1e-12);
        }
        for (k, r) in db.keyframes().zip(rot_before.iter()) {
            assert!((k.pose.rotation().matrix() - r).norm() < 1e-15);
        }
        assert_eq!(db.scale_anchor(), anchor);
    }

    #[test]
    fn redundancy_counts_other_observers_only() {
        let mut db = empty_db();
        let vocab = Vocabulary::empty();
        let mut ids = Vec::new();
        for _ in 0..4 {
            ids.push(db.insert_keyframe(Keyframe::new(
                KeyframeId::UNSET,
                Timestamp::ZERO,
                Se3::identity(),
                vec![Vec2::zeros(); 2],
                vec![BinaryDescriptor::ZERO; 2],
                CameraIntrinsics::from_focal(500.0, 640, 480),
                &vocab,
            )));
        }
        // Landmark 0 seen by all four; landmark 1 seen only by keyframe 0.
        let shared = db.insert_landmark(Landmark::new(
            LandmarkId::UNSET,
            Vec3::new(0.0, 0.0, 1.0),
            BinaryDescriptor::ZERO,
            ids[0],
        ));
        let private = db.insert_landmark(Landmark::new(
            LandmarkId::UNSET,
            Vec3::new(1.0, 0.0, 1.0),
            BinaryDescriptor::ZERO,
            ids[0],
        ));
        for &k in &ids {
            db.link_observation(k, 0, shared);
        }
        db.link_observation(ids[0], 1, private);

        // Keyframe 0 sees two landmarks, one of which has 3 other observers.
        assert!((db.redundancy(ids[0], 3) - 0.5).abs() < 1e-12);
        // Keyframe 1 sees only the shared landmark -> fully redundant.
        assert!((db.redundancy(ids[1], 3) - 1.0).abs() < 1e-12);
        // A keyframe observing nothing is *unevaluable*, not redundant.
        //
        // This assertion was inverted. Scoring a barren keyframe 1.0 reads as
        // "it contributes nothing, so drop it", which is reasonable in
        // isolation and catastrophic in context: if the producer is not
        // populating landmark links then every keyframe is barren, they all
        // score 1.0, and `cull` deletes the entire map. Measured on EuRoC, the
        // keyframe count fell from 184 at 600 frames to 56 at 1200 for exactly
        // this reason, taking the relocalization database with it.
        //
        // ORB-SLAM's `nRedundantObservations > 0.9 * nMPs` is false when
        // `nMPs == 0`, for the same reason.
        let barren = db.insert_keyframe(Keyframe::new(
            KeyframeId::UNSET,
            Timestamp::ZERO,
            Se3::identity(),
            vec![],
            vec![],
            CameraIntrinsics::from_focal(500.0, 640, 480),
            &vocab,
        ));
        assert_eq!(db.redundancy(barren, 3), 0.0);
        assert_eq!(db.redundancy(KeyframeId(999), 3), 0.0);
        assert!(db.is_sole_observer(ids[0]));
        assert!(!db.is_sole_observer(ids[1]));
    }

    #[test]
    fn cull_drops_redundant_keyframes_and_preserves_coverage() {
        // A revisit session: the camera loops past the same landmarks, so most
        // keyframes are genuinely redundant.
        let (mut db, _) = synth::revisit_map(4, 12, 20260801);
        let kf_before = db.keyframe_count();
        let lm_before = db.landmark_count();
        let bytes_before = db.memory_bytes();

        let removed = db.cull(&CullPolicy::default());

        assert!(removed > 0, "nothing was culled from a redundant session");
        assert_eq!(db.keyframe_count(), kf_before - removed);
        assert!(db.memory_bytes() < bytes_before);
        // Coverage: `protect_sole_observers` guarantees no landmark is lost.
        assert_eq!(db.landmark_count(), lm_before);
        // Every surviving landmark is still seen by someone.
        for lm in db.landmarks() {
            assert!(!lm.is_orphaned());
            for obs in &lm.observations {
                assert!(db.keyframe(*obs).is_some(), "dangling observation {obs}");
            }
        }
        // Every surviving keyframe's landmark references still resolve.
        for kf in db.keyframes() {
            for l in kf.landmarks.iter().flatten() {
                assert!(db.landmark(*l).is_some(), "dangling landmark {l}");
            }
        }
    }

    #[test]
    fn cull_never_drops_a_sole_observer() {
        let mut db = empty_db();
        let vocab = Vocabulary::empty();
        let mut ids = Vec::new();
        for _ in 0..10 {
            ids.push(db.insert_keyframe(Keyframe::new(
                KeyframeId::UNSET,
                Timestamp::ZERO,
                Se3::identity(),
                vec![Vec2::zeros(); 2],
                vec![BinaryDescriptor::ZERO; 2],
                CameraIntrinsics::from_focal(500.0, 640, 480),
                &vocab,
            )));
        }
        let shared = db.insert_landmark(Landmark::new(
            LandmarkId::UNSET,
            Vec3::new(0.0, 0.0, 1.0),
            BinaryDescriptor::ZERO,
            ids[0],
        ));
        for &k in &ids {
            db.link_observation(k, 0, shared);
        }
        // Keyframe 5 alone sees this one.
        let private = db.insert_landmark(Landmark::new(
            LandmarkId::UNSET,
            Vec3::new(1.0, 0.0, 1.0),
            BinaryDescriptor::ZERO,
            ids[5],
        ));
        db.link_observation(ids[5], 1, private);

        let removed = db.cull(&CullPolicy::default());
        assert!(removed > 0);
        assert!(db.keyframe(ids[5]).is_some(), "sole observer was culled");
        assert!(db.landmark(private).is_some());
        assert_eq!(db.landmark_count(), 2);
    }

    #[test]
    fn cull_respects_protected_oldest_and_newest() {
        let (mut db, _) = synth::revisit_map(3, 12, 5);
        let ids: Vec<KeyframeId> = db.keyframes().map(|k| k.id).collect();
        let policy = CullPolicy {
            protect_oldest: 2,
            protect_newest: 3,
            protect_sole_observers: false,
            ..CullPolicy::default()
        };
        db.cull(&policy);
        for id in ids.iter().take(2) {
            assert!(db.keyframe(*id).is_some(), "protected oldest {id} culled");
        }
        for id in ids.iter().rev().take(3) {
            assert!(db.keyframe(*id).is_some(), "protected newest {id} culled");
        }
    }

    #[test]
    fn cull_with_a_hard_cap_bounds_the_keyframe_count() {
        let (mut db, _) = synth::corridor_map(20, 3);
        let policy = CullPolicy {
            // Nothing is redundant in a corridor sweep, so only the cap bites.
            redundancy_ratio: 2.0,
            protect_oldest: 1,
            protect_newest: 1,
            protect_sole_observers: false,
            max_keyframes: 8,
            ..CullPolicy::default()
        };
        assert!(db.keyframe_count() > 8);
        db.cull(&policy);
        assert!(
            db.keyframe_count() <= 8 + 2,
            "cap not honoured: {} keyframes",
            db.keyframe_count()
        );
    }

    #[test]
    fn cull_of_an_empty_map_is_a_no_op() {
        let mut db = empty_db();
        assert_eq!(db.cull(&CullPolicy::default()), 0);
        assert!(db.is_empty());
        assert_eq!(db.memory_bytes(), db.memory_bytes());
    }

    /// Run a revisiting session and sample [`MapDb::memory_bytes`] after the
    /// given lap counts, with culling either in the loop or absent.
    ///
    /// The two variants consume an identically seeded RNG and therefore see the
    /// *same* keyframe sequence, which makes the comparison paired.
    fn revisit_session(cull: bool, sample_at: &[usize]) -> Vec<usize> {
        let mut rng = DeterministicRng::new("mem", 20260801);
        let (mut db, scene) = synth::revisit_map(1, 12, 20260801);
        let policy = CullPolicy::default();
        let mut samples = Vec::new();
        for lap in 2..=sample_at.iter().copied().max().unwrap_or(1) {
            synth::append_lap(&mut db, &scene, &mut rng);
            if cull {
                db.cull(&policy);
            }
            if sample_at.contains(&lap) {
                samples.push(db.memory_bytes());
            }
        }
        samples
    }

    #[test]
    fn memory_bytes_grows_sublinearly_under_culling_on_a_long_session() {
        // spec.md §9: "Unbounded map memory -> tab killed on long sessions."
        // A revisiting session sees the same landmarks over and over, so with
        // culling in the loop the map must approach a steady state rather than
        // growing with wall time.
        let culled = revisit_session(true, &[4, 8, 16]);
        let (m4, m8, m16) = (culled[0], culled[1], culled[2]);

        assert!(
            m8 < 2 * m4,
            "doubling the session doubled memory: {m4} -> {m8}"
        );
        assert!(
            m16 < 2 * m8,
            "memory is not saturating: {m4} -> {m8} -> {m16}"
        );

        // Saturation, stated as a ratio rather than as a difference of
        // differences.
        //
        // This assertion used to read `(m16 - m8) < (m8 - m4) * 3 / 2` on
        // `usize`, and it panicked on subtraction overflow — because culling
        // works, memory *fell* from 4 laps to 8 and `m8 - m4` went negative.
        // Redoing it in signed arithmetic does not rescue it either: once the
        // map has saturated, the per-lap difference is ±3% jitter in how many
        // keyframes survive a cull, so comparing two such differences is a coin
        // toss on noise and would have been flaky in whichever direction it
        // landed. The property actually worth asserting is that quadrupling the
        // session does not materially grow the map at all; measured, 16 laps
        // sits at 0.97x of 4 laps, against the 4x that linear growth would give.
        assert!(
            m16 * 4 <= m4 * 5,
            "quadrupling the session grew memory by more than 25%: {m4} -> {m16}"
        );

        // ... and the bound has to be culling's doing, not the fixture's. The
        // identical session with the cull removed must grow roughly linearly,
        // or this test would keep passing after someone made `cull` a no-op.
        let uncontrolled = revisit_session(false, &[4, 8, 16]);
        let (u4, u16) = (uncontrolled[0], uncontrolled[2]);
        assert!(
            u16 >= 3 * u4,
            "the uncontrolled session did not grow, so the fixture cannot \
             distinguish culling from doing nothing: {u4} -> {u16}"
        );
        assert!(
            m16 * 4 < u16,
            "culling saved less than 4x on a 16-lap revisit: {m16} vs {u16}"
        );
    }

    #[test]
    fn memory_bytes_is_deterministic_and_monotone_in_content() {
        let (small, _) = synth::corridor_map(4, 1);
        let (big, _) = synth::corridor_map(12, 1);
        assert_eq!(small.memory_bytes(), small.memory_bytes());
        assert!(big.memory_bytes() > small.memory_bytes());
        assert!(small.memory_bytes() > 0);
    }

    #[test]
    fn remove_keyframe_reports_orphaned_landmarks() {
        let mut db = empty_db();
        let kf = db.insert_keyframe(Keyframe::new(
            KeyframeId::UNSET,
            Timestamp::ZERO,
            Se3::identity(),
            vec![Vec2::zeros(); 2],
            vec![BinaryDescriptor::ZERO; 2],
            CameraIntrinsics::from_focal(500.0, 640, 480),
            &Vocabulary::empty(),
        ));
        let lm = db.insert_landmark(Landmark::new(
            LandmarkId::UNSET,
            Vec3::new(0.0, 0.0, 1.0),
            BinaryDescriptor::ZERO,
            kf,
        ));
        db.link_observation(kf, 0, lm);
        assert_eq!(db.remove_keyframe(kf), (true, 1));
        assert_eq!(db.landmark_count(), 0);
        assert_eq!(db.remove_keyframe(kf), (false, 0));
    }
}
