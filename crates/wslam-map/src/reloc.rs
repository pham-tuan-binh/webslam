//! Place recognition with **mandatory** geometric verification.
//!
//! spec.md §5 states the rule this module exists to enforce: *"a false-positive
//! loop closure corrupts the map irrecoverably and is worse than no loop
//! closure at all. Geometric verification after every place-recognition hit is
//! non-negotiable, and false-positive rate is a first-class metric."*
//!
//! That is made structural rather than procedural. [`Relocalizer::query`]
//! returns [`Candidate`]s, which carry a bag-of-words score and *nothing a
//! caller can act on*. The only type carrying a pose is [`Verified`], it is
//! `#[non_exhaustive]` so no crate but this one can construct it, and the only
//! function that returns one runs PnP RANSAC first. There is no code path from
//! "high BoW score" to "here is your pose" that skips the geometry.
//!
//! ## Why P3P rather than 3D-3D alignment
//!
//! The frozen `verify` signature hands us `keypoints: &[Vec2]` — the query side
//! is a raw camera frame with 2-D features and no depth. Horn/Umeyama absolute
//! orientation needs 3-D points on *both* sides, which we do not have and could
//! only fake by triangulating against a pose we have not yet computed. So this
//! is perspective-3-point (Grunert 1841, in the form tabulated by Haralick et
//! al., IJCV 1994) inside RANSAC, followed by a Gauss-Newton refinement on the
//! inliers that also produces the covariance.
//!
//! wslam-map may not depend on wslam-track — that would be a sideways
//! dependency and spec.md §7 makes the layer boundaries crate boundaries — so
//! the ~120 lines of solver live here. It is a different solver from L3's
//! anyway: this one runs once on the backend thread with a wide outlier
//! fraction, not every frame with a motion prior.

use crate::db::MapDb;
use crate::descriptor::{nearest_two, BinaryDescriptor};
use crate::keyframe::{KeyframeId, LandmarkId};
use crate::vocabulary::BowVector;
use wslam_core::covariance::{enforce_psd, symmetrize};
use wslam_core::math::{hat, umeyama};
use wslam_core::{CameraIntrinsics, DeterministicRng, Mat6, Se3, Vec2, Vec3, Vec6};

/// Maximum Hamming distance for a descriptor match to be considered at all.
/// ORB-SLAM2 uses 50-100 out of 256 depending on the stage; 64 is the quarter
/// point and comfortably inside the noise floor for random 256-bit codes, whose
/// pairwise distance concentrates at 128 +/- 8.
pub const MAX_MATCH_HAMMING: u32 = 64;

/// Lowe ratio between the best and second-best Hamming distance. A match whose
/// runner-up is nearly as close is ambiguous and worth less than nothing to
/// RANSAC.
pub const MATCH_RATIO: f64 = 0.8;

/// Covisibility neighbours whose landmarks join the candidate's matching pool
/// during verification. ORB-SLAM expands relocalization matching the same way;
/// past ~5 neighbours the added landmarks stop being visible from the query
/// pose and only dilute the ratio test.
const MAX_COVIS_NEIGHBOURS: usize = 5;

/// A neighbour must share at least this many landmarks with the candidate to
/// contribute its own. Below this the "neighbour" is likely a place-recognition
/// alias rather than an adjacent view, and its landmarks are noise.
const MIN_SHARED_LANDMARKS: usize = 8;

/// Ratio-tested matches needed before RANSAC runs at all. P3P needs 4; below
/// this the pose is too often a fluke worth rejecting before the guided pass.
const MIN_SEED_MATCHES: usize = 8;

/// Search window around a landmark's predicted pixel in the guided second
/// pass. Wide enough to absorb the coarse pose's reprojection error, narrow
/// enough that the window itself does the disambiguation a ratio test would.
const GUIDED_RADIUS_PX: f64 = 12.0;

/// DBoW2 keeps only keyframes sharing at least this fraction of the maximum
/// observed common-word count. It cuts the candidate list by an order of
/// magnitude for no measurable recall cost.
pub const COMMON_WORD_FRACTION: f64 = 0.8;

/// Huber threshold for the pose refinement, as a multiple of the RANSAC inlier
/// threshold.
const HUBER_SCALE: f64 = 1.0;

/// Thresholds for place recognition and its verification.
///
/// `min_inliers` and `ransac_threshold_px` together *are* the false-positive
/// gate that spec.md §6 L4 calls a release criterion. Raising `min_inliers` is
/// the intended response to any observed false positive.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RelocConfig {
    /// Minimum bag-of-words score for a keyframe to become a candidate.
    pub min_bow_score: f64,
    /// How many candidates to verify, best-scoring first.
    pub max_candidates: usize,
    /// Minimum PnP inlier count to accept a verification.
    pub min_inliers: usize,
    /// Reprojection inlier threshold, in pixels.
    pub ransac_threshold_px: f64,
    /// Maximum RANSAC iterations. The loop also terminates adaptively, as a
    /// deterministic function of the inlier ratio found so far.
    pub ransac_iterations: usize,
}

impl Default for RelocConfig {
    fn default() -> Self {
        RelocConfig {
            min_bow_score: 0.015,
            max_candidates: 5,
            min_inliers: 20,
            ransac_threshold_px: 3.0,
            ransac_iterations: 300,
        }
    }
}

/// A place-recognition hit. **Not** an accepted relocalization.
///
/// Deliberately carries no pose: everything a caller could do with a candidate
/// other than hand it to [`Relocalizer::verify`] is a bug.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidate {
    /// The matched keyframe.
    pub keyframe: KeyframeId,
    /// Bag-of-words similarity in `[0, 1]`.
    pub score: f64,
}

/// A relocalization that survived geometric verification.
///
/// `#[non_exhaustive]`: only this module can construct one, and it only does so
/// after PnP RANSAC has met [`RelocConfig::min_inliers`]. That is the structural
/// half of spec.md §5's "geometric verification ... is non-negotiable".
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Verified {
    /// The keyframe the query was matched to.
    pub keyframe: KeyframeId,
    /// `T_world_camera` of the *query* frame, in the map's coordinate frame and
    /// therefore in the map's units.
    pub pose: Se3,
    /// Number of correspondences supporting the pose.
    pub inliers: usize,
    /// 6x6 pose covariance, `[translation; rotation]`, right-perturbation, in
    /// the body frame — the same convention as everything else in the workspace.
    pub covariance: Mat6,
    /// The surviving 2D–3D correspondences: `(query keypoint index, map
    /// landmark)`. What a caller needs to relate the query's *own* geometry to
    /// the map's — the 3D–3D pairs behind a Sim(3) epoch merge — without
    /// re-running the matcher.
    pub correspondences: Vec<(usize, LandmarkId)>,
}

/// Why a candidate failed geometric verification.
///
/// spec.md §8 asks the dev viewer to show *"pose-graph edges including loop
/// candidates rejected by geometric verification — this is how the
/// false-positive threshold gets tuned by eye rather than by guesswork"*. A
/// bare `None` cannot be tuned by eye, and it cannot be tested either: it does
/// not distinguish "the appearance matcher threw the impostor out cheaply" from
/// "PnP looked at the geometry and found it inconsistent". Only the second is
/// evidence that the gate spec.md §5 calls non-negotiable actually ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rejection {
    /// The candidate names a keyframe that is not in the map.
    UnknownKeyframe,
    /// The candidate keyframe has too few triangulated landmarks to constrain a
    /// pose at all.
    NoMappedLandmarks {
        /// Landmark-bearing features on the candidate keyframe.
        landmarks: usize,
    },
    /// Descriptor matching produced too few 2D-3D correspondences to reach PnP.
    /// The appearance stage rejected this one; the geometry never ran.
    TooFewMatches {
        /// Correspondences that survived the ratio test and one-to-one pruning.
        matches: usize,
        /// How many were needed.
        required: usize,
    },
    /// RANSAC found no pose at all consistent with any minimal sample.
    PnpFailed {
        /// Correspondences offered to RANSAC.
        matches: usize,
    },
    /// **The geometric rejection.** PnP ran on enough correspondences and the
    /// best pose it could find was not supported by enough of them.
    TooFewInliers {
        /// Correspondences offered to RANSAC.
        matches: usize,
        /// Correspondences the best pose explained.
        inliers: usize,
        /// How many were needed.
        required: usize,
    },
    /// A pose was found but its information matrix is singular, so no
    /// covariance can be attached. spec.md §4 L6: a pose we cannot put a
    /// covariance on is not a pose we are willing to emit.
    NoCovariance {
        /// Correspondences supporting the pose.
        inliers: usize,
    },
}

impl Rejection {
    /// Whether PnP actually ran on this candidate — that is, whether the
    /// decision was made by the geometry rather than by the descriptor matcher.
    #[must_use]
    pub fn reached_geometry(self) -> bool {
        matches!(
            self,
            Rejection::PnpFailed { .. }
                | Rejection::TooFewInliers { .. }
                | Rejection::NoCovariance { .. }
        )
    }
}

impl std::fmt::Display for Rejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Rejection::UnknownKeyframe => write!(f, "unknown keyframe"),
            Rejection::NoMappedLandmarks { landmarks } => {
                write!(f, "only {landmarks} mapped landmarks")
            }
            Rejection::TooFewMatches { matches, required } => {
                write!(f, "{matches} descriptor matches < {required}")
            }
            Rejection::PnpFailed { matches } => write!(f, "PnP failed on {matches} matches"),
            Rejection::TooFewInliers {
                matches,
                inliers,
                required,
            } => write!(f, "{inliers}/{matches} inliers < {required}"),
            Rejection::NoCovariance { inliers } => {
                write!(f, "{inliers} inliers but singular information matrix")
            }
        }
    }
}

/// Bag-of-words query plus geometric verification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Relocalizer {
    config: RelocConfig,
}

impl Relocalizer {
    /// Build a relocalizer.
    #[must_use]
    pub fn new(config: RelocConfig) -> Self {
        Relocalizer { config }
    }

    /// The thresholds in force.
    #[must_use]
    pub fn config(&self) -> &RelocConfig {
        &self.config
    }

    /// Rank map keyframes against a query bag-of-words vector.
    ///
    /// Uses the inverted index, so cost is proportional to the number of
    /// keyframes sharing a word rather than to the map size. `exclude_recent`
    /// drops the newest N keyframes from consideration — for loop closure they
    /// are the temporal neighbours the tracker is already matching against, and
    /// "recognising" them proves nothing.
    #[must_use]
    pub fn query(&self, db: &MapDb, bow: &BowVector, exclude_recent: usize) -> Vec<Candidate> {
        if bow.is_empty() || db.keyframe_count() == 0 {
            return Vec::new();
        }
        // Keyframe ids ascend, so "the newest N" is the tail of the id order.
        let ids: Vec<KeyframeId> = db.keyframes().map(|k| k.id).collect();
        let keep = ids.len().saturating_sub(exclude_recent);
        let cutoff = ids.get(keep).copied();

        let mut common: std::collections::BTreeMap<KeyframeId, usize> = Default::default();
        for word in bow.words() {
            for &id in db.keyframes_with_word(word) {
                if cutoff.is_some_and(|c| id >= c) {
                    continue;
                }
                *common.entry(id).or_insert(0) += 1;
            }
        }
        let max_common = common.values().copied().max().unwrap_or(0);
        if max_common == 0 {
            return Vec::new();
        }
        let min_common = (COMMON_WORD_FRACTION * max_common as f64).ceil() as usize;

        let mut out: Vec<Candidate> = common
            .into_iter()
            .filter(|&(_, c)| c >= min_common.max(1))
            .filter_map(|(id, _)| {
                let kf = db.keyframe(id)?;
                let score = bow.score(&kf.bow);
                (score >= self.config.min_bow_score).then_some(Candidate {
                    keyframe: id,
                    score,
                })
            })
            .collect();
        // Descending score; ties break on the lower id so the order is a
        // function of the map alone.
        out.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.keyframe.cmp(&b.keyframe))
        });
        out.truncate(self.config.max_candidates);
        out
    }

    /// Geometrically verify a candidate. `None` means reject.
    ///
    /// Descriptor matching against the candidate keyframe's *landmark-bearing*
    /// features, then P3P RANSAC on the resulting 2D-3D correspondences, then a
    /// Huber Gauss-Newton refinement. Rejects unless at least
    /// [`RelocConfig::min_inliers`] correspondences survive **and** the
    /// refinement produces an invertible information matrix — a pose we cannot
    /// put a covariance on is not a pose we are willing to emit (spec.md §4 L6).
    #[must_use]
    pub fn verify(
        &self,
        db: &MapDb,
        candidate: &Candidate,
        keypoints: &[Vec2],
        descriptors: &[BinaryDescriptor],
        k: &CameraIntrinsics,
        rng: &mut DeterministicRng,
    ) -> Option<Verified> {
        self.verify_detailed(db, candidate, keypoints, descriptors, k, rng)
            .ok()
    }

    /// [`Relocalizer::verify`], reporting *why* a candidate was rejected.
    ///
    /// Same decision, more information: `verify` is the frozen contract
    /// signature and simply discards the reason. See [`Rejection`] for what the
    /// reason is for.
    #[allow(clippy::result_large_err)] // `Rejection` is a handful of usizes
    pub fn verify_detailed(
        &self,
        db: &MapDb,
        candidate: &Candidate,
        keypoints: &[Vec2],
        descriptors: &[BinaryDescriptor],
        k: &CameraIntrinsics,
        rng: &mut DeterministicRng,
    ) -> std::result::Result<Verified, Rejection> {
        let kf = db
            .keyframe(candidate.keyframe)
            .ok_or(Rejection::UnknownKeyframe)?;

        // Only features already triangulated into a landmark can constrain a
        // pose, so match against that subset — but not the candidate's subset
        // alone. A single keyframe carries only the landmarks its own detector
        // happened to re-find, and matching against just those starved PnP one
        // or two correspondences short of `min_inliers` on real sequences
        // (measured on EuRoC MH_03: rejections clustered at 19 matches against
        // a required 20). The candidate's covisibility neighbourhood sees the
        // same place from adjacent poses, so its landmarks are legitimate
        // correspondences for the query too — the same expansion ORB-SLAM's
        // relocalization performs. Pool them, deduplicated by landmark id.
        let mut pool_ids: Vec<LandmarkId> = kf.landmarks.iter().flatten().copied().collect();
        if pool_ids.len() < 4 {
            return Err(Rejection::NoMappedLandmarks {
                landmarks: pool_ids.len(),
            });
        }
        // Neighbours ranked by how many landmarks they share with the
        // candidate. BTreeMap keeps the ranking deterministic on ties.
        let mut shared: std::collections::BTreeMap<KeyframeId, usize> = Default::default();
        for lid in &pool_ids {
            let Some(lm) = db.landmark(*lid) else {
                continue;
            };
            for &obs in &lm.observations {
                if obs != candidate.keyframe {
                    *shared.entry(obs).or_insert(0) += 1;
                }
            }
        }
        let mut neighbours: Vec<(KeyframeId, usize)> = shared.into_iter().collect();
        neighbours.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        for (nid, count) in neighbours.into_iter().take(MAX_COVIS_NEIGHBOURS) {
            if count < MIN_SHARED_LANDMARKS {
                break;
            }
            let Some(nkf) = db.keyframe(nid) else {
                continue;
            };
            pool_ids.extend(nkf.landmarks.iter().flatten().copied());
        }
        pool_ids.sort_unstable();
        pool_ids.dedup();

        // Each landmark contributes its canonical descriptor and 3D position.
        // `pool_lids` stays parallel so an inlier index maps back to its id.
        let mut map_descriptors = Vec::new();
        let mut map_points = Vec::new();
        let mut pool_lids = Vec::new();
        for lid in &pool_ids {
            let Some(lm) = db.landmark(*lid) else {
                continue;
            };
            map_descriptors.push(lm.descriptor);
            map_points.push(lm.position);
            pool_lids.push(*lid);
        }
        if map_descriptors.len() < 4 {
            return Err(Rejection::NoMappedLandmarks {
                landmarks: map_descriptors.len(),
            });
        }

        // Ratio-tested nearest neighbour, then one-to-one: two query features
        // claiming the same landmark are both suspect, so keep only the closer.
        let mut claim: Vec<Option<(usize, u32)>> = vec![None; map_descriptors.len()];
        for (qi, qd) in descriptors.iter().enumerate() {
            if qi >= keypoints.len() {
                break;
            }
            let Some((mi, best, second)) = nearest_two(qd, &map_descriptors) else {
                break;
            };
            if best > MAX_MATCH_HAMMING {
                continue;
            }
            if second != u32::MAX && (best as f64) >= MATCH_RATIO * second as f64 {
                continue;
            }
            match claim[mi] {
                Some((_, prev)) if prev <= best => {}
                _ => claim[mi] = Some((qi, best)),
            }
        }

        let mut world = Vec::new();
        let mut pixels = Vec::new();
        let mut pairs = Vec::new();
        for (mi, c) in claim.iter().enumerate() {
            if let Some((qi, _)) = c {
                world.push(map_points[mi]);
                pixels.push(keypoints[*qi]);
                pairs.push((*qi, pool_lids[mi]));
            }
        }
        // RANSAC needs a seed set, not the full quota. `min_inliers` is the
        // *final* gate, applied to reprojection inliers after the guided
        // second pass below; demanding it of the ratio-tested seed matches as
        // well starved verification one or two matches short on real
        // sequences (EuRoC MH_03: rejections clustered at 19 of 20) and reloc
        // never fired at all. P3P needs 4; below ~8 the RANSAC pose is too
        // often a fluke worth rejecting cheaply here.
        if world.len() < MIN_SEED_MATCHES {
            let reason = Rejection::TooFewMatches {
                matches: world.len(),
                required: MIN_SEED_MATCHES,
            };
            log::debug!("reloc rejected {}: {reason}", candidate.keyframe);
            return Err(reason);
        }

        let (pose, mask) = solve_pnp_ransac(
            &world,
            &pixels,
            k,
            self.config.ransac_threshold_px,
            self.config.ransac_iterations,
            rng,
        )
        .ok_or(Rejection::PnpFailed {
            matches: world.len(),
        })?;

        let mut inlier_idx: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &ok)| ok.then_some(i))
            .collect();
        // A RANSAC pose supported by almost nothing is not worth a guided pass.
        if inlier_idx.len() < 4 {
            let reason = Rejection::TooFewInliers {
                matches: world.len(),
                inliers: inlier_idx.len(),
                required: self.config.min_inliers,
            };
            log::debug!(
                "reloc rejected {}: {reason} (bow {:.3})",
                candidate.keyframe,
                candidate.score
            );
            return Err(reason);
        }

        // Projection-guided second pass — ORB-SLAM's SearchByProjection. The
        // coarse pose says where each pooled landmark *should* appear in the
        // query image; a query keypoint near that prediction with a compatible
        // descriptor is a correspondence the appearance-only stage missed. No
        // ratio test here: the spatial window already disambiguates, and the
        // ratio test is exactly what starved the seed stage once the pool
        // contained many near-duplicate views of the same scene.
        {
            let inv = pose.inverse();
            let mut claimed_query: Vec<bool> = vec![false; keypoints.len()];
            for &(qi, _) in claim.iter().flatten() {
                claimed_query[qi] = true;
            }
            for (mi, slot) in claim.iter_mut().enumerate() {
                if slot.is_some() {
                    continue;
                }
                let Some(predicted) = k.project(&inv.act(&map_points[mi])) else {
                    continue;
                };
                let r2 = GUIDED_RADIUS_PX * GUIDED_RADIUS_PX;
                let mut best: Option<(usize, u32)> = None;
                for (qi, kp) in keypoints.iter().enumerate() {
                    if claimed_query[qi] || (kp - predicted).norm_squared() > r2 {
                        continue;
                    }
                    let d = descriptors[qi].hamming(&map_descriptors[mi]);
                    if d <= MAX_MATCH_HAMMING && best.is_none_or(|(_, b)| d < b) {
                        best = Some((qi, d));
                    }
                }
                if let Some((qi, d)) = best {
                    claimed_query[qi] = true;
                    *slot = Some((qi, d));
                }
            }

            world.clear();
            pixels.clear();
            pairs.clear();
            for (mi, c) in claim.iter().enumerate() {
                if let Some((qi, _)) = c {
                    world.push(map_points[mi]);
                    pixels.push(keypoints[*qi]);
                    pairs.push((*qi, pool_lids[mi]));
                }
            }
            let mut guided_mask = vec![false; world.len()];
            let guided = count_inliers(
                &pose,
                &world,
                &pixels,
                k,
                self.config.ransac_threshold_px,
                &mut guided_mask,
            );
            inlier_idx = guided_mask
                .iter()
                .enumerate()
                .filter_map(|(i, &ok)| ok.then_some(i))
                .collect();
            debug_assert_eq!(guided, inlier_idx.len());
        }

        if inlier_idx.len() < self.config.min_inliers {
            let reason = Rejection::TooFewInliers {
                matches: world.len(),
                inliers: inlier_idx.len(),
                required: self.config.min_inliers,
            };
            log::debug!(
                "reloc rejected {}: {reason} (bow {:.3})",
                candidate.keyframe,
                candidate.score
            );
            return Err(reason);
        }

        let refined = refine_pose(
            pose,
            &world,
            &pixels,
            &inlier_idx,
            k,
            HUBER_SCALE * self.config.ransac_threshold_px,
        )
        .ok_or(Rejection::NoCovariance {
            inliers: inlier_idx.len(),
        })?;

        // Re-score against the refined pose: refinement can gain or lose a few.
        let mut final_mask = vec![false; world.len()];
        let final_inliers = count_inliers(
            &refined.pose,
            &world,
            &pixels,
            k,
            self.config.ransac_threshold_px,
            &mut final_mask,
        );
        if final_inliers < self.config.min_inliers {
            return Err(Rejection::TooFewInliers {
                matches: world.len(),
                inliers: final_inliers,
                required: self.config.min_inliers,
            });
        }

        Ok(Verified {
            keyframe: candidate.keyframe,
            pose: refined.pose,
            inliers: final_inliers,
            covariance: refined.covariance,
            correspondences: final_mask
                .iter()
                .enumerate()
                .filter_map(|(i, &ok)| ok.then_some(pairs[i]))
                .collect(),
        })
    }

    /// The only accept path: query, then verify every candidate in score order,
    /// returning the first that survives geometry.
    ///
    /// There is deliberately no variant that returns the top candidate without
    /// verifying it.
    #[allow(clippy::too_many_arguments)] // the signature is the frozen contract's
    #[must_use]
    pub fn relocalize(
        &self,
        db: &MapDb,
        bow: &BowVector,
        keypoints: &[Vec2],
        descriptors: &[BinaryDescriptor],
        k: &CameraIntrinsics,
        exclude_recent: usize,
        rng: &mut DeterministicRng,
    ) -> Option<Verified> {
        for candidate in self.query(db, bow, exclude_recent) {
            if let Some(v) = self.verify(db, &candidate, keypoints, descriptors, k, rng) {
                return Some(v);
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// PnP
// ---------------------------------------------------------------------------

/// Outcome of the Gauss-Newton refinement.
struct Refined {
    pose: Se3,
    covariance: Mat6,
}

/// Reprojection inlier count for a pose. Also fills `mask`.
///
/// [`CameraIntrinsics::project`] returns `None` behind the image plane, so the
/// cheirality check is free — a landmark behind the camera can otherwise
/// project onto a perfectly plausible pixel.
fn count_inliers(
    pose: &Se3,
    world: &[Vec3],
    pixels: &[Vec2],
    k: &CameraIntrinsics,
    threshold_px: f64,
    mask: &mut [bool],
) -> usize {
    let inv = pose.inverse();
    let t2 = threshold_px * threshold_px;
    let mut n = 0;
    for (i, (p, z)) in world.iter().zip(pixels.iter()).enumerate() {
        let ok = match k.project(&inv.act(p)) {
            Some(proj) => (proj - z).norm_squared() <= t2,
            None => false,
        };
        if i < mask.len() {
            mask[i] = ok;
        }
        if ok {
            n += 1;
        }
    }
    n
}

/// P3P RANSAC over `DeterministicRng`. Returns `(T_world_camera, inlier mask)`.
fn solve_pnp_ransac(
    world: &[Vec3],
    pixels: &[Vec2],
    k: &CameraIntrinsics,
    threshold_px: f64,
    max_iterations: usize,
    rng: &mut DeterministicRng,
) -> Option<(Se3, Vec<bool>)> {
    let n = world.len();
    if n < 4 || pixels.len() != n {
        return None;
    }
    let bearings: Vec<Vec3> = pixels.iter().map(|p| k.unproject_bearing(*p)).collect();

    let mut best_pose: Option<Se3> = None;
    let mut best_count = 0usize;
    let mut mask = vec![false; n];
    let mut scratch = vec![false; n];
    let mut sample = Vec::with_capacity(3);
    let mut budget = max_iterations.max(1);

    let mut it = 0;
    while it < budget {
        it += 1;
        rng.sample_distinct(n, 3, &mut sample);
        if sample.len() < 3 {
            break;
        }
        let b = [
            bearings[sample[0]],
            bearings[sample[1]],
            bearings[sample[2]],
        ];
        let w = [world[sample[0]], world[sample[1]], world[sample[2]]];
        for pose in p3p(&b, &w) {
            let count = count_inliers(&pose, world, pixels, k, threshold_px, &mut scratch);
            if count > best_count {
                best_count = count;
                best_pose = Some(pose);
                mask.copy_from_slice(&scratch);
                // Adaptive stopping, standard RANSAC: a deterministic function
                // of the data, so replay stays bit-exact.
                let w_in = best_count as f64 / n as f64;
                let p_clean = w_in * w_in * w_in;
                if p_clean >= 1.0 - 1e-12 {
                    budget = it;
                } else if p_clean > 0.0 {
                    let needed = (1.0 - 0.999f64).ln() / (1.0 - p_clean).ln();
                    if needed.is_finite() && needed >= 0.0 {
                        budget = budget.min(it + needed.ceil() as usize);
                    }
                }
            }
        }
    }
    best_pose.filter(|_| best_count >= 4).map(|p| (p, mask))
}

/// Grunert's P3P. Returns up to four `T_world_camera` solutions.
///
/// `bearings` must be unit vectors in camera coordinates. Each returned pose is
/// checked against the three original distance equations, so a solution that
/// only satisfies the quartic — a spurious root, or a coefficient typo — is
/// discarded rather than fed to RANSAC.
#[must_use]
pub fn p3p(bearings: &[Vec3; 3], world: &[Vec3; 3]) -> Vec<Se3> {
    let a = (world[1] - world[2]).norm();
    let b = (world[0] - world[2]).norm();
    let c = (world[0] - world[1]).norm();
    if a < 1e-9 || b < 1e-9 || c < 1e-9 {
        return Vec::new(); // coincident control points
    }
    // Collinear control points leave the rotation about their common line
    // unconstrained. Reject rather than return an arbitrary one.
    let area2 = (world[1] - world[0]).cross(&(world[2] - world[0])).norm();
    if area2 < 1e-6 * (a * b).max(b * c).max(a * c) {
        return Vec::new();
    }

    let cos_alpha = bearings[1].dot(&bearings[2]).clamp(-1.0, 1.0);
    let cos_beta = bearings[0].dot(&bearings[2]).clamp(-1.0, 1.0);
    let cos_gamma = bearings[0].dot(&bearings[1]).clamp(-1.0, 1.0);

    let (a2, b2, c2) = (a * a, b * b, c * c);
    let k1 = (a2 - c2) / b2;
    let k2 = (a2 + c2) / b2;

    // Haralick et al. (IJCV 13(3), 1994), tabulation of Grunert's 1841 solution.
    let a4 = (k1 - 1.0).powi(2) - 4.0 * c2 * cos_alpha * cos_alpha / b2;
    let a3 = 4.0
        * (k1 * (1.0 - k1) * cos_beta - (1.0 - k2) * cos_alpha * cos_gamma
            + 2.0 * c2 * cos_alpha * cos_alpha * cos_beta / b2);
    let a2_ = 2.0
        * (k1 * k1 - 1.0
            + 2.0 * k1 * k1 * cos_beta * cos_beta
            + 2.0 * ((b2 - c2) / b2) * cos_alpha * cos_alpha
            - 4.0 * k2 * cos_alpha * cos_beta * cos_gamma
            + 2.0 * ((b2 - a2) / b2) * cos_gamma * cos_gamma);
    let a1 = 4.0
        * (-k1 * (1.0 + k1) * cos_beta + 2.0 * (a2 / b2) * cos_gamma * cos_gamma * cos_beta
            - (1.0 - k2) * cos_alpha * cos_gamma);
    let a0 = (1.0 + k1).powi(2) - 4.0 * (a2 / b2) * cos_gamma * cos_gamma;

    let mut out = Vec::new();
    for v in real_roots(&[a0, a1, a2_, a3, a4]) {
        if v <= 0.0 {
            continue;
        }
        let denom = 2.0 * (cos_gamma - v * cos_alpha);
        if denom.abs() < 1e-12 {
            continue;
        }
        let u = ((k1 - 1.0) * v * v - 2.0 * k1 * cos_beta * v + 1.0 + k1) / denom;
        if u <= 0.0 {
            continue;
        }
        let denom_s1 = 1.0 + u * u - 2.0 * u * cos_gamma;
        if denom_s1 <= 1e-12 {
            continue;
        }
        let s1 = (c2 / denom_s1).sqrt();
        let (s2, s3) = (u * s1, v * s1);
        if !(s1.is_finite() && s2.is_finite() && s3.is_finite()) {
            continue;
        }

        // Independent check against the two equations the quartic did not
        // directly encode. A coefficient error shows up here as "no solutions",
        // never as a wrong pose.
        let ra = (s2 * s2 + s3 * s3 - 2.0 * s2 * s3 * cos_alpha - a2).abs();
        let rb = (s1 * s1 + s3 * s3 - 2.0 * s1 * s3 * cos_beta - b2).abs();
        let tol = 1e-6 * (a2 + b2 + c2);
        if ra > tol || rb > tol {
            continue;
        }

        let cam = [bearings[0] * s1, bearings[1] * s2, bearings[2] * s3];
        // Absolute orientation with the scale locked: this is the standard
        // finish to Grunert, and `umeyama` is already tested in wslam-core.
        let Some(al) = umeyama(world, &cam, false) else {
            continue;
        };
        let t_cam_world = Se3::new(al.transform.rotation(), al.transform.translation());
        out.push(t_cam_world.inverse());
    }
    out
}

/// Motion-only Gauss-Newton with a Huber loss, plus the pose covariance.
fn refine_pose(
    initial: Se3,
    world: &[Vec3],
    pixels: &[Vec2],
    inliers: &[usize],
    k: &CameraIntrinsics,
    huber_px: f64,
) -> Option<Refined> {
    if inliers.len() < 4 {
        return None;
    }
    let mut pose = initial;
    for _ in 0..10 {
        let inv = pose.inverse();
        let mut h = Mat6::zeros();
        let mut g = Vec6::zeros();
        let mut used = 0;
        for &i in inliers {
            let p_cam = inv.act(&world[i]);
            if p_cam.z <= 1e-6 {
                continue;
            }
            let r = k.project_unchecked(&p_cam) - pixels[i];
            let j = projection_jacobian(k, &p_cam);
            // Huber: linear beyond `huber_px`, so a straggler cannot drag the
            // solution the way a squared loss would.
            let norm = r.norm();
            let w = if norm <= huber_px || norm == 0.0 {
                1.0
            } else {
                huber_px / norm
            };
            h += j.transpose() * (w * j);
            g += j.transpose() * (w * r);
            used += 1;
        }
        if used < 4 {
            return None;
        }
        let delta = -h.try_inverse()? * g;
        pose = pose.plus(&delta);
        if delta.norm() < 1e-12 {
            break;
        }
    }

    // Covariance from the unweighted normal equations at the solution, scaled
    // by the measured residual variance: sigma^2 (J^T J)^-1 with
    // sigma^2 = sum(r^2) / (2n - 6), the usual least-squares estimate.
    let inv = pose.inverse();
    let mut h = Mat6::zeros();
    let mut sse = 0.0;
    let mut used = 0usize;
    for &i in inliers {
        let p_cam = inv.act(&world[i]);
        if p_cam.z <= 1e-6 {
            continue;
        }
        let r = k.project_unchecked(&p_cam) - pixels[i];
        let j = projection_jacobian(k, &p_cam);
        h += j.transpose() * j;
        sse += r.norm_squared();
        used += 1;
    }
    let dof = (2 * used).saturating_sub(6);
    if used < 4 || dof == 0 {
        return None;
    }
    let sigma2 = (sse / dof as f64).max(1e-12);
    let cov = symmetrize(&(h.try_inverse()? * sigma2));
    let cov = enforce_psd(&cov, 1e-18)?;
    Some(Refined {
        pose,
        covariance: cov,
    })
}

/// `d(pixel) / d(delta)` for `T_world_camera` under right-perturbation.
///
/// With `p_cam = T^-1 P` and `T' = T exp(delta)`, `p_cam' = exp(-delta) p_cam`,
/// so `d p_cam / d[rho; phi] = [-I, hat(p_cam)]`. Chaining the pinhole
/// projection Jacobian gives the 2x6 block, in `[translation; rotation]` order
/// to match the workspace covariance convention.
fn projection_jacobian(k: &CameraIntrinsics, p_cam: &Vec3) -> nalgebra::Matrix2x6<f64> {
    let dpi = k.projection_jacobian(p_cam);
    let mut d = nalgebra::Matrix3x6::<f64>::zeros();
    d.fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&(-nalgebra::Matrix3::<f64>::identity()));
    d.fixed_view_mut::<3, 3>(0, 3).copy_from(&hat(p_cam));
    dpi * d
}

// ---------------------------------------------------------------------------
// Real polynomial roots
// ---------------------------------------------------------------------------

/// Horner evaluation. `c` is in ascending coefficient order.
fn poly_eval(c: &[f64], x: f64) -> f64 {
    let mut v = 0.0;
    for &a in c.iter().rev() {
        v = v * x + a;
    }
    v
}

/// Real roots of a polynomial, ascending coefficient order.
///
/// Rolle recursion rather than Ferrari's closed form: the real roots of `p` are
/// separated by the real roots of `p'`, so each bracket is monotone and a
/// bisection cannot miss or duplicate a root. Ferrari's radicals have branch
/// cases that are easy to get subtly wrong and hard to test; this has one code
/// path for every degree, and the degree-1 base case is trivially correct.
fn real_roots(coeffs: &[f64]) -> Vec<f64> {
    let scale = coeffs.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    if !(scale.is_finite() && scale > 0.0) || coeffs.iter().any(|v| !v.is_finite()) {
        return Vec::new();
    }
    let mut n = coeffs.len();
    while n > 1 && coeffs[n - 1].abs() < 1e-12 * scale {
        n -= 1;
    }
    let c = &coeffs[..n];
    match c.len() {
        0 | 1 => Vec::new(),
        2 => vec![-c[0] / c[1]],
        _ => {
            let deriv: Vec<f64> = c
                .iter()
                .enumerate()
                .skip(1)
                .map(|(i, &a)| a * i as f64)
                .collect();
            let lead = c[c.len() - 1].abs();
            // Cauchy bound: every real root lies in [-R, R].
            let r = 1.0 + c[..c.len() - 1].iter().fold(0.0f64, |m, v| m.max(v.abs())) / lead;

            let mut pts = vec![-r];
            for cr in real_roots(&deriv) {
                if cr > -r && cr < r {
                    pts.push(cr);
                }
            }
            pts.push(r);
            pts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let mut roots: Vec<f64> = Vec::new();
            for w in pts.windows(2) {
                let (lo, hi) = (w[0], w[1]);
                let (flo, fhi) = (poly_eval(c, lo), poly_eval(c, hi));
                if flo == 0.0 {
                    roots.push(lo);
                }
                if flo * fhi < 0.0 {
                    roots.push(bisect(c, lo, hi));
                }
            }
            if let Some(&last) = pts.last() {
                if poly_eval(c, last) == 0.0 {
                    roots.push(last);
                }
            }
            // A repeated root sits exactly at a critical point and produces no
            // sign change; pick it up by residual instead.
            for &cr in pts.iter().skip(1).take(pts.len().saturating_sub(2)) {
                let v = poly_eval(c, cr).abs();
                let mag = c
                    .iter()
                    .enumerate()
                    .map(|(i, &a)| a.abs() * cr.abs().powi(i as i32))
                    .sum::<f64>();
                if v <= 1e-10 * mag.max(1.0) {
                    roots.push(cr);
                }
            }

            roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            roots.dedup_by(|a, b| (*a - *b).abs() <= 1e-9 * (1.0 + a.abs()));
            roots
        }
    }
}

/// Bisection on a bracket with a sign change. 100 halvings exhausts `f64`.
fn bisect(c: &[f64], mut lo: f64, mut hi: f64) -> f64 {
    let flo = poly_eval(c, lo);
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if mid == lo || mid == hi {
            break;
        }
        if poly_eval(c, mid) * flo <= 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth;
    use crate::vocabulary::Vocabulary;
    use approx::assert_relative_eq;
    use wslam_core::covariance::is_valid_covariance;
    use wslam_core::So3;

    fn intrinsics() -> CameraIntrinsics {
        CameraIntrinsics::from_focal(500.0, 640, 480)
    }

    // ---- polynomial roots -------------------------------------------------

    #[test]
    fn real_roots_recover_known_factorisations() {
        // (x-1)(x-2)(x-3)(x+4) = x^4 - 2x^3 - 13x^2 + 38x - 24
        let r = real_roots(&[-24.0, 38.0, -13.0, -2.0, 1.0]);
        assert_eq!(r.len(), 4, "{r:?}");
        for (got, want) in r.iter().zip([-4.0, 1.0, 2.0, 3.0]) {
            assert_relative_eq!(got, &want, epsilon = 1e-9);
        }
        // (x^2 + 1)(x - 5): only one real root.
        let r = real_roots(&[-5.0, 1.0, -5.0, 1.0]);
        assert_eq!(r.len(), 1);
        assert_relative_eq!(r[0], 5.0, epsilon = 1e-9);
        // Linear and constant.
        assert_relative_eq!(real_roots(&[3.0, -1.5])[0], 2.0, epsilon = 1e-12);
        assert!(real_roots(&[7.0]).is_empty());
        // Degenerate coefficient vectors must not panic.
        assert!(real_roots(&[0.0, 0.0, 0.0]).is_empty());
        assert!(real_roots(&[f64::NAN, 1.0]).is_empty());
        assert!(real_roots(&[]).is_empty());
    }

    #[test]
    fn real_roots_find_a_repeated_root() {
        // (x-2)^2 (x+3) = x^3 - x^2 - 8x + 12
        let r = real_roots(&[12.0, -8.0, -1.0, 1.0]);
        assert!(r.iter().any(|v| (v - 2.0).abs() < 1e-6), "{r:?}");
        assert!(r.iter().any(|v| (v + 3.0).abs() < 1e-6), "{r:?}");
    }

    #[test]
    fn real_roots_handle_a_degenerate_leading_coefficient() {
        // 1e-20 x^4 + x^2 - 4 is numerically a quadratic; the tiny quartic term
        // must be trimmed rather than producing 1e10-sized spurious roots.
        let r = real_roots(&[-4.0, 0.0, 1.0, 0.0, 1e-20]);
        assert_eq!(r.len(), 2, "{r:?}");
        assert_relative_eq!(r[0], -2.0, epsilon = 1e-8);
        assert_relative_eq!(r[1], 2.0, epsilon = 1e-8);
    }

    // ---- P3P --------------------------------------------------------------

    /// Bearings for a known pose, so the solver has a ground truth to recover.
    fn p3p_case(pose: &Se3, world: &[Vec3; 3]) -> [Vec3; 3] {
        let inv = pose.inverse();
        [
            inv.act(&world[0]).normalize(),
            inv.act(&world[1]).normalize(),
            inv.act(&world[2]).normalize(),
        ]
    }

    #[test]
    fn p3p_recovers_a_known_pose() {
        let mut rng = DeterministicRng::new("p3p", 20260801);
        let mut recovered = 0;
        for _ in 0..40 {
            let truth = Se3::new(
                So3::exp(&Vec3::new(
                    rng.uniform_range(-0.6, 0.6),
                    rng.uniform_range(-0.6, 0.6),
                    rng.uniform_range(-0.6, 0.6),
                )),
                Vec3::new(
                    rng.uniform_range(-2.0, 2.0),
                    rng.uniform_range(-2.0, 2.0),
                    rng.uniform_range(-2.0, 2.0),
                ),
            );
            // Points comfortably in front of the camera.
            let world = [
                truth.act(&Vec3::new(0.4, -0.3, 3.0)),
                truth.act(&Vec3::new(-0.5, 0.6, 4.2)),
                truth.act(&Vec3::new(0.9, 0.8, 2.6)),
            ];
            let bearings = p3p_case(&truth, &world);
            let solutions = p3p(&bearings, &world);
            assert!(!solutions.is_empty(), "P3P returned nothing");
            assert!(
                solutions.len() <= 4,
                "P3P returned {} solutions",
                solutions.len()
            );
            if solutions.iter().any(|s| s.minus(&truth).norm() < 1e-6) {
                recovered += 1;
            }
        }
        assert_eq!(recovered, 40, "true pose missing from the solution set");
    }

    #[test]
    fn p3p_rejects_degenerate_configurations() {
        let truth = Se3::from_translation(Vec3::new(0.0, 0.0, -3.0));
        // Collinear control points: the rotation about their line is free.
        let collinear = [
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        ];
        assert!(p3p(&p3p_case(&truth, &collinear), &collinear).is_empty());
        // Coincident control points.
        let coincident = [Vec3::zeros(), Vec3::zeros(), Vec3::new(1.0, 0.0, 0.0)];
        assert!(p3p(&p3p_case(&truth, &coincident), &coincident).is_empty());
    }

    // ---- PnP RANSAC -------------------------------------------------------

    /// `n` world points and their projections under `truth`, with `outliers` of
    /// the pixels replaced by nonsense.
    fn pnp_case(
        truth: &Se3,
        n: usize,
        outliers: usize,
        rng: &mut DeterministicRng,
    ) -> (Vec<Vec3>, Vec<Vec2>) {
        let k = intrinsics();
        let inv = truth.inverse();
        let mut world = Vec::new();
        let mut pixels = Vec::new();
        while world.len() < n {
            let p = Vec3::new(
                rng.uniform_range(-3.0, 3.0),
                rng.uniform_range(-2.0, 2.0),
                rng.uniform_range(2.0, 8.0),
            );
            let p_world = truth.act(&p);
            if let Some(px) = k.project(&inv.act(&p_world)) {
                if k.contains(px, 2.0) {
                    world.push(p_world);
                    pixels.push(px);
                }
            }
        }
        for px in pixels.iter_mut().take(outliers) {
            *px = Vec2::new(rng.uniform_range(0.0, 640.0), rng.uniform_range(0.0, 480.0));
        }
        (world, pixels)
    }

    #[test]
    fn pnp_ransac_recovers_the_pose_under_heavy_outlier_contamination() {
        let mut rng = DeterministicRng::new("ransac", 7);
        let truth = Se3::new(
            So3::exp(&Vec3::new(0.15, -0.25, 0.1)),
            Vec3::new(0.7, -0.4, 1.2),
        );
        let (world, pixels) = pnp_case(&truth, 100, 40, &mut rng);
        let (pose, mask) =
            solve_pnp_ransac(&world, &pixels, &intrinsics(), 2.0, 500, &mut rng).expect("solved");
        assert!(pose.minus(&truth).norm() < 1e-3, "{:?}", pose.minus(&truth));
        // The 60 clean correspondences should all be inliers.
        assert!(mask.iter().filter(|&&m| m).count() >= 58);
    }

    #[test]
    fn pnp_ransac_is_deterministic_for_a_fixed_seed() {
        let mut setup = DeterministicRng::new("setup", 3);
        let truth = Se3::new(
            So3::exp(&Vec3::new(0.1, 0.2, -0.1)),
            Vec3::new(0.2, 0.1, 0.5),
        );
        let (world, pixels) = pnp_case(&truth, 60, 20, &mut setup);
        let run = |seed| {
            let mut rng = DeterministicRng::new("ransac", seed);
            solve_pnp_ransac(&world, &pixels, &intrinsics(), 2.0, 200, &mut rng)
                .map(|(p, m)| (p.log(), m.iter().filter(|&&x| x).count()))
        };
        assert_eq!(run(11), run(11));
    }

    #[test]
    fn pnp_ransac_refuses_underdetermined_input() {
        let mut rng = DeterministicRng::new("t", 1);
        let world = vec![Vec3::new(0.0, 0.0, 2.0); 3];
        let pixels = vec![Vec2::new(320.0, 240.0); 3];
        assert!(solve_pnp_ransac(&world, &pixels, &intrinsics(), 2.0, 50, &mut rng).is_none());
        assert!(solve_pnp_ransac(&[], &[], &intrinsics(), 2.0, 50, &mut rng).is_none());
    }

    #[test]
    fn refinement_reduces_reprojection_error_and_yields_a_valid_covariance() {
        let mut rng = DeterministicRng::new("refine", 5);
        let truth = Se3::new(
            So3::exp(&Vec3::new(0.05, 0.1, -0.05)),
            Vec3::new(0.3, 0.0, 0.2),
        );
        let (world, mut pixels) = pnp_case(&truth, 80, 0, &mut rng);
        for p in pixels.iter_mut() {
            *p += Vec2::new(rng.normal() * 0.5, rng.normal() * 0.5);
        }
        let inliers: Vec<usize> = (0..world.len()).collect();
        // Start from a perturbed pose so the refinement has work to do.
        let start = truth.plus(&Vec6::new(0.05, -0.04, 0.06, 0.01, -0.01, 0.008));
        let refined =
            refine_pose(start, &world, &pixels, &inliers, &intrinsics(), 3.0).expect("refined");
        assert!(
            refined.pose.minus(&truth).norm() < start.minus(&truth).norm() / 5.0,
            "refinement did not converge toward truth"
        );
        assert!(is_valid_covariance(&refined.covariance, 1e-9));
        // 0.5 px noise on 80 points is a well-conditioned solve: sub-centimetre.
        assert!(refined.covariance[(0, 0)].sqrt() < 0.05);
        assert!(refined.covariance.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn refinement_refuses_too_few_correspondences() {
        let mut rng = DeterministicRng::new("t", 2);
        let truth = Se3::identity();
        let (world, pixels) = pnp_case(&truth, 10, 0, &mut rng);
        assert!(refine_pose(truth, &world, &pixels, &[0, 1, 2], &intrinsics(), 3.0).is_none());
    }

    #[test]
    fn projection_jacobian_matches_central_differences() {
        let k = intrinsics();
        let pose = Se3::new(
            So3::exp(&Vec3::new(0.2, -0.3, 0.15)),
            Vec3::new(0.4, -0.2, 0.9),
        );
        let p_world = Vec3::new(0.7, -0.5, 3.4);
        let p_cam = pose.inverse().act(&p_world);
        let j = projection_jacobian(&k, &p_cam);

        let eps = 1e-7;
        for i in 0..6 {
            let mut d = Vec6::zeros();
            d[i] = eps;
            let plus = k.project_unchecked(&pose.plus(&d).inverse().act(&p_world));
            let minus = k.project_unchecked(&pose.plus(&(-d)).inverse().act(&p_world));
            let num = (plus - minus) / (2.0 * eps);
            assert_relative_eq!(j[(0, i)], num.x, epsilon = 1e-4, max_relative = 1e-5);
            assert_relative_eq!(j[(1, i)], num.y, epsilon = 1e-4, max_relative = 1e-5);
        }
    }

    // ---- relocalization ---------------------------------------------------

    #[test]
    fn query_ranks_the_source_keyframe_first() {
        let (db, scene) = synth::corridor_map(12, 20260801);
        let reloc = Relocalizer::new(RelocConfig::default());
        for target in [0usize, 5, 11] {
            let obs = synth::observe_noiseless(&scene, scene.poses[target]);
            let bow = db.transform(&obs.descriptors);
            let candidates = reloc.query(&db, &bow, 0);
            assert!(!candidates.is_empty());
            assert_eq!(
                candidates[0].keyframe,
                KeyframeId(target as u64),
                "top candidate {:?} for keyframe {target}",
                candidates[0]
            );
            assert!(
                candidates[0].score > 0.9,
                "self score {}",
                candidates[0].score
            );
        }
    }

    #[test]
    fn query_excludes_the_most_recent_keyframes() {
        let (db, scene) = synth::corridor_map(12, 4);
        let reloc = Relocalizer::new(RelocConfig {
            min_bow_score: 0.0,
            max_candidates: 50,
            ..RelocConfig::default()
        });
        let obs = synth::observe_noiseless(&scene, scene.poses[11]);
        let bow = db.transform(&obs.descriptors);
        let all = reloc.query(&db, &bow, 0);
        assert!(all.iter().any(|c| c.keyframe == KeyframeId(11)));
        let trimmed = reloc.query(&db, &bow, 4);
        assert!(
            trimmed.iter().all(|c| c.keyframe.0 < 8),
            "excluded keyframes came back: {trimmed:?}"
        );
    }

    #[test]
    fn query_on_an_empty_map_or_empty_vector_returns_nothing() {
        let reloc = Relocalizer::new(RelocConfig::default());
        let empty_db = MapDb::new(std::sync::Arc::new(Vocabulary::empty()));
        assert!(reloc.query(&empty_db, &BowVector::empty(), 0).is_empty());
        let (db, scene) = synth::corridor_map(4, 1);
        let _ = scene;
        assert!(reloc.query(&db, &BowVector::empty(), 0).is_empty());
        // Excluding everything leaves nothing to match.
        let obs = synth::observe_noiseless(&scene, scene.poses[0]);
        let bow = db.transform(&obs.descriptors);
        assert!(reloc.query(&db, &bow, 4).is_empty());
    }

    #[test]
    fn relocalizes_into_its_own_keyframe_with_a_pose_close_to_truth() {
        let (db, scene) = synth::corridor_map(12, 20260801);
        let reloc = Relocalizer::new(RelocConfig::default());
        let mut rng = DeterministicRng::new("reloc", 99);

        for target in [1usize, 6, 10] {
            let truth = scene.poses[target];
            let obs = synth::observe(&scene, truth, &mut rng);
            let bow = db.transform(&obs.descriptors);
            let v = reloc
                .relocalize(
                    &db,
                    &bow,
                    &obs.keypoints,
                    &obs.descriptors,
                    &scene.intrinsics,
                    0,
                    &mut rng,
                )
                .unwrap_or_else(|| panic!("failed to relocalize into keyframe {target}"));

            assert_eq!(v.keyframe, KeyframeId(target as u64));
            assert!(v.inliers >= RelocConfig::default().min_inliers);
            let err = v.pose.minus(&truth);
            let translation = err.fixed_rows::<3>(0).norm();
            let rotation = err.fixed_rows::<3>(3).norm();
            assert!(translation < 0.02, "translation error {translation} m");
            assert!(rotation < 0.005, "rotation error {rotation} rad");
            assert!(is_valid_covariance(&v.covariance, 1e-9));
        }
    }

    #[test]
    fn relocalizes_from_a_viewpoint_between_two_keyframes() {
        // The real case: the camera comes back on a *different* path, so the
        // query view sits between mapped keyframes (spec.md §6 L4).
        let (db, scene) = synth::corridor_map(12, 5150);
        let reloc = Relocalizer::new(RelocConfig::default());
        let mut rng = DeterministicRng::new("reloc-between", 3);
        let truth = scene.poses[4].interpolate(&scene.poses[5], 0.5);
        let obs = synth::observe(&scene, truth, &mut rng);
        let bow = db.transform(&obs.descriptors);
        let v = reloc
            .relocalize(
                &db,
                &bow,
                &obs.keypoints,
                &obs.descriptors,
                &scene.intrinsics,
                0,
                &mut rng,
            )
            .expect("relocalized");
        let err = v.pose.minus(&truth);
        assert!(err.fixed_rows::<3>(0).norm() < 0.05, "{err:?}");
    }

    /// **Release gate.** spec.md §6 L4: *"False-positive rate of place
    /// recognition, measured separately and reported prominently ... Target
    /// zero."* A query taken from a completely different scene must never yield
    /// a verified pose, even with the bag-of-words threshold disabled so that
    /// every keyframe in the map is offered up for verification.
    #[test]
    fn false_positive_place_recognition_is_rejected_by_geometric_verification() {
        let (db, scene) = synth::corridor_map(12, 20260801);
        // min_bow_score = 0 and max_candidates = everything: this test measures
        // the *geometry*, not the appearance threshold. Every keyframe in the
        // map is offered up for verification on every trial.
        let reloc = Relocalizer::new(RelocConfig {
            min_bow_score: 0.0,
            max_candidates: 64,
            ..RelocConfig::default()
        });
        let mut rng = DeterministicRng::new("false-positive", 20260801);

        let mut trials = 0;
        let mut adversarial = 0;
        // A trial only counts as exercising the gate if some candidate got all
        // the way to PnP. `Rejection::reached_geometry` is true exactly for the
        // stages after descriptor matching, and matching only hands over when
        // it has produced at least `min_inliers` correspondences — so this
        // counter is a direct measure of "the geometry was given enough
        // evidence to accept, and refused".
        let mut adversarial_reaching_geometry = 0;
        let mut most_correspondences = 0usize;
        let mut most_inliers = 0usize;

        for seed in 0..12u64 {
            // (a) A different room: different landmarks, different descriptors.
            //     Rejected at matching, which is the cheap half of the gate.
            // (b) The adversarial case: the *same* descriptors, permuted onto
            //     different landmarks. Bag-of-words says "definitely here",
            //     every descriptor match succeeds, and only PnP can tell that
            //     the correspondences are nonsense. This is the failure mode
            //     spec.md §5 calls irrecoverable.
            let unrelated = synth::Scene::new(1_000_000 + seed, 6);
            let doppelganger = scene.shuffled_appearance(seed);
            for (which, other) in [(false, &unrelated), (true, &doppelganger)] {
                for pose in &other.poses {
                    let obs = synth::observe(other, *pose, &mut rng);
                    let bow = db.transform(&obs.descriptors);
                    trials += 1;

                    let candidates = reloc.query(&db, &bow, 0);
                    assert!(
                        !candidates.is_empty(),
                        "nothing was offered for verification, so this trial \
                         tested nothing (seed {seed})"
                    );

                    let mut reached_geometry = false;
                    for candidate in &candidates {
                        match reloc.verify_detailed(
                            &db,
                            candidate,
                            &obs.keypoints,
                            &obs.descriptors,
                            &other.intrinsics,
                            &mut rng,
                        ) {
                            Ok(v) => panic!(
                                "FALSE POSITIVE ({}) seed {seed}: verified into {} \
                                 with {} inliers at bow score {:.3}",
                                if which { "doppelganger" } else { "unrelated" },
                                v.keyframe,
                                v.inliers,
                                candidate.score
                            ),
                            Err(r) => {
                                reached_geometry |= r.reached_geometry();
                                match r {
                                    Rejection::TooFewMatches { matches, .. }
                                    | Rejection::PnpFailed { matches } => {
                                        most_correspondences = most_correspondences.max(matches);
                                    }
                                    Rejection::TooFewInliers {
                                        matches, inliers, ..
                                    } => {
                                        most_correspondences = most_correspondences.max(matches);
                                        most_inliers = most_inliers.max(inliers);
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                    if which {
                        adversarial += 1;
                        adversarial_reaching_geometry += usize::from(reached_geometry);
                    }
                }
            }
        }

        assert!(
            trials >= 100,
            "the gate must actually have run: {trials} trials"
        );
        assert!(adversarial >= 48, "{adversarial} adversarial trials");

        // The precondition that matters, and the one this test used to get
        // wrong. It asserted `bow score > 0.3` on every impostor trial, which
        // (a) is not achievable — the impostor's best score ranges over
        // 0.292..0.478, so the bound fails outright — and (b) would not have
        // shown anything if it were, because a *genuinely different room*
        // scores 0.23..0.40 against this map. With 245 landmarks quantised into
        // 245 words, any 70-descriptor query collides with ~70*70/245 = 20 of a
        // keyframe's words by chance, so no bag-of-words threshold separates
        // the impostor from an unrelated scene at this fixture size and 0.3
        // discriminated nothing.
        //
        // What the comment above actually claims is that PnP is the thing doing
        // the rejecting, and that is directly checkable: every impostor trial
        // must reach the geometry, which by construction means the matcher
        // handed PnP at least `min_inliers` correspondences and PnP threw them
        // out anyway.
        assert_eq!(
            adversarial_reaching_geometry,
            adversarial,
            "{} of {adversarial} impostor trials were rejected before PnP ran, so \
             the geometric gate was not the thing under test",
            adversarial - adversarial_reaching_geometry
        );
        assert!(
            most_correspondences >= RelocConfig::default().min_inliers,
            "PnP was never offered enough correspondences to accept even in \
             principle: {most_correspondences}"
        );
        // ... and it was never close. If this ever climbs toward `min_inliers`
        // the margin is gone and the threshold needs raising, which spec.md §5
        // names as the response to any observed false positive.
        assert!(
            most_inliers * 2 < RelocConfig::default().min_inliers,
            "an impostor got {most_inliers} inliers against a bar of {}; the \
             false-positive margin has eroded",
            RelocConfig::default().min_inliers
        );
    }

    #[test]
    fn verification_rejects_a_high_score_candidate_with_inconsistent_geometry() {
        // The adversarial case the BoW score cannot catch: the *exact*
        // descriptors of a mapped keyframe, so the appearance score is 1.0, but
        // attached to permuted pixel locations. Only geometry can reject this,
        // and it must.
        let (db, scene) = synth::corridor_map(12, 31337);
        let reloc = Relocalizer::new(RelocConfig::default());
        let mut rng = DeterministicRng::new("scramble", 4);

        let kf = db.keyframe(KeyframeId(6)).unwrap();
        let descriptors = kf.descriptors.clone();
        let mut keypoints = kf.keypoints.clone();
        rng.shuffle(&mut keypoints);

        let bow = db.transform(&descriptors);
        let candidate = Candidate {
            keyframe: KeyframeId(6),
            score: bow.score(&kf.bow),
        };
        assert!(candidate.score > 0.99, "setup: score {}", candidate.score);

        assert!(
            reloc
                .verify(
                    &db,
                    &candidate,
                    &keypoints,
                    &descriptors,
                    &scene.intrinsics,
                    &mut rng
                )
                .is_none(),
            "a perfect appearance score got through without consistent geometry"
        );
        // ... while the unscrambled version of exactly the same query passes,
        // so the rejection is about the geometry and not about the setup.
        assert!(reloc
            .verify(
                &db,
                &candidate,
                &kf.keypoints.clone(),
                &descriptors,
                &scene.intrinsics,
                &mut rng
            )
            .is_some());
    }

    #[test]
    fn rejection_reasons_name_the_stage_that_did_the_rejecting() {
        // Every stage must be reachable and must identify itself, because the
        // whole value of the reason is telling "appearance threw it out" apart
        // from "the geometry threw it out" (spec.md §8's rejected-candidate
        // overlay, and the release gate above).
        let (mut db, scene) = synth::corridor_map(8, 4711);
        let reloc = Relocalizer::new(RelocConfig::default());
        let mut rng = DeterministicRng::new("reasons", 2);
        let obs = synth::observe(&scene, scene.poses[3], &mut rng);
        let at = |id| Candidate {
            keyframe: id,
            score: 1.0,
        };
        let run = |reloc: &Relocalizer,
                   db: &MapDb,
                   c,
                   kp: &[Vec2],
                   d: &[BinaryDescriptor],
                   rng: &mut _| {
            reloc.verify_detailed(db, &c, kp, d, &scene.intrinsics, rng)
        };

        assert_eq!(
            run(
                &reloc,
                &db,
                at(KeyframeId(9999)),
                &obs.keypoints,
                &obs.descriptors,
                &mut rng
            ),
            Err(Rejection::UnknownKeyframe)
        );

        // A keyframe none of whose features are triangulated cannot constrain
        // a pose, whatever the appearance says.
        let barren = db.insert_keyframe(crate::keyframe::Keyframe::new(
            KeyframeId::UNSET,
            wslam_core::Timestamp::ZERO,
            Se3::identity(),
            obs.keypoints.clone(),
            obs.descriptors.clone(),
            scene.intrinsics,
            &scene.vocabulary,
        ));
        assert_eq!(
            run(
                &reloc,
                &db,
                at(barren),
                &obs.keypoints,
                &obs.descriptors,
                &mut rng
            ),
            Err(Rejection::NoMappedLandmarks { landmarks: 0 })
        );

        // No query features at all: rejected by the matcher, and the reason
        // says so rather than implying the geometry looked at it. The bar at
        // this stage is the RANSAC seed minimum, not `min_inliers` — the full
        // quota is only demanded of reprojection inliers after the guided
        // second pass, because demanding it of ratio-tested seed matches is
        // what starved verification on real sequences (EuRoC MH_03:
        // rejections clustered at 19 matches of 20 required, reloc never
        // fired).
        let empty = run(&reloc, &db, at(KeyframeId(3)), &[], &[], &mut rng);
        assert_eq!(
            empty,
            Err(Rejection::TooFewMatches {
                matches: 0,
                required: MIN_SEED_MATCHES
            })
        );
        assert!(!empty.unwrap_err().reached_geometry());

        // An unreachable `min_inliers` now rejects at the *geometry* stage:
        // the seed matches exist and PnP runs, and the final inlier count is
        // what falls short. The count reported is the post-guided-pass one, so
        // it must be at least the seed matches that RANSAC agreed on.
        let strict = Relocalizer::new(RelocConfig {
            min_inliers: 100_000,
            ..RelocConfig::default()
        });
        match run(
            &strict,
            &db,
            at(KeyframeId(3)),
            &obs.keypoints,
            &obs.descriptors,
            &mut rng,
        ) {
            Err(Rejection::TooFewInliers {
                inliers, required, ..
            }) => {
                assert_eq!(required, 100_000);
                assert!(inliers > 20, "{inliers}");
            }
            other => panic!("expected TooFewInliers, got {other:?}"),
        }

        // The same descriptors against shuffled pixels: the matcher is happy,
        // and only the geometry can say no.
        let kf = db.keyframe(KeyframeId(3)).unwrap();
        let descriptors = kf.descriptors.clone();
        let mut keypoints = kf.keypoints.clone();
        rng.shuffle(&mut keypoints);
        let scrambled = run(
            &reloc,
            &db,
            at(KeyframeId(3)),
            &keypoints,
            &descriptors,
            &mut rng,
        );
        let reason = scrambled.expect_err("scrambled geometry must be rejected");
        assert!(
            reason.reached_geometry(),
            "the impostor was rejected at {reason}, before the geometry ran"
        );

        // ... and the honest query still passes, so none of the above is an
        // artifact of the setup.
        assert!(run(
            &reloc,
            &db,
            at(KeyframeId(3)),
            &obs.keypoints,
            &obs.descriptors,
            &mut rng
        )
        .is_ok());

        // `verify` is `verify_detailed` with the reason discarded.
        assert_eq!(
            reloc.verify(
                &db,
                &at(KeyframeId(9999)),
                &obs.keypoints,
                &obs.descriptors,
                &scene.intrinsics,
                &mut DeterministicRng::new("t", 1)
            ),
            None
        );
    }

    #[test]
    fn verify_refuses_when_the_inlier_bar_is_raised_above_the_evidence() {
        let (db, scene) = synth::corridor_map(8, 88);
        let mut rng = DeterministicRng::new("bar", 1);
        let obs = synth::observe(&scene, scene.poses[3], &mut rng);
        let candidate = Candidate {
            keyframe: KeyframeId(3),
            score: 1.0,
        };
        let lax = Relocalizer::new(RelocConfig::default());
        assert!(lax
            .verify(
                &db,
                &candidate,
                &obs.keypoints,
                &obs.descriptors,
                &scene.intrinsics,
                &mut rng
            )
            .is_some());
        // Raising the threshold is the documented response to a false positive
        // (spec.md §5), so it has to actually bite.
        let strict = Relocalizer::new(RelocConfig {
            min_inliers: 100_000,
            ..RelocConfig::default()
        });
        assert!(strict
            .verify(
                &db,
                &candidate,
                &obs.keypoints,
                &obs.descriptors,
                &scene.intrinsics,
                &mut rng
            )
            .is_none());
    }

    #[test]
    fn verify_handles_unknown_and_landmarkless_candidates() {
        let (db, scene) = synth::corridor_map(6, 2);
        let reloc = Relocalizer::new(RelocConfig::default());
        let mut rng = DeterministicRng::new("t", 1);
        let obs = synth::observe_noiseless(&scene, scene.poses[0]);
        // A candidate naming a keyframe that is not in the map.
        assert!(reloc
            .verify(
                &db,
                &Candidate {
                    keyframe: KeyframeId(9999),
                    score: 1.0
                },
                &obs.keypoints,
                &obs.descriptors,
                &scene.intrinsics,
                &mut rng
            )
            .is_none());
        // No query features at all.
        assert!(reloc
            .verify(
                &db,
                &Candidate {
                    keyframe: KeyframeId(0),
                    score: 1.0
                },
                &[],
                &[],
                &scene.intrinsics,
                &mut rng
            )
            .is_none());
    }

    #[test]
    fn relocalization_is_reproducible_for_a_fixed_seed() {
        let (db, scene) = synth::corridor_map(10, 6);
        let reloc = Relocalizer::new(RelocConfig::default());
        let obs = synth::observe(&scene, scene.poses[5], &mut DeterministicRng::new("o", 1));
        let bow = db.transform(&obs.descriptors);
        let run = || {
            let mut rng = DeterministicRng::new("reloc", 20260801);
            reloc
                .relocalize(
                    &db,
                    &bow,
                    &obs.keypoints,
                    &obs.descriptors,
                    &scene.intrinsics,
                    0,
                    &mut rng,
                )
                .map(|v| (v.keyframe, v.inliers, v.pose.log()))
        };
        assert_eq!(run(), run());
    }
}
