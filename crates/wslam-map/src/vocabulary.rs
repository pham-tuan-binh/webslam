//! Bag-of-binary-words vocabulary tree — DBoW2, reimplemented.
//!
//! spec.md §7: *"DBoW2 is small. The vocabulary is the artifact; the code is a
//! tree search over binary descriptors. Reimplementable in days, and the
//! trained vocabulary file is reusable as data."* This module is that tree
//! search, following Gálvez-López & Tardós (IEEE T-RO 28(5), 2012).
//!
//! The clustering is k-**medians** in Hamming space, not k-means: the
//! arithmetic mean of binary vectors is not a binary vector, whereas the
//! per-bit majority ([`BinaryDescriptor::majority`]) is exactly the point that
//! minimises summed Hamming distance to a cluster.
//!
//! ## Scoring
//!
//! [`BowVector::score`] is DBoW2's L1 score. For L1-normalised, non-negative
//! vectors that quantity is algebraically identical to histogram intersection
//! `sum_i min(a_i, b_i)` — see `score_matches_the_dbow2_l1_formula` — which is
//! 1 for identical vectors, 0 for disjoint ones, and needs no clamping to land
//! in `[0, 1]` as spec.md §4 L4 requires.

use crate::descriptor::BinaryDescriptor;
#[cfg(test)]
use crate::descriptor::DESCRIPTOR_BYTES;
use wslam_core::{DeterministicRng, Error, Result};

/// Maximum Lloyd iterations per tree node. DBoW2 uses the same order; the
/// assignment almost always stops moving after three or four.
const MAX_LLOYD_ITERATIONS: usize = 12;

/// Hamming radius at or below which a set of descriptors is one quantisation
/// cell and must not be subdivided.
///
/// The distance between two *independent* 256-bit codes is `Binomial(256, 1/2)`
/// — 128 bits with a standard deviation of exactly 8 — while two views of the
/// same physical point differ by a handful of bits. A node whose every member
/// sits within 8 bits of its centroid therefore holds one point observed
/// repeatedly, not several resolvable ones.
///
/// Splitting such a node is not merely wasteful, it is harmful. Place
/// recognition works only if two views of the same landmark reach the *same*
/// leaf; a tree that subdivides to a fixed depth regardless of the data
/// scatters them over `branching^k` words and the bag-of-words score for a
/// genuine revisit collapses. Gálvez-López & Tardós never hit this because a
/// real vocabulary is trained on millions of descriptors, so no node is ever
/// this tight — but a small or repetitive training set makes it the normal
/// case, and `training_recovers_synthetic_clusters` is exactly that.
///
/// The bound is applied in two places, because both can over-split: the node
/// itself ([`Vocabulary::grow`]) and the k-medians seeding, which otherwise
/// happily places `branching` seeds inside a cloud that contains two groups.
const MIN_SPLIT_RADIUS: u32 = 8;

/// Magic bytes at the head of a serialised vocabulary.
pub const VOCAB_MAGIC: &[u8; 8] = b"WSLAMVOC";

/// One node of the vocabulary tree.
#[derive(Debug, Clone, PartialEq)]
struct VocabNode {
    /// Cluster centre. Meaningless for the root.
    descriptor: BinaryDescriptor,
    /// Indices into `Vocabulary::nodes`.
    children: Vec<u32>,
    /// Word id, set only on leaves.
    word: Option<u32>,
}

/// A leaf's payload.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Word {
    node: u32,
    /// Inverse document frequency. Zero for a word that every training
    /// descriptor visits — such a word discriminates nothing.
    weight: f64,
}

/// A hierarchical vocabulary tree over binary descriptors.
///
/// Train once offline, ship the serialised bytes as data (spec.md §7 keeps a
/// `vocab/` directory in git-lfs for exactly this), and load it at session
/// start. The tree is immutable after training.
#[derive(Debug, Clone, PartialEq)]
pub struct Vocabulary {
    branching: usize,
    depth: usize,
    nodes: Vec<VocabNode>,
    words: Vec<Word>,
    training_descriptors: usize,
}

impl Default for Vocabulary {
    fn default() -> Self {
        Vocabulary::empty()
    }
}

impl Vocabulary {
    /// A vocabulary with no words. [`Vocabulary::transform`] returns an empty
    /// [`BowVector`] and every score is zero — the honest behaviour before a
    /// vocabulary has been loaded, rather than a panic or a fake match.
    #[must_use]
    pub fn empty() -> Self {
        Vocabulary {
            branching: 0,
            depth: 0,
            nodes: vec![VocabNode {
                descriptor: BinaryDescriptor::ZERO,
                children: Vec::new(),
                word: None,
            }],
            words: Vec::new(),
            training_descriptors: 0,
        }
    }

    /// Train a vocabulary by hierarchical k-medians.
    ///
    /// `branching` children per node, `depth` levels below the root, so at most
    /// `branching^depth` words. `rng` seeds the k-means++ initialisation and is
    /// the only randomness involved; the same seed and the same descriptors
    /// give the same tree (spec.md §6, "Every RNG is seeded").
    ///
    /// Degenerate inputs return an empty vocabulary rather than failing:
    /// training on nothing is a legitimate thing for a caller to do while
    /// bootstrapping.
    #[must_use]
    pub fn train(
        descriptors: &[BinaryDescriptor],
        branching: usize,
        depth: usize,
        rng: &mut DeterministicRng,
    ) -> Self {
        if descriptors.is_empty() || branching < 2 || depth == 0 {
            return Vocabulary::empty();
        }

        let mut vocab = Vocabulary {
            branching,
            depth,
            nodes: vec![VocabNode {
                descriptor: BinaryDescriptor::ZERO,
                children: Vec::new(),
                word: None,
            }],
            words: Vec::new(),
            training_descriptors: descriptors.len(),
        };

        let all: Vec<u32> = (0..descriptors.len() as u32).collect();
        vocab.grow(0, descriptors, &all, 0, rng);
        vocab.compute_weights(descriptors);
        vocab
    }

    /// Recursively split `members` beneath `node`.
    fn grow(
        &mut self,
        node: u32,
        descriptors: &[BinaryDescriptor],
        members: &[u32],
        level: usize,
        rng: &mut DeterministicRng,
    ) {
        if members.is_empty() {
            return;
        }
        // A node that cannot be split further becomes a word. The radius test
        // is the third case and it is not an optimisation: see
        // [`MIN_SPLIT_RADIUS`]. It must come *before* the
        // `members.len() <= branching` shortcut below, which would otherwise
        // give every member of a tight cluster its own word.
        if level >= self.depth
            || members.len() == 1
            || cluster_radius(descriptors, members, &centroid(descriptors, members))
                <= MIN_SPLIT_RADIUS
        {
            let word_id = self.words.len() as u32;
            self.words.push(Word { node, weight: 0.0 });
            self.nodes[node as usize].word = Some(word_id);
            return;
        }

        let clusters = if members.len() <= self.branching {
            // Fewer descriptors than branches: each becomes its own centre.
            members.iter().map(|&i| vec![i]).collect()
        } else {
            k_medians(descriptors, members, self.branching, rng)
        };

        for cluster in clusters {
            if cluster.is_empty() {
                continue;
            }
            let centre = centroid(descriptors, &cluster);
            let child = self.nodes.len() as u32;
            self.nodes.push(VocabNode {
                descriptor: centre,
                children: Vec::new(),
                word: None,
            });
            self.nodes[node as usize].children.push(child);
            self.grow(child, descriptors, &cluster, level + 1, rng);
        }
    }

    /// tf-idf weights from the training set.
    ///
    /// `idf(w) = ln(N / n_w)` with `N` the number of training descriptors and
    /// `n_w` how many of them land in word `w`. DBoW2 counts *images*; the
    /// frozen `train` signature takes a flat descriptor list, so descriptor
    /// counts are the available proxy. It has the property that matters: a word
    /// every descriptor visits gets weight zero.
    fn compute_weights(&mut self, descriptors: &[BinaryDescriptor]) {
        if self.words.is_empty() {
            return;
        }
        let mut counts = vec![0usize; self.words.len()];
        for d in descriptors {
            if let Some(w) = self.word_of(d) {
                counts[w as usize] += 1;
            }
        }
        let n = descriptors.len() as f64;
        for (word, &count) in self.words.iter_mut().zip(counts.iter()) {
            word.weight = if count == 0 {
                0.0
            } else {
                (n / count as f64).ln().max(0.0)
            };
        }
    }

    /// Quantise one descriptor to a word id by greedy descent.
    #[must_use]
    pub fn word_of(&self, descriptor: &BinaryDescriptor) -> Option<u32> {
        if self.words.is_empty() {
            return None;
        }
        let mut node = 0u32;
        loop {
            let n = &self.nodes[node as usize];
            if let Some(w) = n.word {
                return Some(w);
            }
            if n.children.is_empty() {
                return None;
            }
            let mut best = n.children[0];
            let mut best_d = descriptor.hamming(&self.nodes[best as usize].descriptor);
            for &c in &n.children[1..] {
                let d = descriptor.hamming(&self.nodes[c as usize].descriptor);
                // Strict `<` so ties resolve to the first child, deterministically.
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            node = best;
        }
    }

    /// Convert a set of descriptors into an L1-normalised tf-idf bag of words.
    #[must_use]
    pub fn transform(&self, descriptors: &[BinaryDescriptor]) -> BowVector {
        if self.words.is_empty() || descriptors.is_empty() {
            return BowVector::empty();
        }
        let mut counts: std::collections::BTreeMap<u32, f64> = std::collections::BTreeMap::new();
        for d in descriptors {
            if let Some(w) = self.word_of(d) {
                *counts.entry(w).or_insert(0.0) += 1.0;
            }
        }
        let total = descriptors.len() as f64;
        let entries: Vec<(u32, f64)> = counts
            .into_iter()
            .filter_map(|(w, c)| {
                let v = (c / total) * self.words[w as usize].weight;
                (v > 0.0).then_some((w, v))
            })
            .collect();
        BowVector::from_sorted_unnormalized(entries)
    }

    /// Number of leaves.
    #[must_use]
    pub fn word_count(&self) -> usize {
        self.words.len()
    }

    /// Whether the vocabulary has no words at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    /// Children per node, as trained.
    #[must_use]
    pub fn branching(&self) -> usize {
        self.branching
    }

    /// Tree depth, as trained.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Number of tree nodes including the root.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Inverse document frequency of a word; 0 for an unknown id.
    #[must_use]
    pub fn word_weight(&self, word: u32) -> f64 {
        self.words.get(word as usize).map_or(0.0, |w| w.weight)
    }

    /// Approximate heap footprint, for [`crate::MapDb::memory_bytes`].
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        let node = std::mem::size_of::<VocabNode>();
        std::mem::size_of::<Self>()
            + self
                .nodes
                .iter()
                .map(|n| node + n.children.len() * std::mem::size_of::<u32>())
                .sum::<usize>()
            + self.words.len() * std::mem::size_of::<Word>()
    }

    /// Serialise to the versioned little-endian format.
    ///
    /// Explicit byte layout, no serde: the vocabulary is a long-lived artifact
    /// shipped as data, and it must be readable by a build that shares nothing
    /// with this one but the format document.
    #[must_use]
    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(64 + self.nodes.len() * 48);
        out.extend_from_slice(VOCAB_MAGIC);
        out.extend_from_slice(&wslam_core::MAP_FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // reserved
        out.extend_from_slice(&(self.branching as u32).to_le_bytes());
        out.extend_from_slice(&(self.depth as u32).to_le_bytes());
        out.extend_from_slice(&(self.nodes.len() as u32).to_le_bytes());
        out.extend_from_slice(&(self.words.len() as u32).to_le_bytes());
        out.extend_from_slice(&(self.training_descriptors as u64).to_le_bytes());
        for n in &self.nodes {
            out.extend_from_slice(&n.descriptor.0);
            out.extend_from_slice(&n.word.map_or(-1i32, |w| w as i32).to_le_bytes());
            out.extend_from_slice(&(n.children.len() as u32).to_le_bytes());
            for &c in &n.children {
                out.extend_from_slice(&c.to_le_bytes());
            }
        }
        for w in &self.words {
            out.extend_from_slice(&w.node.to_le_bytes());
            out.extend_from_slice(&w.weight.to_le_bytes());
        }
        out
    }

    /// Read a vocabulary produced by [`Vocabulary::serialize`].
    ///
    /// Rejects an unknown version with [`Error::MapVersion`], and any structural
    /// inconsistency — truncation, out-of-range child index, dangling word — with
    /// [`Error::MapFormat`]. Never panics on hostile input.
    pub fn deserialize(bytes: &[u8]) -> Result<Self> {
        let mut r = crate::serialize::Reader::new(bytes);
        let magic = r.bytes(8)?;
        if magic != VOCAB_MAGIC {
            return Err(Error::MapFormat("vocabulary magic mismatch".into()));
        }
        let version = r.u16()?;
        if version != wslam_core::MAP_FORMAT_VERSION {
            return Err(Error::MapVersion {
                found: version,
                supported: wslam_core::MAP_FORMAT_VERSION,
            });
        }
        let _reserved = r.u16()?;
        let branching = r.u32()? as usize;
        let depth = r.u32()? as usize;
        let node_count = r.u32()? as usize;
        let word_count = r.u32()? as usize;
        let training_descriptors = r.u64()? as usize;

        // A node is at least 40 bytes and a word 12, so a header claiming more
        // than the buffer can hold is rejected before any allocation.
        r.check_remaining(node_count.saturating_mul(40) + word_count.saturating_mul(12))?;

        let mut nodes = Vec::with_capacity(node_count);
        for _ in 0..node_count {
            let mut d = [0u8; 32];
            d.copy_from_slice(r.bytes(32)?);
            let word_tag = r.i32()?;
            let child_count = r.u32()? as usize;
            r.check_remaining(child_count.saturating_mul(4))?;
            let mut children = Vec::with_capacity(child_count);
            for _ in 0..child_count {
                children.push(r.u32()?);
            }
            let word = if word_tag < 0 {
                None
            } else {
                Some(word_tag as u32)
            };
            nodes.push(VocabNode {
                descriptor: BinaryDescriptor(d),
                children,
                word,
            });
        }
        let mut words = Vec::with_capacity(word_count);
        for _ in 0..word_count {
            words.push(Word {
                node: r.u32()?,
                weight: r.f64()?,
            });
        }

        if nodes.is_empty() {
            return Err(Error::MapFormat("vocabulary has no root node".into()));
        }
        for n in &nodes {
            if n.children.iter().any(|&c| c as usize >= nodes.len()) {
                return Err(Error::MapFormat(
                    "vocabulary child index out of range".into(),
                ));
            }
            if n.word.is_some_and(|w| w as usize >= words.len()) {
                return Err(Error::MapFormat("vocabulary word id out of range".into()));
            }
        }
        if words.iter().any(|w| w.node as usize >= nodes.len()) {
            return Err(Error::MapFormat("vocabulary word node out of range".into()));
        }
        if words
            .iter()
            .any(|w| !w.weight.is_finite() || w.weight < 0.0)
        {
            return Err(Error::MapFormat(
                "vocabulary weight is not a finite idf".into(),
            ));
        }

        Ok(Vocabulary {
            branching,
            depth,
            nodes,
            words,
            training_descriptors,
        })
    }
}

/// The per-bit majority of a cluster.
fn centroid(descriptors: &[BinaryDescriptor], members: &[u32]) -> BinaryDescriptor {
    let picked: Vec<BinaryDescriptor> = members.iter().map(|&i| descriptors[i as usize]).collect();
    BinaryDescriptor::majority(&picked)
}

/// Largest Hamming distance from any member to `centre`.
///
/// The *maximum* rather than the mean, deliberately: a node holding four
/// hundred copies of one descriptor and four copies of another has a mean
/// distance near zero yet is obviously two cells, and a mean-based rule would
/// merge the rare word away — the opposite of what idf weighting is for.
fn cluster_radius(
    descriptors: &[BinaryDescriptor],
    members: &[u32],
    centre: &BinaryDescriptor,
) -> u32 {
    members
        .iter()
        .map(|&i| descriptors[i as usize].hamming(centre))
        .max()
        .unwrap_or(0)
}

/// Cap on the number of distinct near-duplicate cells before aggregation is
/// abandoned for the node.
///
/// Aggregation only *matters* when a node is redundant, and a node with this
/// many distinct cells plainly is not. Bailing out to one cell per descriptor
/// past the cap therefore costs nothing in quality and keeps the pass
/// `O(MAX_CELLS^2 + n)` instead of `O(n^2)` — the vocabulary is an offline
/// artifact (spec.md §7) but it is still trained on millions of descriptors.
const MAX_CELLS: usize = 4096;

/// Group `members` into atomic cells of mutually near-duplicate descriptors.
///
/// This is what makes "a split never cuts a group of repeated observations in
/// half" structural rather than hopeful, and it is needed because no
/// distance-to-centroid rule can achieve it:
///
/// With `k` seeds and more than `k` well-separated groups, the *unseeded*
/// groups are equidistant from every centre, so Lloyd assigns their members by
/// whatever noise happens to break the tie — scattering one group across all
/// `k` clusters. Lloyd cannot recover, either, because the Hamming median is a
/// per-bit majority: the minority bits contributed by a stray sub-group are
/// thresholded away entirely, so the centre never drifts toward them the way a
/// Euclidean mean would. The symmetry is exact and permanent.
///
/// Clustering cells rather than descriptors sidesteps it: the tie is still
/// broken arbitrarily, but now it moves a whole group at a time.
fn near_duplicate_cells(descriptors: &[BinaryDescriptor], members: &[u32]) -> Vec<Vec<u32>> {
    let mut cells: Vec<Vec<u32>> = Vec::new();
    let mut reps: Vec<BinaryDescriptor> = Vec::new();
    for &m in members {
        let d = descriptors[m as usize];
        match reps.iter().position(|r| d.hamming(r) <= MIN_SPLIT_RADIUS) {
            Some(i) => cells[i].push(m),
            None if reps.len() < MAX_CELLS => {
                reps.push(d);
                cells.push(vec![m]);
            }
            None => return members.iter().map(|&x| vec![x]).collect(),
        }
    }
    cells
}

/// k-medians in Hamming space with k-means++ seeding, over near-duplicate
/// cells rather than over individual descriptors.
///
/// Returns the non-empty clusters as index lists. Deterministic given `rng`.
fn k_medians(
    descriptors: &[BinaryDescriptor],
    members: &[u32],
    k: usize,
    rng: &mut DeterministicRng,
) -> Vec<Vec<u32>> {
    let units = near_duplicate_cells(descriptors, members);
    // Fewer resolvable cells than branches: the cells *are* the clusters, and
    // inventing more would only fragment them.
    if units.len() <= k {
        return units;
    }
    let reps: Vec<BinaryDescriptor> = units.iter().map(|u| centroid(descriptors, u)).collect();

    let mut centres: Vec<BinaryDescriptor> = Vec::with_capacity(k);
    centres.push(reps[rng.below(reps.len())]);

    // k-means++ seeding, on squared Hamming distance. Spreading the seeds is
    // what stops a whole subtree collapsing into one word.
    let mut d2 = vec![0f64; reps.len()];
    while centres.len() < k {
        let mut total = 0.0;
        let mut farthest = 0u32;
        for ((slot, r), unit) in d2.iter_mut().zip(reps.iter()).zip(units.iter()) {
            let d = centres.iter().map(|c| r.hamming(c)).min().unwrap_or(0);
            farthest = farthest.max(d);
            // Weighted by cell population: a cell standing for four hundred
            // observations deserves four hundred times the seeding pressure of
            // a singleton, which is what plain k-means++ over descriptors gave
            // for free and cell aggregation would otherwise throw away.
            *slot = (d as f64) * (d as f64) * unit.len() as f64;
            total += *slot;
        }
        if total <= 0.0 {
            break; // every remaining cell is already a centre
        }
        // Stop seeding once the whole cloud is inside the noise radius of some
        // centre: `k` is a branching factor, not a claim that the data contains
        // `k` groups. Without this, k-means++ dutifully drops four seeds into a
        // node holding two groups. See [`MIN_SPLIT_RADIUS`].
        if farthest <= MIN_SPLIT_RADIUS {
            break;
        }
        let mut t = rng.uniform() * total;
        let mut chosen = reps.len() - 1;
        for (i, &w) in d2.iter().enumerate() {
            t -= w;
            if t <= 0.0 {
                chosen = i;
                break;
            }
        }
        centres.push(reps[chosen]);
    }

    let mut assignment = vec![0usize; units.len()];
    for iteration in 0..MAX_LLOYD_ITERATIONS {
        let mut changed = false;
        for (slot, r) in assignment.iter_mut().zip(reps.iter()) {
            let mut best = 0usize;
            let mut best_d = u32::MAX;
            for (ci, c) in centres.iter().enumerate() {
                let dist = r.hamming(c);
                if dist < best_d {
                    best_d = dist;
                    best = ci;
                }
            }
            if *slot != best || iteration == 0 {
                *slot = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }
        for (ci, centre) in centres.iter_mut().enumerate() {
            // The centre is the majority over every *member* of the assigned
            // cells, not over the cell representatives: that is the actual
            // k-medians centre of the cluster.
            let picked: Vec<BinaryDescriptor> = units
                .iter()
                .zip(assignment.iter())
                .filter(|(_, &a)| a == ci)
                .flat_map(|(u, _)| u.iter().map(|&m| descriptors[m as usize]))
                .collect();
            // An emptied cluster keeps its old centre; it will simply be
            // dropped below.
            if !picked.is_empty() {
                *centre = BinaryDescriptor::majority(&picked);
            }
        }
    }

    let mut clusters = vec![Vec::new(); centres.len()];
    for (unit, &a) in units.iter().zip(assignment.iter()) {
        clusters[a].extend_from_slice(unit);
    }
    clusters.retain(|c| !c.is_empty());
    clusters
}

/// A sparse tf-idf bag-of-words vector, sorted by word id and L1-normalised.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BowVector {
    entries: Vec<(u32, f64)>,
}

impl BowVector {
    /// The empty vector. Scores zero against everything, including itself —
    /// "no information" rather than "perfect match".
    #[must_use]
    pub fn empty() -> Self {
        BowVector {
            entries: Vec::new(),
        }
    }

    /// Build from `(word, weight)` pairs sorted by word, normalising to unit L1
    /// norm. Non-positive and non-finite weights are dropped.
    #[must_use]
    fn from_sorted_unnormalized(mut entries: Vec<(u32, f64)>) -> Self {
        entries.retain(|&(_, v)| v.is_finite() && v > 0.0);
        let norm: f64 = entries.iter().map(|&(_, v)| v).sum();
        if norm <= 0.0 {
            return BowVector::empty();
        }
        for e in entries.iter_mut() {
            e.1 /= norm;
        }
        BowVector { entries }
    }

    /// Build from arbitrary `(word, weight)` pairs. Duplicates are summed, the
    /// result is sorted and L1-normalised. Exposed for tests and for callers
    /// reconstructing a vector without a vocabulary to hand.
    #[must_use]
    pub fn from_pairs(pairs: &[(u32, f64)]) -> Self {
        let mut map: std::collections::BTreeMap<u32, f64> = std::collections::BTreeMap::new();
        for &(w, v) in pairs {
            *map.entry(w).or_insert(0.0) += v;
        }
        BowVector::from_sorted_unnormalized(map.into_iter().collect())
    }

    /// Number of distinct words present.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no word is present.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The `(word, weight)` pairs, ascending by word id.
    pub fn iter(&self) -> impl Iterator<Item = (u32, f64)> + '_ {
        self.entries.iter().copied()
    }

    /// Weight of a word, 0 if absent.
    #[must_use]
    pub fn weight(&self, word: u32) -> f64 {
        self.entries
            .binary_search_by_key(&word, |&(w, _)| w)
            .map_or(0.0, |i| self.entries[i].1)
    }

    /// The words present, ascending.
    pub fn words(&self) -> impl Iterator<Item = u32> + '_ {
        self.entries.iter().map(|&(w, _)| w)
    }

    /// DBoW2 L1 similarity, in `[0, 1]`.
    ///
    /// DBoW2 accumulates `|a_i - b_i| - |a_i| - |b_i|` over shared words and
    /// halves the negated sum. For non-negative L1-normalised vectors that is
    /// `sum_i min(a_i, b_i)`, evaluated here by a merge walk. 1 for identical
    /// vectors, 0 when the word sets are disjoint.
    #[must_use]
    pub fn score(&self, other: &Self) -> f64 {
        let (mut i, mut j) = (0usize, 0usize);
        let mut acc = 0.0;
        while i < self.entries.len() && j < other.entries.len() {
            let (wa, va) = self.entries[i];
            let (wb, vb) = other.entries[j];
            match wa.cmp(&wb) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    acc += va.min(vb);
                    i += 1;
                    j += 1;
                }
            }
        }
        acc.clamp(0.0, 1.0)
    }

    /// Approximate heap footprint.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.entries.len() * std::mem::size_of::<(u32, f64)>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// `n_clusters` well-separated base descriptors, each with `per_cluster`
    /// members differing from it by `jitter` random bit flips. Separation is
    /// exact: base `i` sets a disjoint block of bits, so inter-cluster Hamming
    /// distance is far larger than intra-cluster.
    fn clustered(
        n_clusters: usize,
        per_cluster: usize,
        jitter: usize,
        rng: &mut DeterministicRng,
    ) -> (Vec<BinaryDescriptor>, Vec<usize>) {
        let mut descs = Vec::new();
        let mut labels = Vec::new();
        for c in 0..n_clusters {
            let mut base = BinaryDescriptor::ZERO;
            // Disjoint 16-bit blocks -> pairwise distance 32 bits.
            for b in 0..16 {
                base.set_bit((c * 16 + b) % 256, true);
            }
            for _ in 0..per_cluster {
                let mut d = base;
                for _ in 0..jitter {
                    let bit = rng.below(256);
                    d.set_bit(bit, !d.bit(bit));
                }
                descs.push(d);
                labels.push(c);
            }
        }
        (descs, labels)
    }

    #[test]
    fn training_recovers_synthetic_clusters() {
        let mut rng = DeterministicRng::new("vocab-test", 20260801);
        let (descs, labels) = clustered(8, 25, 2, &mut rng);
        let vocab = Vocabulary::train(&descs, 4, 3, &mut rng);
        assert!(vocab.word_count() >= 8, "got {} words", vocab.word_count());

        // Every descriptor of a cluster must quantise to the same word, and
        // different clusters must land in different words.
        let mut word_of_cluster = [None; 8];
        for (d, &c) in descs.iter().zip(labels.iter()) {
            let w = vocab.word_of(d).expect("trained vocabulary quantises");
            match word_of_cluster[c] {
                None => word_of_cluster[c] = Some(w),
                Some(prev) => assert_eq!(prev, w, "cluster {c} split across words"),
            }
        }
        let mut seen: Vec<u32> = word_of_cluster.iter().map(|w| w.unwrap()).collect();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), 8, "clusters collapsed into shared words");
    }

    /// Dense, ORB-like bases: ~half the bits set, mutually ~128 apart.
    fn clustered_dense(
        n_clusters: usize,
        per_cluster: usize,
        jitter: usize,
        rng: &mut DeterministicRng,
    ) -> (Vec<BinaryDescriptor>, Vec<usize>) {
        let mut descs = Vec::new();
        let mut labels = Vec::new();
        for c in 0..n_clusters {
            let mut base = BinaryDescriptor::ZERO;
            for b in base.0.iter_mut() {
                *b = rng.below(256) as u8;
            }
            for _ in 0..per_cluster {
                let mut d = base;
                for _ in 0..jitter {
                    let bit = rng.below(256);
                    d.set_bit(bit, !d.bit(bit));
                }
                descs.push(d);
                labels.push(c);
            }
        }
        (descs, labels)
    }

    /// Which word each labelled group quantises to, or `Err` if a group was
    /// split across two words.
    fn words_per_group(
        vocab: &Vocabulary,
        descs: &[BinaryDescriptor],
        labels: &[usize],
        groups: usize,
    ) -> std::result::Result<Vec<u32>, usize> {
        let mut word_of_group = vec![None; groups];
        for (d, &g) in descs.iter().zip(labels.iter()) {
            let w = vocab.word_of(d).expect("a trained vocabulary quantises");
            match word_of_group[g] {
                None => word_of_group[g] = Some(w),
                Some(prev) if prev != w => return Err(g),
                Some(_) => {}
            }
        }
        Ok(word_of_group.into_iter().map(|w| w.unwrap()).collect())
    }

    #[test]
    fn a_group_is_never_split_whatever_the_tree_shape() {
        // The regression `training_recovers_synthetic_clusters` caught, swept.
        // With `branching` seeds and more groups than that, the unseeded groups
        // are equidistant from every centre; assigning descriptors one at a
        // time scatters each of them over all the clusters, and because the
        // Hamming median thresholds minority bits away, no later Lloyd
        // iteration can pull them back. Clustering near-duplicate *cells* makes
        // this structurally impossible, whatever the tree shape or the data.
        for seed in [1u64, 7, 4242, 20260801] {
            for &(groups, branching, depth) in &[
                (8usize, 4usize, 3usize),
                (12, 3, 3),
                (5, 4, 2),
                (9, 2, 5),
                (16, 4, 3),
            ] {
                for dense in [false, true] {
                    let mut rng = DeterministicRng::new("split", seed);
                    let (descs, labels) = if dense {
                        clustered_dense(groups, 17, 2, &mut rng)
                    } else {
                        clustered(groups, 17, 2, &mut rng)
                    };
                    let vocab = Vocabulary::train(&descs, branching, depth, &mut rng);
                    if let Err(g) = words_per_group(&vocab, &descs, &labels, groups) {
                        panic!(
                            "group {g} split across words: dense={dense} seed={seed} \
                             groups={groups} branching={branching} depth={depth}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn well_separated_clusters_each_get_their_own_word() {
        // Dense bases with about half their bits set, which is what a real ORB
        // descriptor looks like. Every group must end up alone in a word — the
        // property place recognition actually depends on.
        //
        // The sparse `clustered` fixture (16 bits of 256) cannot demand this at
        // every tree shape, and the reason is worth recording rather than
        // hiding. The Hamming median of a cluster holding several *sparse*
        // groups with disjoint bit blocks is all-zero, which sits 16 bits from
        // every group while the groups sit 32 bits from each other. That child
        // is therefore nearer to everything than any group's own centre is to
        // any other group, so it absorbs every unseeded group and the tree
        // degenerates into a chain yielding `(branching - 1) * depth + 1` words.
        // Quantisation stays *correct* — nothing is ever split, which is the
        // property that matters — but capacity is linear rather than
        // exponential in depth, so groups start sharing words once it runs out.
        // Dense codes have no such degenerate median and the tree stays wide,
        // as the sweep below shows.
        for seed in [1u64, 7, 4242, 20260801] {
            for &(groups, branching, depth) in &[
                (8usize, 4usize, 3usize),
                (12, 3, 3),
                (5, 4, 2),
                (9, 2, 5),
                (16, 4, 3),
            ] {
                let mut rng = DeterministicRng::new("dense", seed);
                let (descs, labels) = clustered_dense(groups, 17, 2, &mut rng);
                let vocab = Vocabulary::train(&descs, branching, depth, &mut rng);
                let mut seen = words_per_group(&vocab, &descs, &labels, groups)
                    .unwrap_or_else(|g| panic!("group {g} split (seed {seed})"));
                assert_eq!(
                    vocab.word_count(),
                    groups,
                    "seed {seed}, {groups} groups, branching {branching}, depth {depth}"
                );
                seen.sort_unstable();
                seen.dedup();
                assert_eq!(seen.len(), groups, "groups collapsed into shared words");
            }
        }
    }

    #[test]
    fn near_duplicate_cells_are_a_bounded_ball_around_their_first_member() {
        // The unit beneath the property above. Cells are leader-clustered: a
        // descriptor joins a cell when it is within MIN_SPLIT_RADIUS of that
        // cell's *first* member, so a cell is a ball of radius
        // MIN_SPLIT_RADIUS and its diameter is bounded at twice that. It is
        // deliberately not single-linkage, which would let cells chain
        // arbitrarily far across the space one bit at a time.
        let a = BinaryDescriptor::ZERO;
        let mut b = BinaryDescriptor::ZERO;
        for i in 0..MIN_SPLIT_RADIUS as usize {
            b.set_bit(i, true);
        }
        // Exactly at the radius: same cell.
        assert_eq!(a.hamming(&b), MIN_SPLIT_RADIUS);
        assert_eq!(near_duplicate_cells(&[a, b], &[0, 1]), vec![vec![0, 1]]);
        // One bit further: a new cell, even though it is 1 bit from `b`, which
        // is already in the first cell. Leader, not chain.
        let mut c = b;
        c.set_bit(200, true);
        assert_eq!(a.hamming(&c), MIN_SPLIT_RADIUS + 1);
        assert_eq!(b.hamming(&c), 1);
        assert_eq!(
            near_duplicate_cells(&[a, b, c], &[0, 1, 2]),
            vec![vec![0, 1], vec![2]]
        );
        // A maximally distant descriptor is always its own cell.
        let far = BinaryDescriptor([0xFF; DESCRIPTOR_BYTES]);
        assert_eq!(
            near_duplicate_cells(&[a, far], &[0, 1]),
            vec![vec![0], vec![1]]
        );
        // Every member is placed exactly once, whatever the input.
        let mut rng = DeterministicRng::new("cells", 3);
        let (descs, _) = clustered(6, 11, 2, &mut rng);
        let all: Vec<u32> = (0..descs.len() as u32).collect();
        let cells = near_duplicate_cells(&descs, &all);
        assert_eq!(cells.len(), 6, "one cell per well-separated group");
        let mut flat: Vec<u32> = cells.into_iter().flatten().collect();
        flat.sort_unstable();
        assert_eq!(flat, all);
    }

    #[test]
    fn training_is_deterministic_for_a_given_seed() {
        let mut r0 = DeterministicRng::new("t", 7);
        let (descs, _) = clustered(6, 20, 3, &mut r0);
        let a = Vocabulary::train(&descs, 3, 3, &mut DeterministicRng::new("t", 99));
        let b = Vocabulary::train(&descs, 3, 3, &mut DeterministicRng::new("t", 99));
        assert_eq!(a, b);
        let c = Vocabulary::train(&descs, 3, 3, &mut DeterministicRng::new("t", 100));
        // A different seed may well give a different tree; what must not happen
        // is the same seed giving different trees.
        assert_eq!(a.word_count(), a.word_count());
        let _ = c;
    }

    #[test]
    fn identical_descriptor_sets_score_one() {
        let mut rng = DeterministicRng::new("t", 5);
        let (descs, _) = clustered(8, 20, 2, &mut rng);
        let vocab = Vocabulary::train(&descs, 4, 3, &mut rng);
        let a = vocab.transform(&descs[0..80]);
        let b = vocab.transform(&descs[0..80]);
        assert!(!a.is_empty());
        assert_relative_eq!(a.score(&b), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn disjoint_descriptor_sets_score_zero() {
        let mut rng = DeterministicRng::new("t", 6);
        let (descs, _) = clustered(8, 20, 1, &mut rng);
        let vocab = Vocabulary::train(&descs, 4, 3, &mut rng);
        // Clusters 0-1 versus clusters 6-7: no shared word.
        let a = vocab.transform(&descs[0..40]);
        let b = vocab.transform(&descs[120..160]);
        assert!(!a.is_empty() && !b.is_empty());
        assert_relative_eq!(a.score(&b), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn score_decays_with_overlap() {
        let mut rng = DeterministicRng::new("t", 8);
        let (descs, _) = clustered(8, 20, 1, &mut rng);
        let vocab = Vocabulary::train(&descs, 4, 3, &mut rng);
        let reference = vocab.transform(&descs[0..80]); // clusters 0-3
        let heavy = vocab.transform(&descs[0..60]); // clusters 0-2, 3/4 shared
        let light = vocab.transform(&descs[60..140]); // clusters 3-6, 1/4 shared
        let none = vocab.transform(&descs[80..160]); // clusters 4-7
        assert!(reference.score(&heavy) > reference.score(&light));
        assert!(reference.score(&light) > reference.score(&none));
        assert_relative_eq!(reference.score(&none), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn score_matches_the_dbow2_l1_formula() {
        // DBoW2: score = -0.5 * sum_common(|a-b| - |a| - |b|). Check ours
        // against that directly rather than against itself.
        let a = BowVector::from_pairs(&[(1, 3.0), (2, 1.0), (5, 6.0)]);
        let b = BowVector::from_pairs(&[(2, 4.0), (5, 4.0), (9, 2.0)]);
        let mut dbow = 0.0;
        for (w, va) in a.iter() {
            let vb = b.weight(w);
            if vb > 0.0 {
                dbow += (va - vb).abs() - va.abs() - vb.abs();
            }
        }
        assert_relative_eq!(a.score(&b), -0.5 * dbow, epsilon = 1e-15);
        assert!((0.0..=1.0).contains(&a.score(&b)));
        // Symmetric.
        assert_relative_eq!(a.score(&b), b.score(&a), epsilon = 1e-15);
    }

    #[test]
    fn idf_zeroes_a_word_every_descriptor_visits() {
        // One cluster only: the whole training set quantises to a single word,
        // so ln(N/N) = 0 and the vector carries no information at all.
        let mut rng = DeterministicRng::new("t", 11);
        let (descs, _) = clustered(1, 40, 0, &mut rng);
        let vocab = Vocabulary::train(&descs, 4, 2, &mut rng);
        assert_eq!(vocab.word_count(), 1);
        assert_relative_eq!(vocab.word_weight(0), 0.0, epsilon = 1e-12);
        // ... and therefore transforming gives an empty vector, which scores 0.
        let v = vocab.transform(&descs);
        assert!(v.is_empty());
        assert_relative_eq!(v.score(&v), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn rare_words_outweigh_common_ones() {
        let mut rng = DeterministicRng::new("t", 12);
        let (mut descs, _) = clustered(2, 4, 0, &mut rng);
        // Flood cluster 0 so it is 100x more common than cluster 1.
        let common = descs[0];
        for _ in 0..400 {
            descs.push(common);
        }
        let vocab = Vocabulary::train(&descs, 2, 2, &mut rng);
        let w_common = vocab.word_of(&common).unwrap();
        let w_rare = vocab.word_of(&descs[4]).unwrap();
        assert_ne!(w_common, w_rare);
        assert!(
            vocab.word_weight(w_rare) > vocab.word_weight(w_common),
            "idf must favour the rare word: rare {} vs common {}",
            vocab.word_weight(w_rare),
            vocab.word_weight(w_common)
        );
    }

    #[test]
    fn empty_vocabulary_is_inert_rather_than_fatal() {
        let v = Vocabulary::empty();
        assert_eq!(v.word_count(), 0);
        assert!(v.is_empty());
        let d = [BinaryDescriptor::ZERO; 4];
        assert!(v.word_of(&d[0]).is_none());
        assert!(v.transform(&d).is_empty());
        // Degenerate training parameters do not produce a broken tree.
        let mut rng = DeterministicRng::new("t", 1);
        assert!(Vocabulary::train(&[], 4, 3, &mut rng).is_empty());
        assert!(Vocabulary::train(&d, 1, 3, &mut rng).is_empty());
        assert!(Vocabulary::train(&d, 4, 0, &mut rng).is_empty());
    }

    #[test]
    fn transform_of_an_empty_descriptor_set_is_empty() {
        let mut rng = DeterministicRng::new("t", 2);
        let (descs, _) = clustered(4, 10, 1, &mut rng);
        let vocab = Vocabulary::train(&descs, 2, 3, &mut rng);
        assert!(vocab.transform(&[]).is_empty());
    }

    #[test]
    fn vocabulary_serialization_roundtrips() {
        let mut rng = DeterministicRng::new("t", 13);
        let (descs, _) = clustered(8, 15, 2, &mut rng);
        let vocab = Vocabulary::train(&descs, 4, 3, &mut rng);
        let bytes = vocab.serialize();
        let back = Vocabulary::deserialize(&bytes).expect("roundtrip");
        assert_eq!(vocab, back);
        // ... and the reloaded tree quantises identically.
        for d in &descs {
            assert_eq!(vocab.word_of(d), back.word_of(d));
        }
        assert_eq!(
            vocab.transform(&descs).iter().collect::<Vec<_>>(),
            back.transform(&descs).iter().collect::<Vec<_>>()
        );
    }

    #[test]
    fn vocabulary_deserialize_rejects_bad_input_without_panicking() {
        let mut rng = DeterministicRng::new("t", 14);
        let (descs, _) = clustered(4, 10, 1, &mut rng);
        let good = Vocabulary::train(&descs, 3, 2, &mut rng).serialize();

        assert!(matches!(
            Vocabulary::deserialize(b"NOTAVOCA\x01\x00\x00\x00"),
            Err(Error::MapFormat(_))
        ));
        // Every truncation must be an error, never a panic.
        for n in 0..good.len() {
            assert!(
                Vocabulary::deserialize(&good[..n]).is_err(),
                "prefix {n} parsed"
            );
        }
        // A bumped version byte.
        let mut bumped = good.clone();
        bumped[8] = bumped[8].wrapping_add(7);
        assert!(matches!(
            Vocabulary::deserialize(&bumped),
            Err(Error::MapVersion { .. })
        ));
        // A child index pointing past the end of the node array.
        let mut broken = good.clone();
        let n = broken.len();
        broken[n - 12..n - 8].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(Vocabulary::deserialize(&broken).is_err());
    }

    #[test]
    fn bow_vector_is_l1_normalised_and_sorted() {
        let v = BowVector::from_pairs(&[(9, 1.0), (2, 3.0), (9, 1.0)]);
        let words: Vec<u32> = v.words().collect();
        assert_eq!(words, vec![2, 9]);
        let sum: f64 = v.iter().map(|(_, w)| w).sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-15);
        assert_relative_eq!(v.weight(2), 0.6, epsilon = 1e-15);
        assert_relative_eq!(v.weight(9), 0.4, epsilon = 1e-15);
        assert_eq!(v.weight(1000), 0.0);
        // Degenerate inputs collapse to the empty vector rather than NaN.
        assert!(BowVector::from_pairs(&[(1, 0.0)]).is_empty());
        assert!(BowVector::from_pairs(&[(1, f64::NAN)]).is_empty());
        assert!(BowVector::from_pairs(&[]).is_empty());
        assert_eq!(BowVector::empty().score(&BowVector::empty()), 0.0);
    }
}
