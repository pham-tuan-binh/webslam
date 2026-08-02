//! Versioned binary map format.
//!
//! spec.md §2 gives the reason this file exists: *"Anchor scale once by any of
//! the above, persist the map, and every subsequent session recovers metric by
//! relocalizing. This converts scale from a hard per-session estimation problem
//! into a one-time one."* The scale anchor is therefore not an afterthought
//! stored somewhere in the payload — it sits in the header, next to the camera
//! model, because a map without a declared anchor and a declared lens is not
//! reusable as a ruler.
//!
//! ## Design rules
//!
//! - **Little-endian, explicit, no serde.** Every field is written by hand. A
//!   map is a long-lived artifact that a future build must read using nothing
//!   but this file as documentation, and a derive macro's field order is not
//!   documentation.
//! - **Unknown versions are rejected**, with [`Error::MapVersion`] naming both
//!   versions. Silently reading a format you do not understand corrupts the
//!   anchor, which is worse than refusing.
//! - **Hostile input returns `Err`, never panics.** Every read is bounds
//!   checked ([`Reader`]) and a CRC-32 over the payload catches the corruption
//!   that still parses.
//! - Bag-of-words vectors are **recomputed** on load rather than stored. They
//!   are a pure function of the descriptors and the vocabulary, both of which
//!   are in the file; storing them would let the two drift apart.

use crate::db::MapDb;
use crate::descriptor::{BinaryDescriptor, DESCRIPTOR_BYTES};
use crate::keyframe::{Keyframe, KeyframeId, Landmark, LandmarkId};
use crate::vocabulary::Vocabulary;
use std::sync::Arc;
use wslam_core::{
    CameraIntrinsics, CameraModel, Error, RadialTangential, Result, ScaleEstimate, ScaleKind, Se3,
    So3, Timestamp, Vec2, Vec3, MAP_FORMAT_VERSION,
};

/// Magic bytes at the head of a serialised map.
pub const MAP_MAGIC: &[u8; 8] = b"WSLAMMAP";

/// Byte offset of the CRC field, and therefore the start of the checksummed
/// payload.
const CRC_OFFSET: usize = 12;
const HEADER_BYTES: usize = 16;

/// Scale-anchor block: kind tag, padding, value, variance.
const ANCHOR_BYTES: usize = 1 + 7 + 8 + 8;
/// Camera-model tag plus its padding.
const MODEL_TAG_BYTES: usize = 1 + 7;
/// `fx, fy, cx, cy` then `width, height` then the five distortion
/// coefficients — see [`write_intrinsics`], which this must track.
const INTRINSICS_BYTES: usize = 4 * 8 + 4 + 4 + 5 * 8;
/// Quaternion `wxyz` then translation `xyz` — see [`write_pose`].
const POSE_BYTES: usize = 7 * 8;

/// Byte offset of the keyframe-count field, i.e. the end of the fixed header.
///
/// Derived rather than hand-counted, and pinned by
/// `the_record_counts_are_where_the_format_says_they_are`. The previous
/// hand-counted value was 32 bytes short — it charged 48 bytes for the
/// reference intrinsics instead of 80 — which silently aimed the
/// absurd-record-count test at a distortion coefficient instead.
const RECORD_COUNT_OFFSET: usize = HEADER_BYTES + ANCHOR_BYTES + MODEL_TAG_BYTES + INTRINSICS_BYTES;

/// Smallest per-record sizes, used to reject an oversized count before
/// allocating for it.
///
/// These must be genuine *lower* bounds — a record can be longer, never
/// shorter — or a valid map would be rejected. A keyframe is its id,
/// timestamp, pose, intrinsics and feature count, with zero features.
const MIN_KEYFRAME_BYTES: usize = 8 + 8 + POSE_BYTES + INTRINSICS_BYTES + 4;
/// A landmark is its id, position, descriptor and observation count, with zero
/// observations.
const MIN_LANDMARK_BYTES: usize = 8 + 24 + DESCRIPTOR_BYTES + 4;

/// Serialise a map, vocabulary included.
///
/// The vocabulary travels with the map because word ids are meaningless without
/// it and a map is useless without its inverted index.
#[must_use]
pub fn serialize_map(db: &MapDb) -> Vec<u8> {
    let mut out = Vec::with_capacity(4096);
    out.extend_from_slice(MAP_MAGIC);
    out.extend_from_slice(&MAP_FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes()); // flags, reserved
    out.extend_from_slice(&0u32.to_le_bytes()); // CRC placeholder
    debug_assert_eq!(out.len(), HEADER_BYTES);

    // --- the point of the format ---
    let anchor = db.scale_anchor();
    out.push(anchor.source.tag());
    out.extend_from_slice(&[0u8; 7]); // pad to 8
    out.extend_from_slice(&anchor.value.to_le_bytes());
    out.extend_from_slice(&anchor.variance.to_le_bytes());

    // Reference camera model. Per-keyframe intrinsics still travel with each
    // keyframe (L2 refines focal length online), but a reader needs to know at
    // header time whether this map was built with distortion modelled at all.
    let reference = db
        .keyframes()
        .next()
        .map_or_else(|| CameraIntrinsics::from_focal(1.0, 0, 0), |k| k.intrinsics);
    let model = if reference.distortion.is_identity() {
        CameraModel::Pinhole
    } else {
        CameraModel::RadialTangential
    };
    out.push(match model {
        CameraModel::Pinhole => 0,
        CameraModel::RadialTangential => 1,
    });
    out.extend_from_slice(&[0u8; 7]);
    write_intrinsics(&mut out, &reference);

    debug_assert_eq!(out.len(), RECORD_COUNT_OFFSET);
    out.extend_from_slice(&(db.keyframe_count() as u32).to_le_bytes());
    out.extend_from_slice(&(db.landmark_count() as u32).to_le_bytes());
    out.extend_from_slice(&db.next_keyframe_id().to_le_bytes());
    out.extend_from_slice(&db.next_landmark_id().to_le_bytes());

    let vocab = db.vocabulary().serialize();
    out.extend_from_slice(&(vocab.len() as u64).to_le_bytes());
    out.extend_from_slice(&vocab);

    for kf in db.keyframes() {
        out.extend_from_slice(&kf.id.0.to_le_bytes());
        out.extend_from_slice(&kf.timestamp.nanos().to_le_bytes());
        write_pose(&mut out, &kf.pose);
        write_intrinsics(&mut out, &kf.intrinsics);
        let n = kf
            .keypoints
            .len()
            .min(kf.descriptors.len())
            .min(kf.landmarks.len());
        out.extend_from_slice(&(n as u32).to_le_bytes());
        for i in 0..n {
            out.extend_from_slice(&kf.keypoints[i].x.to_le_bytes());
            out.extend_from_slice(&kf.keypoints[i].y.to_le_bytes());
            out.extend_from_slice(&kf.descriptors[i].0);
            match kf.landmarks[i] {
                Some(id) => {
                    out.push(1);
                    out.extend_from_slice(&id.0.to_le_bytes());
                }
                None => {
                    out.push(0);
                    out.extend_from_slice(&0u64.to_le_bytes());
                }
            }
        }
    }

    for lm in db.landmarks() {
        out.extend_from_slice(&lm.id.0.to_le_bytes());
        out.extend_from_slice(&lm.position.x.to_le_bytes());
        out.extend_from_slice(&lm.position.y.to_le_bytes());
        out.extend_from_slice(&lm.position.z.to_le_bytes());
        out.extend_from_slice(&lm.descriptor.0);
        out.extend_from_slice(&(lm.observations.len() as u32).to_le_bytes());
        for obs in &lm.observations {
            out.extend_from_slice(&obs.0.to_le_bytes());
        }
    }

    let crc = crc32(&out[CRC_OFFSET + 4..]);
    out[CRC_OFFSET..CRC_OFFSET + 4].copy_from_slice(&crc.to_le_bytes());
    out
}

/// Read a map produced by [`serialize_map`].
///
/// Returns the map and the vocabulary it was built against, so the caller can
/// share the same `Arc` with anything else that needs to transform descriptors.
pub fn deserialize_map(bytes: &[u8]) -> Result<(MapDb, Arc<Vocabulary>)> {
    let mut r = Reader::new(bytes);
    if r.bytes(8)? != MAP_MAGIC {
        return Err(Error::MapFormat("map magic mismatch".into()));
    }
    let version = r.u16()?;
    if version != MAP_FORMAT_VERSION {
        return Err(Error::MapVersion {
            found: version,
            supported: MAP_FORMAT_VERSION,
        });
    }
    let _flags = r.u16()?;
    let crc = r.u32()?;
    if crc32(&bytes[CRC_OFFSET + 4..]) != crc {
        return Err(Error::MapFormat("map checksum mismatch".into()));
    }

    let kind_tag = r.u8()?;
    let scale_source = ScaleKind::from_tag(kind_tag)
        .ok_or_else(|| Error::MapFormat(format!("unknown scale kind tag {kind_tag}")))?;
    r.skip(7)?;
    let scale_value = r.f64()?;
    let scale_variance = r.f64()?;
    if !scale_value.is_finite() || scale_value <= 0.0 || scale_variance.is_nan() {
        return Err(Error::MapFormat(
            "scale anchor is not a usable estimate".into(),
        ));
    }
    let anchor = ScaleEstimate {
        source: scale_source,
        value: scale_value,
        variance: scale_variance,
    };

    let model_tag = r.u8()?;
    if model_tag > 1 {
        return Err(Error::MapFormat(format!(
            "unknown camera model tag {model_tag}"
        )));
    }
    r.skip(7)?;
    let _reference = read_intrinsics(&mut r)?;

    let keyframe_count = r.u32()? as usize;
    let landmark_count = r.u32()? as usize;
    let next_keyframe = r.u64()?;
    let next_landmark = r.u64()?;
    r.check_remaining(
        keyframe_count.saturating_mul(MIN_KEYFRAME_BYTES)
            + landmark_count.saturating_mul(MIN_LANDMARK_BYTES),
    )?;

    let vocab_len = r.u64()?;
    let vocab_bytes = r.bytes(usize::try_from(vocab_len).map_err(|_| {
        Error::MapFormat("vocabulary length does not fit in this address space".into())
    })?)?;
    let vocabulary = Arc::new(Vocabulary::deserialize(vocab_bytes)?);

    let mut db = MapDb::new(Arc::clone(&vocabulary));
    for _ in 0..keyframe_count {
        let id = KeyframeId(r.u64()?);
        let timestamp = Timestamp::from_nanos(r.i64()?);
        let pose = read_pose(&mut r)?;
        let intrinsics = read_intrinsics(&mut r)?;
        let n = r.u32()? as usize;
        r.check_remaining(n.saturating_mul(8 + 8 + DESCRIPTOR_BYTES + 9))?;
        let mut keypoints = Vec::with_capacity(n);
        let mut descriptors = Vec::with_capacity(n);
        let mut landmarks = Vec::with_capacity(n);
        for _ in 0..n {
            let x = r.f64()?;
            let y = r.f64()?;
            let mut d = [0u8; DESCRIPTOR_BYTES];
            d.copy_from_slice(r.bytes(DESCRIPTOR_BYTES)?);
            let tag = r.u8()?;
            let lid = r.u64()?;
            keypoints.push(Vec2::new(x, y));
            descriptors.push(BinaryDescriptor(d));
            landmarks.push(match tag {
                0 => None,
                1 => Some(LandmarkId(lid)),
                other => {
                    return Err(Error::MapFormat(format!(
                        "keyframe landmark tag {other} is neither 0 nor 1"
                    )))
                }
            });
        }
        let bow = vocabulary.transform(&descriptors);
        db.insert_keyframe(Keyframe {
            id,
            timestamp,
            pose,
            keypoints,
            descriptors,
            landmarks,
            bow,
            intrinsics,
        });
    }

    for _ in 0..landmark_count {
        let id = LandmarkId(r.u64()?);
        let position = Vec3::new(r.f64()?, r.f64()?, r.f64()?);
        if !position.iter().all(|v| v.is_finite()) {
            return Err(Error::MapFormat("landmark position is not finite".into()));
        }
        let mut d = [0u8; DESCRIPTOR_BYTES];
        d.copy_from_slice(r.bytes(DESCRIPTOR_BYTES)?);
        let n = r.u32()? as usize;
        r.check_remaining(n.saturating_mul(8))?;
        let mut observations = Vec::with_capacity(n);
        for _ in 0..n {
            observations.push(KeyframeId(r.u64()?));
        }
        db.insert_landmark(Landmark {
            id,
            position,
            descriptor: BinaryDescriptor(d),
            observations,
        });
    }

    db.set_next_ids(next_keyframe, next_landmark);
    db.set_scale_anchor(anchor);
    Ok((db, vocabulary))
}

fn write_pose(out: &mut Vec<u8>, pose: &Se3) {
    let q = pose.rotation().quaternion();
    for v in [q.w, q.i, q.j, q.k] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    let t = pose.translation();
    for v in [t.x, t.y, t.z] {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

fn read_pose(r: &mut Reader<'_>) -> Result<Se3> {
    let (w, x, y, z) = (r.f64()?, r.f64()?, r.f64()?, r.f64()?);
    let t = Vec3::new(r.f64()?, r.f64()?, r.f64()?);
    if ![w, x, y, z].iter().all(|v| v.is_finite())
        || !t.iter().all(|v| v.is_finite())
        || (w * w + x * x + y * y + z * z) < 1e-12
    {
        return Err(Error::MapFormat(
            "keyframe pose is not a valid transform".into(),
        ));
    }
    Ok(Se3::new(So3::from_wxyz(w, x, y, z), t))
}

fn write_intrinsics(out: &mut Vec<u8>, k: &CameraIntrinsics) {
    for v in [k.fx, k.fy, k.cx, k.cy] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&k.width.to_le_bytes());
    out.extend_from_slice(&k.height.to_le_bytes());
    let d = k.distortion;
    for v in [d.k1, d.k2, d.p1, d.p2, d.k3] {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

fn read_intrinsics(r: &mut Reader<'_>) -> Result<CameraIntrinsics> {
    let (fx, fy, cx, cy) = (r.f64()?, r.f64()?, r.f64()?, r.f64()?);
    let width = r.u32()?;
    let height = r.u32()?;
    let distortion = RadialTangential {
        k1: r.f64()?,
        k2: r.f64()?,
        p1: r.f64()?,
        p2: r.f64()?,
        k3: r.f64()?,
    };
    if ![fx, fy, cx, cy].iter().all(|v| v.is_finite())
        || [
            distortion.k1,
            distortion.k2,
            distortion.p1,
            distortion.p2,
            distortion.k3,
        ]
        .iter()
        .any(|v| !v.is_finite())
    {
        return Err(Error::MapFormat("camera intrinsics are not finite".into()));
    }
    Ok(CameraIntrinsics {
        fx,
        fy,
        cx,
        cy,
        width,
        height,
        distortion,
    })
}

/// Bounds-checked little-endian reader.
///
/// Every accessor returns [`Error::MapFormat`] on truncation. This is the type
/// that makes "hostile input never panics" a property of the module rather than
/// of the author's attention.
pub(crate) struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }

    pub(crate) fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| Error::MapFormat("record length overflows the address space".into()))?;
        if end > self.buf.len() {
            return Err(Error::MapFormat(format!(
                "truncated at offset {} (wanted {n} bytes, {} left)",
                self.pos,
                self.buf.len() - self.pos
            )));
        }
        let out = &self.buf[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    pub(crate) fn skip(&mut self, n: usize) -> Result<()> {
        self.bytes(n).map(|_| ())
    }

    /// Reject a header field that claims more bytes than remain, before any
    /// allocation is sized from it.
    pub(crate) fn check_remaining(&self, needed: usize) -> Result<()> {
        if needed > self.buf.len().saturating_sub(self.pos) {
            return Err(Error::MapFormat(format!(
                "header claims {needed} bytes of records but only {} remain",
                self.buf.len() - self.pos
            )));
        }
        Ok(())
    }

    pub(crate) fn u8(&mut self) -> Result<u8> {
        Ok(self.bytes(1)?[0])
    }

    pub(crate) fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.array::<2>()?))
    }

    pub(crate) fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.array::<4>()?))
    }

    pub(crate) fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.array::<4>()?))
    }

    pub(crate) fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.array::<8>()?))
    }

    pub(crate) fn i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.array::<8>()?))
    }

    pub(crate) fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.array::<8>()?))
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut out = [0u8; N];
        out.copy_from_slice(self.bytes(N)?);
        Ok(out)
    }
}

/// CRC-32 (IEEE 802.3), table built once.
///
/// Not for security — for the corruption that still parses. Truncation is
/// caught by [`Reader`]; a flipped bit in the middle of a landmark position is
/// not, and a silently wrong map is exactly the failure mode spec.md §5 calls
/// irrecoverable.
fn crc32(data: &[u8]) -> u32 {
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut t = [0u32; 256];
        for (i, slot) in t.iter_mut().enumerate() {
            let mut c = i as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 {
                    0xEDB8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                };
            }
            *slot = c;
        }
        t
    });
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc = table[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth;
    use proptest::prelude::*;
    use wslam_core::DeterministicRng;

    fn assert_maps_equal(a: &MapDb, b: &MapDb) {
        assert_eq!(a.scale_anchor().source, b.scale_anchor().source);
        assert_eq!(
            a.scale_anchor().value.to_bits(),
            b.scale_anchor().value.to_bits(),
            "scale anchor value must survive bit for bit"
        );
        assert_eq!(
            a.scale_anchor().variance.to_bits(),
            b.scale_anchor().variance.to_bits(),
            "scale anchor variance must survive bit for bit, infinity included"
        );
        assert_eq!(a.keyframe_count(), b.keyframe_count());
        assert_eq!(a.landmark_count(), b.landmark_count());
        assert_eq!(a.next_keyframe_id(), b.next_keyframe_id());
        assert_eq!(a.next_landmark_id(), b.next_landmark_id());

        for (x, y) in a.keyframes().zip(b.keyframes()) {
            assert_eq!(x.id, y.id);
            assert_eq!(x.timestamp, y.timestamp);
            assert_eq!(x.descriptors, y.descriptors);
            assert_eq!(x.landmarks, y.landmarks);
            assert_eq!(x.intrinsics, y.intrinsics);
            assert_eq!(x.bow, y.bow);
            for (p, q) in x.keypoints.iter().zip(y.keypoints.iter()) {
                assert_eq!(p.x.to_bits(), q.x.to_bits());
                assert_eq!(p.y.to_bits(), q.y.to_bits());
            }
            // Poses survive to round-off of the quaternion representation.
            assert!(
                x.pose.minus(&y.pose).norm() < 1e-12,
                "{} pose drifted: {:?}",
                x.id,
                x.pose.minus(&y.pose)
            );
        }
        for (x, y) in a.landmarks().zip(b.landmarks()) {
            assert_eq!(x.id, y.id);
            assert_eq!(x.descriptor, y.descriptor);
            assert_eq!(x.observations, y.observations);
            for i in 0..3 {
                assert_eq!(x.position[i].to_bits(), y.position[i].to_bits());
            }
        }
        // The inverted index is rebuilt on load and must agree.
        assert_eq!(a.indexed_word_count(), b.indexed_word_count());
        for kf in a.keyframes() {
            for w in kf.bow.words() {
                assert_eq!(a.keyframes_with_word(w), b.keyframes_with_word(w));
            }
        }
    }

    #[test]
    fn roundtrip_preserves_an_anchored_map() {
        let (mut db, _) = synth::corridor_map(8, 20260801);
        db.set_scale_anchor(ScaleEstimate::metric(ScaleKind::Fiducial, 1.375, 2.5e-4));
        let bytes = serialize_map(&db);
        let (back, vocab) = deserialize_map(&bytes).expect("roundtrip");
        assert_maps_equal(&db, &back);
        assert_eq!(&**db.vocabulary(), &*vocab);
        // Re-serialising the reloaded map reproduces the same bytes.
        assert_eq!(serialize_map(&back), bytes);
    }

    #[test]
    fn roundtrip_preserves_an_unanchored_map_with_infinite_variance() {
        // ScaleKind::None carries variance = +inf. A format that writes it as a
        // float and reads back NaN or a finite number would silently promote an
        // up-to-scale map to a metric one, which spec.md §3 forbids outright.
        let (db, _) = synth::corridor_map(4, 11);
        assert!(db.scale_anchor().variance.is_infinite());
        let (back, _) = deserialize_map(&serialize_map(&db)).expect("roundtrip");
        assert_eq!(back.scale_anchor().source, ScaleKind::None);
        assert!(back.scale_anchor().variance.is_infinite());
        assert_maps_equal(&db, &back);
    }

    #[test]
    fn roundtrip_preserves_lens_distortion() {
        let (mut db, _) = synth::corridor_map(3, 12);
        let distortion = RadialTangential {
            k1: -0.283,
            k2: 0.091,
            p1: 0.0012,
            p2: -0.0008,
            k3: -0.011,
        };
        let ids: Vec<KeyframeId> = db.keyframes().map(|k| k.id).collect();
        for id in ids {
            db.keyframe_mut(id).unwrap().intrinsics.distortion = distortion;
        }
        let bytes = serialize_map(&db);
        // Header records that this map is not a pinhole map.
        assert_eq!(bytes[HEADER_BYTES + 24], 1, "camera model tag");
        let (back, _) = deserialize_map(&bytes).expect("roundtrip");
        for kf in back.keyframes() {
            assert_eq!(kf.intrinsics.distortion, distortion);
        }
    }

    #[test]
    fn empty_map_roundtrips() {
        let db = MapDb::new(Arc::new(Vocabulary::empty()));
        let (back, _) = deserialize_map(&serialize_map(&db)).expect("roundtrip");
        assert!(back.is_empty());
        assert_maps_equal(&db, &back);
    }

    #[test]
    fn bumped_version_returns_map_version_error() {
        let (db, _) = synth::corridor_map(3, 1);
        let mut bytes = serialize_map(&db);
        bytes[8] = bytes[8].wrapping_add(1);
        // The CRC covers only the payload, so the version check is what fires.
        match deserialize_map(&bytes) {
            Err(Error::MapVersion { found, supported }) => {
                assert_eq!(found, MAP_FORMAT_VERSION + 1);
                assert_eq!(supported, MAP_FORMAT_VERSION);
            }
            other => panic!("expected MapVersion, got {other:?}"),
        }
        // ... and a far-future version too.
        bytes[8..10].copy_from_slice(&u16::MAX.to_le_bytes());
        assert!(matches!(
            deserialize_map(&bytes),
            Err(Error::MapVersion {
                found: u16::MAX,
                ..
            })
        ));
    }

    #[test]
    fn bad_magic_is_rejected() {
        let (db, _) = synth::corridor_map(2, 1);
        let mut bytes = serialize_map(&db);
        bytes[0] = b'X';
        assert!(matches!(deserialize_map(&bytes), Err(Error::MapFormat(_))));
        assert!(matches!(deserialize_map(b""), Err(Error::MapFormat(_))));
        assert!(matches!(
            deserialize_map(b"WSLAM"),
            Err(Error::MapFormat(_))
        ));
    }

    #[test]
    fn every_truncation_is_an_error_and_never_a_panic() {
        let (db, _) = synth::corridor_map(4, 2);
        let bytes = serialize_map(&db);
        assert!(bytes.len() > 1000, "test needs a non-trivial map");
        for n in 0..bytes.len() {
            let r = deserialize_map(&bytes[..n]);
            assert!(r.is_err(), "prefix of length {n} parsed as a valid map");
        }
        assert!(deserialize_map(&bytes).is_ok());
    }

    #[test]
    fn a_single_corrupted_byte_is_caught_by_the_checksum() {
        let (mut db, _) = synth::corridor_map(4, 3);
        db.set_scale_anchor(ScaleEstimate::metric(ScaleKind::Declared, 2.0, 1e-6));
        let bytes = serialize_map(&db);
        let mut rng = DeterministicRng::new("corrupt", 20260801);
        for _ in 0..200 {
            let mut bad = bytes.clone();
            let i = CRC_OFFSET + 4 + rng.below(bad.len() - CRC_OFFSET - 4);
            let flip = 1u8 << rng.below(8);
            bad[i] ^= flip;
            assert!(
                deserialize_map(&bad).is_err(),
                "flipping bit {flip:#b} of byte {i} was not detected"
            );
        }
    }

    #[test]
    fn the_record_counts_are_where_the_format_says_they_are() {
        // [`RECORD_COUNT_OFFSET`] used to be hand-counted, and was 32 bytes
        // short because it charged 48 bytes for the reference intrinsics
        // instead of 80. Nothing caught it, because the only test that used the
        // offset just checked that a *rejection* happened somewhere. Pin the
        // field by its contents so the constant cannot drift again.
        let (db, _) = synth::corridor_map(3, 9);
        let bytes = serialize_map(&db);
        let at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap()) as usize;
        assert_eq!(at(RECORD_COUNT_OFFSET), db.keyframe_count());
        assert_eq!(at(RECORD_COUNT_OFFSET + 4), db.landmark_count());

        // MIN_KEYFRAME_BYTES and MIN_LANDMARK_BYTES gate every record-count
        // claim, so they must be genuine lower bounds: if either overstates a
        // record, a perfectly good map is refused on load.
        let fixed = RECORD_COUNT_OFFSET + 4 + 4 + 8 + 8 + 8 + db.vocabulary().serialize().len();
        assert!(
            bytes.len()
                >= fixed
                    + db.keyframe_count() * MIN_KEYFRAME_BYTES
                    + db.landmark_count() * MIN_LANDMARK_BYTES,
            "the per-record minima are not lower bounds"
        );
        assert!(deserialize_map(&bytes).is_ok());
    }

    #[test]
    fn an_absurd_record_count_is_rejected_without_allocating_for_it() {
        let (db, _) = synth::corridor_map(2, 4);
        let sane = serialize_map(&db);
        let patched = |offset: usize, value: u32| {
            let mut bytes = sane.clone();
            bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            // Re-checksum, or the CRC would reject it first and the sanity
            // check under test would never run.
            let crc = crc32(&bytes[CRC_OFFSET + 4..]);
            bytes[CRC_OFFSET..CRC_OFFSET + 4].copy_from_slice(&crc.to_le_bytes());
            bytes
        };
        let field = |offset: usize| {
            u32::from_le_bytes(sane[offset..offset + 4].try_into().unwrap()) as usize
        };

        // Every count in the format that sizes an allocation. Each must be
        // refused by the bounds check *before* the loop that would allocate
        // from it — which is what `Reader::check_remaining` is, and what the
        // "header claims" message identifies. Any other error would mean the
        // reader got as far as trying.
        let vocab_len = db.vocabulary().serialize().len();
        let first_keyframe = RECORD_COUNT_OFFSET + 4 + 4 + 8 + 8 + 8 + vocab_len;
        let feature_count = first_keyframe + 8 + 8 + POSE_BYTES + INTRINSICS_BYTES;
        let cases = [
            ("keyframe count", RECORD_COUNT_OFFSET, db.keyframe_count()),
            (
                "landmark count",
                RECORD_COUNT_OFFSET + 4,
                db.landmark_count(),
            ),
            (
                "first keyframe's feature count",
                feature_count,
                db.keyframes().next().unwrap().len(),
            ),
        ];
        for (name, offset, expected) in cases {
            // Confirm we are aiming at the field we think we are. Without this
            // the test can pass while patching something inert.
            assert_eq!(field(offset), expected, "{name} is not at offset {offset}");
            for claim in [u32::MAX, u32::MAX / 2, 1 << 20] {
                match deserialize_map(&patched(offset, claim)) {
                    Err(Error::MapFormat(m)) => assert!(
                        m.starts_with("header claims"),
                        "{name} = {claim} was rejected, but not by the \
                         pre-allocation bounds check: {m}"
                    ),
                    other => panic!("{name} = {claim} was not rejected: {other:?}"),
                }
            }
        }
        // The unpatched map still loads, so none of the above is an artifact of
        // the re-checksumming.
        assert!(deserialize_map(&sane).is_ok());
    }

    #[test]
    fn an_unknown_scale_kind_tag_is_rejected() {
        let (db, _) = synth::corridor_map(2, 5);
        let mut bytes = serialize_map(&db);
        bytes[HEADER_BYTES] = 200;
        let crc = crc32(&bytes[CRC_OFFSET + 4..]);
        bytes[CRC_OFFSET..CRC_OFFSET + 4].copy_from_slice(&crc.to_le_bytes());
        assert!(matches!(deserialize_map(&bytes), Err(Error::MapFormat(_))));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(24))]

        /// spec.md §6 Tier 1 names "map serialise/deserialise round-trip" as a
        /// property test. Random maps in, identical poses, landmarks,
        /// descriptors and — the point of the format — the scale anchor out.
        #[test]
        fn random_maps_survive_a_roundtrip(
            seed in 0u64..4096,
            keyframes in 0usize..7,
            kind in 0u8..6,
            value in 0.01f64..100.0,
            variance in 0.0f64..10.0,
        ) {
            let (mut db, _) = synth::corridor_map(keyframes, seed);
            let source = ScaleKind::from_tag(kind).unwrap();
            let anchor = if source == ScaleKind::None {
                ScaleEstimate::unscaled()
            } else {
                ScaleEstimate::metric(source, value, variance)
            };
            db.set_scale_anchor(anchor);

            let bytes = serialize_map(&db);
            let (back, _) = deserialize_map(&bytes).expect("roundtrip");
            assert_maps_equal(&db, &back);
        }

        /// Arbitrary bytes must never panic the reader.
        #[test]
        fn arbitrary_bytes_never_panic(raw in proptest::collection::vec(any::<u8>(), 0..512)) {
            let _ = deserialize_map(&raw);
            // ... and neither must a valid header followed by garbage.
            let mut framed = MAP_MAGIC.to_vec();
            framed.extend_from_slice(&MAP_FORMAT_VERSION.to_le_bytes());
            framed.extend_from_slice(&0u16.to_le_bytes());
            framed.extend_from_slice(&crc32(&raw).to_le_bytes());
            framed.extend_from_slice(&raw);
            let _ = deserialize_map(&framed);
        }
    }

    #[test]
    fn crc32_matches_the_published_check_value() {
        // The IEEE 802.3 check value for "123456789" is 0xCBF43926.
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
        assert_ne!(crc32(b"a"), crc32(b"b"));
    }

    #[test]
    fn reader_reports_truncation_rather_than_panicking() {
        let mut r = Reader::new(&[1, 2, 3]);
        assert_eq!(r.u8().unwrap(), 1);
        assert!(r.u32().is_err());
        assert!(r.bytes(usize::MAX).is_err());
        assert!(Reader::new(&[]).check_remaining(1).is_err());
        assert!(Reader::new(&[0]).check_remaining(1).is_ok());
    }
}
