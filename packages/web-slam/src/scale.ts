/**
 * Scale sources, declared explicitly.
 *
 * spec.md §1: "The library never silently guesses scale. Callers choose an
 * anchor and accept its tradeoffs." This factory is where that choice is made,
 * and every option here names its cost in its own doc comment.
 */

import type { ScaleSourceSpec } from './types.js';

/** Fiducial families the `fiducial` source can detect. */
export type FiducialFamily = 'apriltag36h11';

/** Options for {@link ScaleSource.fiducial}. */
export interface FiducialOptions {
  family: FiducialFamily;
  /** Physical edge length of the tag's black border, in metres. */
  sizeMeters: number;
}

/** Options for {@link ScaleSource.declared}. */
export interface DeclaredOptions {
  /**
   * The real-world distance the user is declaring, in metres.
   *
   * Omit to let the UI supply it later via the returned handle; the session
   * stays up-to-scale until it arrives.
   */
  distanceMeters?: number;
  /**
   * Assumed precision of the user's two taps, in pixels. Feeds the reported
   * variance — the taps are exact in intent but not in execution, and a source
   * that reported zero variance here would be lying.
   */
  tapPrecisionPx?: number;
}

/** Options for {@link ScaleSource.learned}. */
export interface LearnedOptions {
  /**
   * URL or identifier of the depth model. **Downloads weights.** Prefer any
   * other source: spec.md §5 notes inertial reaches ~1% "with zero model
   * download", better than every learned prior.
   */
  model: string;
}

/**
 * The four rulers, plus the honest default and the persisted-map case.
 *
 * From spec.md §2 — every metric system that has ever shipped uses one or more
 * of these, and the architecture here makes the choice explicit instead of
 * burying it.
 */
export const ScaleSource = {
  /**
   * Up to scale. Positions are in arbitrary units and `scale.variance` is
   * `Infinity`.
   *
   * The default, and not a degraded mode: a renderer that only needs relative
   * camera motion is correctly served by it.
   */
  none(): ScaleSourceSpec {
    return { kind: 'none', config: Object.freeze({}) };
  },

  /**
   * One user tap on a known distance. Exact, free, needs one interaction.
   *
   * Cost: an interaction. Accuracy: exact, bounded only by tap precision.
   */
  declared(options: DeclaredOptions = {}): ScaleSourceSpec {
    if (
      options.distanceMeters !== undefined &&
      !(options.distanceMeters > 0)
    ) {
      throw new RangeError(
        `ScaleSource.declared: distanceMeters must be positive, got ${options.distanceMeters}`,
      );
    }
    return {
      kind: 'declared',
      config: Object.freeze({
        distanceMeters: options.distanceMeters,
        tapPrecisionPx: options.tapPrecisionPx ?? 3,
      }),
    };
  },

  /**
   * A marker of known physical size in the scene. Exact while visible.
   *
   * Cost: the object must be visible. Accuracy: exact.
   */
  fiducial(options: FiducialOptions): ScaleSourceSpec {
    if (!(options.sizeMeters > 0)) {
      throw new RangeError(
        `ScaleSource.fiducial: sizeMeters must be positive, got ${options.sizeMeters}`,
      );
    }
    return {
      kind: 'fiducial',
      config: Object.freeze({ ...options }),
    };
  },

  /**
   * Relocalize into a previously anchored map and inherit its scale.
   *
   * Cost: must relocalize; requires L4. Accuracy: inherits the original
   * anchor's variance **plus relocalization error** — the returned
   * `scale.variance` is always larger than the anchor's, never equal
   * (spec.md §4 L5).
   */
  map(savedMap: Uint8Array): ScaleSourceSpec {
    if (!(savedMap instanceof Uint8Array) || savedMap.byteLength === 0) {
      throw new TypeError(
        'ScaleSource.map: expected a non-empty Uint8Array from slam.saveMap()',
      );
    }
    return { kind: 'map', config: Object.freeze({ bytes: savedMap }) };
  },

  /**
   * A learned monocular depth prior. Opt-in; **downloads weights**.
   *
   * Cost: model download, GPU. Accuracy: several percent, domain-correlated.
   * The prediction is used as a constraint, never as truth — spec.md §5 records
   * MDE-VIO finding direct metric depth predictions insufficient.
   */
  learned(options: LearnedOptions): ScaleSourceSpec {
    if (!options.model) {
      throw new TypeError('ScaleSource.learned: `model` is required');
    }
    return { kind: 'learned', config: Object.freeze({ ...options }) };
  },

  /**
   * Double-integrated acceleration. ~1% given excitation, with zero download.
   *
   * Cost: needs motion, bias estimation and tight time sync. **Requires sensor
   * tier 3**, and `WebSlam.create` rejects if L0 is unavailable rather than
   * silently degrading (spec.md §3).
   *
   * Accuracy collapses as translational acceleration vanishes, which is how
   * people hold phones — so this source returns no estimate at all during a
   * static hold rather than a confident wrong one.
   */
  inertial(): ScaleSourceSpec {
    return { kind: 'inertial', config: Object.freeze({}) };
  },
} as const;
