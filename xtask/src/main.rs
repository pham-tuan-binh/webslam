//! `cargo xtask` — the build automation.
//!
//! spec.md §7: *"`xtask` rather than Makefiles, keeping the build in Rust and
//! cross-platform for a team that will be on mixed machines."*
//!
//! Run `cargo xtask --help` for the task list.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(name = "xtask", about = "web-slam build automation", version)]
struct Cli {
    #[command(subcommand)]
    task: Task,
}

#[derive(Subcommand)]
enum Task {
    /// Build the wasm artifact into `packages/web-slam/pkg/`.
    BuildWasm {
        /// Build without the WebGPU front-end. The CPU reference is ~25x
        /// slower on the image stages, so this is for debugging only.
        #[arg(long)]
        no_gpu: bool,
        /// Optimise for size and run `wasm-opt`. Off for fast iteration.
        #[arg(long)]
        release: bool,
        /// Build the `SharedArrayBuffer` + rayon variant.
        ///
        /// See docs/DECISIONS.md D2: the default build is single-threaded and
        /// embeddable anywhere. This variant requires the embedder to serve
        /// COOP/COEP headers, which breaks third-party embedding.
        #[arg(long)]
        threads: bool,
    },
    /// Run the tiered test suite from spec.md §6.
    Test {
        #[arg(value_enum, default_value_t = Tier::One)]
        tier: Tier,
    },
    /// Replay a dataset sequence through the native build.
    Replay {
        /// Sequence directory, e.g. `datasets/euroc/MH_01_easy`.
        sequence: PathBuf,
        /// Write a rerun session for later scrubbing.
        #[arg(long)]
        rrd: Option<PathBuf>,
    },
    /// Re-record the checked-in ATE and scale baselines.
    ///
    /// Destructive by design: it overwrites the regression wall. Requires
    /// `--confirm` so it cannot be run by reflex when CI goes red.
    RegenBaselines {
        #[arg(long)]
        confirm: bool,
    },
    /// Fetch the replay datasets. They are gitignored; this is how you get them.
    FetchDatasets {
        #[arg(value_enum, default_value_t = Dataset::All)]
        which: Dataset,
    },
    /// Train a vocabulary artifact from a descriptor dump.
    TrainVocab {
        descriptors: PathBuf,
        #[arg(long, default_value_t = 10)]
        branching: usize,
        #[arg(long, default_value_t = 5)]
        depth: usize,
        #[arg(long, default_value_t = 20260801)]
        seed: u64,
        #[arg(long, default_value = "vocab/wslam-vocab-v1.bin")]
        out: PathBuf,
    },
    /// Check the architectural invariants spec.md §6 makes structural.
    CheckInvariants,
    /// Everything CI runs, in CI's order. Run before pushing.
    Ci,
}

#[derive(Copy, Clone, ValueEnum)]
enum Tier {
    /// Pure, synthetic, closed-form. Under 10 s; every commit.
    One,
    /// Replay against datasets. Subset per commit, full nightly.
    Two,
    /// Rig captures against metric ground truth. Not implemented; see
    /// docs/VERIFICATION.md for what that leaves unvalidated.
    Three,
    /// Browser device matrix. Per milestone.
    Four,
}

#[derive(Copy, Clone, ValueEnum, PartialEq)]
enum Dataset {
    All,
    Euroc,
    TumVi,
    SevenScenes,
}

fn main() -> Result<()> {
    match Cli::parse().task {
        Task::BuildWasm { release, threads, no_gpu } => build_wasm(release, threads, no_gpu),
        Task::Test { tier } => test(tier),
        Task::Replay { sequence, rrd } => replay(&sequence, rrd.as_deref()),
        Task::RegenBaselines { confirm } => regen_baselines(confirm),
        Task::FetchDatasets { which } => fetch_datasets(which),
        Task::TrainVocab {
            descriptors,
            branching,
            depth,
            seed,
            out,
        } => train_vocab(&descriptors, branching, depth, seed, &out),
        Task::CheckInvariants => check_invariants().map(|_| ()),
        Task::Ci => ci(),
    }
}

fn root() -> PathBuf {
    // `xtask` lives one level below the workspace root.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask has a parent")
        .to_path_buf()
}

fn run(program: &str, args: &[&str], cwd: &Path) -> Result<()> {
    println!("\x1b[2m$ {program} {}\x1b[0m", args.join(" "));
    let status = Command::new(program)
        .args(args)
        .current_dir(cwd)
        .status()
        .with_context(|| format!("failed to spawn `{program}` — is it installed and on PATH?"))?;
    if !status.success() {
        bail!("`{program} {}` exited with {status}", args.join(" "));
    }
    Ok(())
}

fn have(program: &str) -> bool {
    Command::new(program)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn build_wasm(release: bool, threads: bool, no_gpu: bool) -> Result<()> {
    let root = root();
    if !have("wasm-bindgen") {
        bail!("wasm-bindgen-cli is missing. Install it with:\n  cargo install wasm-bindgen-cli");
    }

    let profile = if release { "wasm-release" } else { "dev" };
    let mut args = vec![
        "build",
        "-p",
        "wslam-wasm",
        "--target",
        "wasm32-unknown-unknown",
        "--profile",
        profile,
    ];
    let mut features: Vec<&str> = Vec::new();
    if threads {
        features.push("threads");
    }
    if !no_gpu {
        features.push("gpu");
    }
    let joined = features.join(",");
    if !joined.is_empty() {
        args.extend(["--features", &joined]);
    }
    run("cargo", &args, &root)?;

    let profile_dir = if release { "wasm-release" } else { "debug" };
    let wasm = root
        .join("target/wasm32-unknown-unknown")
        .join(profile_dir)
        .join("wslam_wasm.wasm");
    if !wasm.exists() {
        bail!("expected artifact missing: {}", wasm.display());
    }

    let out = root.join("packages/web-slam/pkg");
    std::fs::create_dir_all(&out)?;
    run(
        "wasm-bindgen",
        &[
            wasm.to_str().context("non-UTF-8 path")?,
            "--out-dir",
            out.to_str().context("non-UTF-8 path")?,
            "--target",
            "web",
        ],
        &root,
    )?;

    if release {
        if have("wasm-opt") {
            let path = out.join("wslam_wasm_bg.wasm");
            let before = std::fs::metadata(&path)?.len();
            let p = path.to_str().context("non-UTF-8 path")?;
            run(
                "wasm-opt",
                &["-Os", "--enable-bulk-memory", p, "-o", p],
                &root,
            )?;
            let after = std::fs::metadata(&path)?.len();
            println!(
                "wasm-opt: {:.0} KiB -> {:.0} KiB ({:.0}% smaller)",
                before as f64 / 1024.0,
                after as f64 / 1024.0,
                100.0 * (1.0 - after as f64 / before as f64)
            );
        } else {
            // Binary size is one of the three stated reasons for choosing Rust
            // (spec.md §7), so shipping an un-opted release build silently
            // would undercut the decision.
            println!(
                "\x1b[33mwarning\x1b[0m: wasm-opt not found; release artifact is unoptimised.\n  \
                 Install binaryen: brew install binaryen"
            );
        }
    }

    if threads {
        println!(
            "\n\x1b[33mThis build requires COOP/COEP headers:\x1b[0m\n  \
             Cross-Origin-Opener-Policy: same-origin\n  \
             Cross-Origin-Embedder-Policy: require-corp\n  \
             It cannot be embedded cross-origin. See docs/DECISIONS.md D2."
        );
    }
    println!("wrote {}", out.display());
    Ok(())
}

fn test(tier: Tier) -> Result<()> {
    let root = root();
    match tier {
        Tier::One => {
            // Must stay fast enough that nobody is tempted to skip it
            // (spec.md §6 Tier 1).
            run("cargo", &["test", "--workspace", "--all-features"], &root)
        }
        Tier::Two => {
            if !root.join("datasets/euroc").exists() {
                bail!("datasets/euroc is missing. Run `cargo xtask fetch-datasets euroc` first.");
            }
            run(
                "cargo",
                &["run", "-p", "wslam-replay", "--release", "--", "regress"],
                &root,
            )
        }
        Tier::Three => {
            bail!(
                "Tier 3 (rig ground truth) is not implemented: the robot-arm harness was 
\
                 removed as out of scope. L5 scale error and L6 NEES are currently validated 
\
                 synthetically only — see docs/VERIFICATION.md."
            )
        }
        Tier::Four => {
            bail!(
                "Tier 4 is the browser device matrix and is not automatable from here.\n\
                 Chrome Android: packages/demo + Playwright.\n\
                 iOS Safari: device lab or a human. See spec.md §6."
            )
        }
    }
}

fn replay(sequence: &Path, rrd: Option<&Path>) -> Result<()> {
    let root = root();
    let sequence = sequence.to_str().context("non-UTF-8 path")?;
    let mut args = vec![
        "run",
        "-p",
        "wslam-replay",
        "--release",
        "--",
        "run",
        sequence,
    ];
    let rrd_str;
    if let Some(path) = rrd {
        rrd_str = path.to_str().context("non-UTF-8 path")?.to_string();
        args.extend(["--rrd", &rrd_str]);
    }
    run("cargo", &args, &root)
}

fn regen_baselines(confirm: bool) -> Result<()> {
    if !confirm {
        // spec.md §6 Tier 2 calls the baselines "the regression wall". A wall
        // you can knock down by accident is not a wall.
        bail!(
            "This overwrites harness/baselines/, the Tier-2 regression wall.\n\
             Re-run with --confirm, and explain the change in the commit message."
        );
    }
    run(
        "cargo",
        &[
            "run",
            "-p",
            "wslam-replay",
            "--release",
            "--",
            "regress",
            "--write",
        ],
        &root(),
    )
}

fn fetch_datasets(which: Dataset) -> Result<()> {
    let root = root();
    let script = root.join("datasets/fetch.sh");
    let name = match which {
        Dataset::All => "all",
        Dataset::Euroc => "euroc",
        Dataset::TumVi => "tum-vi",
        Dataset::SevenScenes => "7scenes",
    };
    run(
        "sh",
        &[script.to_str().context("non-UTF-8 path")?, name],
        &root,
    )
}

fn train_vocab(
    descriptors: &Path,
    branching: usize,
    depth: usize,
    seed: u64,
    out: &Path,
) -> Result<()> {
    let root = root();
    let (b, d, s) = (branching.to_string(), depth.to_string(), seed.to_string());
    run(
        "cargo",
        &[
            "run",
            "-p",
            "wslam-replay",
            "--release",
            "--",
            "train-vocab",
            descriptors.to_str().context("non-UTF-8 path")?,
            "--branching",
            &b,
            "--depth",
            &d,
            "--seed",
            &s,
            "--out",
            out.to_str().context("non-UTF-8 path")?,
        ],
        &root,
    )
}

/// One violation of a structural invariant.
#[derive(Debug)]
struct Violation {
    file: PathBuf,
    line: usize,
    text: String,
    rule: &'static str,
}

/// Check the rules spec.md §6 makes structural rather than aspirational.
///
/// Written in Rust rather than as a grep in the CI YAML for one reason: a grep
/// cannot see comments or module boundaries, so it fires on doc comments and on
/// legitimate test-harness stopwatches. **A check that produces false positives
/// gets disabled within a week**, which is worse than no check at all.
fn check_invariants() -> Result<Vec<Violation>> {
    const CLOCK_CALLS: [&str; 4] = [
        "Instant::now",
        "SystemTime::now",
        "Date::now",
        "performance.now",
    ];
    const UNSEEDED_RNG: [&str; 4] = ["from_entropy", "thread_rng", "rand::random", "OsRng"];

    let root = root();
    let mut violations = Vec::new();
    let mut files = Vec::new();
    collect_rust_files(&root.join("crates"), &mut files)?;
    collect_rust_files(&root.join("harness"), &mut files)?;

    for path in files {
        // `wslam-core::time` is where the wall clock is allowed to live; it is
        // the seam the rest of the rule is defined against.
        if path.ends_with("wslam-core/src/time.rs") {
            continue;
        }
        let text = std::fs::read_to_string(&path)?;

        // Everything below the crate's test module is test code, and test code
        // may use a stopwatch. Find where that starts.
        let test_module_line = text
            .lines()
            .position(|l| l.trim_start().starts_with("#[cfg(test)]"))
            .unwrap_or(usize::MAX);

        for (index, line) in text.lines().enumerate() {
            if index >= test_module_line {
                break;
            }
            let trimmed = line.trim_start();
            // Comments describe the rule constantly; they do not violate it.
            if trimmed.starts_with("//") || trimmed.starts_with('*') {
                continue;
            }
            for needle in CLOCK_CALLS {
                if line.contains(needle) {
                    violations.push(Violation {
                        file: path.clone(),
                        line: index + 1,
                        text: trimmed.to_string(),
                        rule: "no wall clock on the estimation path (spec.md §6)",
                    });
                }
            }
            for needle in UNSEEDED_RNG {
                if line.contains(needle) {
                    violations.push(Violation {
                        file: path.clone(),
                        line: index + 1,
                        text: trimmed.to_string(),
                        rule: "every RNG is seeded, RANSAC included (spec.md §6)",
                    });
                }
            }
        }
    }

    if violations.is_empty() {
        println!("\x1b[32minvariants hold\x1b[0m");
        println!("  no wall clock outside wslam-core::time and test code");
        println!("  no unseeded RNG anywhere");
        return Ok(violations);
    }

    for v in &violations {
        eprintln!(
            "\x1b[31mviolation\x1b[0m {}:{}\n  {}\n  rule: {}",
            v.file.display(),
            v.line,
            v.text,
            v.rule
        );
    }
    bail!("{} invariant violation(s)", violations.len())
}

fn collect_rust_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            if path.file_name().is_some_and(|n| n == "target") {
                continue;
            }
            collect_rust_files(&path, out)?;
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
    Ok(())
}

fn ci() -> Result<()> {
    let root = root();
    run("cargo", &["fmt", "--all", "--check"], &root)?;
    check_invariants()?;
    run(
        "cargo",
        &[
            "clippy",
            "--workspace",
            "--all-targets",
            "--all-features",
            "--",
            "-D",
            "warnings",
        ],
        &root,
    )?;
    run("cargo", &["test", "--workspace", "--all-features"], &root)?;
    // The wasm target must keep compiling even when nobody is looking at it;
    // discovering a native-only dependency at release time is expensive.
    run(
        "cargo",
        &[
            "check",
            "-p",
            "wslam-wasm",
            "--target",
            "wasm32-unknown-unknown",
        ],
        &root,
    )?;
    if have("pnpm") {
        run("pnpm", &["install", "--frozen-lockfile"], &root)?;
        run("pnpm", &["-r", "typecheck"], &root)?;
        run("pnpm", &["-r", "test"], &root)?;
    } else {
        println!("\x1b[33mskipping TypeScript checks: pnpm not found\x1b[0m");
    }
    println!("\n\x1b[32mall green\x1b[0m");
    Ok(())
}
