#!/bin/sh
# Fetch the replay datasets. They are gitignored; this is how you get them.
#
# spec.md §6 Tier 2: "EuRoC and TUM-VI through the native build. This is the
# regression wall." Both publish reference ATE numbers, which is what makes
# them double as port validation.
#
#   sh datasets/fetch.sh [all|euroc|tum-vi|7scenes]
#
# Nothing here is redistributed. Each dataset carries its own licence and
# citation requirements; see the printed notices.

set -eu

WHICH="${1:-all}"
DIR="$(cd "$(dirname "$0")" && pwd)"

EUROC_BASE="http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset"
# Per-sequence copies on Hugging Face, tried first. The ETH host has been
# unreachable for months (connect timeout; ETH moved the files into their
# Research Collection, which 403s non-browser clients), so CI needs a host
# that answers. The repository is **private** — EuRoC's rights statement is
# "In Copyright – Non-Commercial Use Permitted", which does not grant
# redistribution, so the copies are not published. Reading it needs
# HF_TOKEN in the environment (a read-scoped token; in CI it comes from the
# repository secret of the same name). Without a token this host is skipped
# and the canonical one is tried, followed by the manual instructions.
EUROC_MIRROR="https://huggingface.co/datasets/binhpham/euroc-mav-sequences/resolve/main"
# The core five. MH_01/MH_03 are easy, V1_03/V2_03 are the hard ones where a
# tracker either survives or does not — regressions show up there first.
EUROC_SEQUENCES="machine_hall/MH_01_easy/MH_01_easy
machine_hall/MH_03_medium/MH_03_medium
vicon_room1/V1_01_easy/V1_01_easy
vicon_room1/V1_03_difficult/V1_03_difficult
vicon_room2/V2_03_difficult/V2_03_difficult"

# Canonical host per https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset
# (cdn3 also works, via a redirect).
TUMVI_BASE="https://vision.in.tum.de/tumvi/exported/euroc/512_16"
# Room sequences only. Per the dataset page, corridor / magistrale / outdoors /
# slides carry ground truth for the *start and end segments only*, which makes
# them useless for an ATE number — the harness would report most of the sequence
# as unmatched and the remainder as a suspiciously good score.
TUMVI_SEQUENCES="dataset-room1_512_16
dataset-room3_512_16
dataset-room5_512_16"

SEVENSCENES_BASE="http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8"
SEVENSCENES_SCENES="chess fire office"

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "error: $1 is required but not installed" >&2
    exit 1
  }
}

need curl
need unzip

# Returns 1 rather than exiting when the host is unreachable, so the caller can
# fall back to the manual instructions. Dataset hosts go down — ETH's was down
# when this script was written — and a fetch script that only knows how to die is
# useless on exactly the day you need it.
download() {
  url="$1"
  out="$2"
  if [ -f "$out" ]; then
    echo "  have $(basename "$out")"
    return 0
  fi
  # The Hugging Face host holds private copies and needs a bearer token.
  # The token is sent to huggingface.co and NOWHERE else — appending it
  # unconditionally would hand a credential to every dataset host in this
  # file. (The CDN redirect HF answers with carries its own signed URL, so
  # dropping the header across the redirect — curl's default — is correct.)
  set -- # reuse $@ for the auth argument, empty by default
  case "$url" in
    https://huggingface.co/*)
      if [ -z "${HF_TOKEN:-}" ]; then
        printf "  \033[33mskipping %s: HF_TOKEN is not set (private host)\033[0m\n" \
          "$(basename "$out")" >&2
        return 1
      fi
      set -- --header "Authorization: Bearer $HF_TOKEN"
      ;;
  esac
  echo "  fetching $(basename "$out")"
  # --continue-at resumes an interrupted multi-gigabyte fetch instead of starting
  # over. --connect-timeout fails fast on a dead host rather than hanging.
  if curl --fail --location --continue-at - --connect-timeout 20 --progress-bar \
          "$@" --output "$out.part" "$url"; then
    mv "$out.part" "$out"
    return 0
  fi
  rm -f "$out.part"
  printf "  \033[33mcould not fetch %s\033[0m\n" "$(basename "$out")" >&2
  return 1
}

# Unpack anything the user downloaded by hand into the target directory.
adopt_manual() {
  target="$1"
  found=0
  for archive in "$target"/*.zip; do
    [ -f "$archive" ] || continue
    name="$(basename "$archive" .zip)"
    [ -d "$target/$name" ] && continue
    echo "  unpacking hand-placed $name"
    unzip -q "$archive" -d "$target/$name"
    found=1
  done
  for archive in "$target"/*.tar; do
    [ -f "$archive" ] || continue
    echo "  unpacking hand-placed $(basename "$archive")"
    tar -xf "$archive" -C "$target"
    found=1
  done
  return $((1 - found))
}

fetch_euroc() {
  echo "EuRoC MAV (ETH Zurich, In Copyright / Non-Commercial Use Permitted)"
  echo "  Burri et al., IJRR 2016. Cite it if you publish numbers from it."
  target="$DIR/euroc"
  mkdir -p "$target"

  adopt_manual "$target" || true

  failed=0
  for seq in $EUROC_SEQUENCES; do
    name="$(basename "$seq")"
    [ -d "$target/$name" ] && { echo "  have $name"; continue; }
    # Mirror first: it answers. The canonical host second, so the day ETH
    # resurrects it the script keeps working even if the mirror goes away.
    if download "$EUROC_MIRROR/$name.zip" "$target/$name.zip" ||
      download "$EUROC_BASE/$seq.zip" "$target/$name.zip"; then
      echo "  unpacking $name"
      unzip -q "$target/$name.zip" -d "$target/$name"
      rm "$target/$name.zip"
    else
      failed=$((failed + 1))
    fi
  done

  if [ "$failed" -gt 0 ]; then
    cat >&2 <<MANUAL

  $failed EuRoC sequence(s) could not be fetched.
  The ETH ASL host is frequently unreachable. Download them by hand:

    https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets

  Take the "ASL Dataset Format" zip for each sequence, drop the .zip files into

    $target/

  and re-run this script — it unpacks whatever it finds there. Any subset works;
  the harness reports per sequence and never pools across them.
MANUAL
    return 1
  fi
}

fetch_tumvi() {
  echo "TUM VI (TU Munich, CC BY 4.0)"
  echo "  Schubert et al., IROS 2018."
  printf "  \033[33mNOTE: these sequences are fisheye (pinhole-equidistant /\n"
  printf "  Kannala-Brandt). This build implements Brown-Conrady only, so the\n"
  printf "  replay harness will REFUSE them rather than misread the coefficients.\n"
  printf "  Fetch them for when a fisheye model lands; use EuRoC today.\033[0m\n"
  target="$DIR/tum-vi"
  mkdir -p "$target"
  adopt_manual "$target" || true
  for seq in $TUMVI_SEQUENCES; do
    [ -d "$target/$seq" ] && { echo "  have $seq"; continue; }
    if download "$TUMVI_BASE/$seq.tar" "$target/$seq.tar"; then
      echo "  unpacking $seq"
      tar -xf "$target/$seq.tar" -C "$target"
      rm "$target/$seq.tar"
    else
      echo "  fetch by hand from https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset" >&2
      echo "  (the \"euroc / DSO 512x512\" column), drop the .tar into $target/ and re-run" >&2
    fi
  done
}

fetch_7scenes() {
  # spec.md §6 L4: "Also run 7-Scenes for a comparable public number."
  echo "7-Scenes (Microsoft, research use only)"
  echo "  Shotton et al., CVPR 2013. Check the licence before publishing."
  target="$DIR/7scenes"
  mkdir -p "$target"
  for scene in $SEVENSCENES_SCENES; do
    [ -d "$target/$scene" ] && { echo "  have $scene"; continue; }
    if download "$SEVENSCENES_BASE/$scene.zip" "$target/$scene.zip"; then
      echo "  unpacking $scene"
      unzip -q "$target/$scene.zip" -d "$target"
      rm "$target/$scene.zip"
    else
      echo "  fetch by hand from https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/" >&2
    fi
  done
}

case "$WHICH" in
  euroc) fetch_euroc ;;
  tum-vi) fetch_tumvi ;;
  7scenes) fetch_7scenes ;;
  all)
    # `|| true` so one dead host does not stop the others; each prints its own
    # manual fallback.
    fetch_euroc || true
    fetch_tumvi || true
    fetch_7scenes || true
    ;;
  *)
    echo "usage: $0 [all|euroc|tum-vi|7scenes]" >&2
    exit 2
    ;;
esac

echo
echo "done. Datasets are gitignored and must never be committed."
