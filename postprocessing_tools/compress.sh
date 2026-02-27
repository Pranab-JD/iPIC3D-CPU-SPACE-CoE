#!/bin/bash
#SBATCH --account=eu-25-67
#SBATCH --partition=qcpu
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --job-name=zstd_hdf5
#SBATCH --output=zstd_hdf5.out

IN_DIR="/mnt/proj3/eu-25-67/RMR/sigma_0.1/domain_17.5_35_17.5/RMR_S0.1_Bz2/"                # Directory consisting of the the proc#.hdf files
OUT_DIR="/mnt/proj3/eu-25-67/RMR/sigma_0.1/domain_17.5_35_17.5/RMR_S0.1_Bz2/"               # Directory where the compressed data is to be stored
LEVEL="${LEVEL:-22}"                                                                        # zstd compression level (1=fast and low; 22=slow and high)
REMOVE_ORIGINALS="${REMOVE_ORIGINALS:-1}"                                                   # 1 => delete input file after success
EXT="${EXT:-.hdf}"                                                                          # match e.g. .hdf5, .h5, etc.

ml HDF5/1.14.3-gompi-2023b zstd/1.5.7-GCCcore-14.3.0 

###! All the above parameters need to be set by the user depending on the cluster you wish to run on 

echo "========================================================================"
echo
date
echo

set -euo pipefail
export OMPI_MCA_btl=^openib

###! Extraction can be done via
# "zstd -d proc123.hdf.zst"
#
# OR, for mutiple files in a folder, try
#
# IN_DIR="compressed_dir"
# OUT_DIR="restored_dir"

# find "$IN_DIR" -type f -name "*.zst" | while read f; do
#     rel="${f#$IN_DIR/}"
#     out="$OUT_DIR/${rel%.zst}"
#     mkdir -p "$(dirname "$out")"
#     zstd -d "$f" -o "$out"
# done

###! Compression

# Use allocated tasks as worker count (fallback to 1)
JOBS="${SLURM_NTASKS:-1}"

# Checks
if [[ ! -d "$IN_DIR" ]]; then
  echo "ERROR: IN_DIR not found: $IN_DIR" >&2
  exit 1
fi
mkdir -p "$OUT_DIR"

command -v zstd >/dev/null 2>&1 || { echo "ERROR: zstd not found in PATH" >&2; exit 1; }

IN_DIR="${IN_DIR%/}"
OUT_DIR="${OUT_DIR%/}"

compress_one() {
  local in="$1"
  local rel="${in#$IN_DIR/}"
  local out="$OUT_DIR/${rel}.zst"
  mkdir -p "$(dirname "$out")"

  # skip if up-to-date
  if [[ -f "$out" && "$out" -nt "$in" ]]; then
    echo "SKIP: $rel"
    return 0
  fi

  local tmp_out="${out}.tmp.$$"
  if zstd "-$LEVEL" -q --no-progress "$in" -o "$tmp_out"; then
    mv -f "$tmp_out" "$out"
    echo "OK: $rel -> ${out#$OUT_DIR/}"
    if [[ "$REMOVE_ORIGINALS" -eq 1 ]]; then
      rm -f "$in"
    fi
  else
    rm -f "$tmp_out" || true
    echo "FAIL: $rel" >&2
    return 1
  fi
}

export -f compress_one
export IN_DIR OUT_DIR LEVEL REMOVE_ORIGINALS

echo "Starting compression..."
echo "IN_DIR=$IN_DIR"
echo "OUT_DIR=$OUT_DIR"
echo "EXT=*$EXT"
echo "LEVEL=$LEVEL"
echo "JOBS=$JOBS"
echo "REMOVE_ORIGINALS=$REMOVE_ORIGINALS"
echo

# Stream find -> xargs (no temp file, no /tmp issues)
find "$IN_DIR" -type f -name "*${EXT}" -print0 | \
  xargs -0 -n1 -P "$JOBS" -I{} bash -c 'compress_one "$@"' _ {}

echo
echo "Done."
echo
echo "========================================================================"
date
echo