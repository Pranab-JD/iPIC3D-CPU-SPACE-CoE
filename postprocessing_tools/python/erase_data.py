#!/usr/bin/env python3
"""
Delete user-specified HDF5 dataset subtrees/quantities from iPIC3D HDF output.
Supports both proc*.hdf and restart*.hdf sets (no mixing).

Examples
--------
# Delete Ex for selected cycles in all proc*.hdf:
srun -n 64 python3 erase_quantities.py /path/proc*.hdf \
  --targets fields/Ex --cycles 0 500 1000-2000 --prune-empty

# Delete Bz completely (all cycles) in restart*.hdf:
srun -n 32 python3 erase_quantities.py /path/restart*.hdf \
  --targets fields/Bz --all-cycles --prune-empty

# Delete Jx for species_2 at cycles 11500 and 12000:
srun -n 64 python3 erase_quantities.py /path/proc*.hdf \
  --targets moments/species_2/Jx --cycles 11500 12000

# Delete one exact dataset path:
srun -n 16 python3 erase_quantities.py /path/proc*.hdf \
  --targets moments/species_1/rho/cycle_15000
"""

import argparse
import re, sys, os, subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional
from mpi4py import MPI
import h5py

PROC_RX    = re.compile(r"^proc\d+\.hdf$", re.IGNORECASE)
RESTART_RX = re.compile(r"^restart.*\.hdf$", re.IGNORECASE)
CYCLE_RX   = re.compile(r"(?:^|/)cycle_(\d+)$")  # dataset/group tail

# ----------------------------- File utils -----------------------------
def expand_file_args(raw_args: List[str]) -> List[str]:
    files: List[str] = []
    for token in raw_args:
        if token.startswith("@"):
            with open(token[1:], "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        files.append(line)
        else:
            files.append(token)
    files = [str(Path(p).expanduser().resolve()) for p in files]
    files = sorted(dict.fromkeys(files))
    return files

def detect_file_label(file_paths: List[str]) -> str:
    basenames = [Path(p).name for p in file_paths]

    is_proc = [bool(PROC_RX.match(bn)) for bn in basenames]
    is_restart = [bool(RESTART_RX.match(bn)) for bn in basenames]

    # Check for unknown patterns
    unknown = [bn for bn, p, r in zip(basenames, is_proc, is_restart) if not (p or r)]
    if unknown:
        raise ValueError(
            f"Unknown file patterns detected: {unknown}. "
            "Only proc*.hdf and restart*.hdf are allowed."
        )

    if all(is_proc):
        return "data files"
    if all(is_restart):
        return "restart files"

    # Mixed case (both present)
    return "data + restart files"

def _fmt_bytes(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if x < 1024 or unit == "PB":
            return f"{x:.2f} {unit}"
        x /= 1024.0
    return f"{x:.2f} PB"

# ----------------------------- Cycle parsing -----------------------------
def parse_cycle_specs(specs: List[str]) -> Set[int]:
    cycles: Set[int] = set()
    for tok in specs:
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            start = int(a); end = int(b)
            if end < start:
                start, end = end, start
            cycles.update(range(start, end + 1))
        else:
            cycles.add(int(tok))
    return cycles

# ----------------------------- HDF deletion logic -----------------------------
def norm_path(p: str) -> str:
    p = p.strip()
    if not p:
        return p
    if not p.startswith("/"):
        p = "/" + p
    # remove trailing slash except root
    if len(p) > 1 and p.endswith("/"):
        p = p[:-1]
    return p

def is_cycle_node(name: str) -> Optional[int]:
    m = CYCLE_RX.search(name)
    return int(m.group(1)) if m else None

def delete_node(h5: h5py.File, path: str, dry_run=False, verbose=False) -> bool:
    """
    Delete a dataset OR group at exact `path`. Returns True if deletion attempted.
    """
    path = norm_path(path)
    if path == "/" or path == "":
        return False
    if path not in h5:
        return False

    parent_path = "/" if path.rfind("/") == 0 else path[: path.rfind("/")]
    key = path.split("/")[-1]

    if verbose or dry_run:
        print(f"  delete {path}", flush=True)
    if dry_run:
        return True

    try:
        del h5[parent_path][key]
        return True
    except KeyError:
        return False

def list_cycle_children(h5: h5py.File, base: str) -> List[Tuple[str, int]]:
    """
    Under `base` (a group), find children that match cycle_<n> and return (path, n).
    Searches only one level below base for speed.
    """
    base = norm_path(base)
    if base not in h5:
        return []
    obj = h5[base]
    if not isinstance(obj, h5py.Group):
        return []
    out: List[Tuple[str, int]] = []
    for k in obj.keys():
        cyc = is_cycle_node(k)
        if cyc is not None:
            out.append((f"{base}/{k}", cyc))
    return out

def find_matching_deletions(h5: h5py.File,
                            targets: List[str],
                            cycles: Optional[Set[int]],
                            all_cycles: bool) -> List[str]:
    """
    Convert user targets into concrete HDF paths to delete.
    - If target ends with /cycle_<n>: delete that exact dataset/group.
    - Else:
        - if all_cycles: delete all cycle_* children under target group.
        - elif cycles given: delete only matching cycle_* children.
        - else: delete the entire target subtree (dangerous but explicit).
    """
    deletions: List[str] = []
    for t in targets:
        t = norm_path(t)

        # Exact cycle path? delete exactly that
        if is_cycle_node(t) is not None:
            deletions.append(t)
            continue

        # If target exists and is group, operate on cycle_* children if requested
        if all_cycles or (cycles is not None):
            for p, c in list_cycle_children(h5, t):
                if all_cycles or (c in cycles):
                    deletions.append(p)
        else:
            # No cycle filter provided: delete whole subtree at target
            deletions.append(t)

    # Deduplicate & delete deeper first (avoid parent deletion before children)
    deletions = sorted(set(deletions), key=lambda p: p.count("/"), reverse=True)
    return deletions

def prune_empty_groups(h5: h5py.File, verbose=False) -> int:
    groups: List[str] = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Group):
            groups.append("/" + name if not name.startswith("/") else name)

    h5.visititems(visitor)
    groups.sort(key=lambda p: p.count("/"), reverse=True)

    removed = 0
    for gpath in groups:
        if gpath == "/":
            continue
        try:
            grp = h5[gpath]
            if len(grp.keys()) == 0:
                parent_path = "/" if gpath.rfind("/") == 0 else gpath[: gpath.rfind("/")]
                key = gpath.split("/")[-1]
                del h5[parent_path][key]
                removed += 1
                if verbose:
                    print(f"  pruned empty group: {gpath}", flush=True)
        except KeyError:
            pass
    return removed

# ----------------------------- Repack -----------------------------
def _python_repack(src_path, dst_path, verbose=False):
    if verbose:
        print(f"    python repack (deep copy) {src_path} -> {dst_path}", flush=True)
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        for k, v in src.attrs.items():
            dst.attrs[k] = v
        for name in src.keys():
            src.copy(name, dst, name=name)

def repack_file(path: Path, did_delete: bool, verbose=False, dry_run=False):
    if not did_delete:
        if verbose:
            print(f"  repack skipped (no deletions): {path}", flush=True)
        return

    tmp_out = path.with_suffix(path.suffix + ".repacked")
    if tmp_out.exists():
        try:
            tmp_out.unlink()
        except Exception as e:
            print(f"  WARNING: could not remove stale {tmp_out}: {e}", flush=True)

    if verbose or dry_run:
        print(f"  repacking {path} -> {tmp_out}", flush=True)
    if dry_run:
        return

    def run(cmd):
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    cmd1 = ["h5repack", "-v", str(path), str(tmp_out)]
    res = run(cmd1)
    if res.returncode != 0:
        if verbose:
            print(f"    h5repack (file1 file2) failed: {res.stderr.strip()}", flush=True)
        cmd2 = ["h5repack", "-v", "-i", str(path), "-o", str(tmp_out)]
        res = run(cmd2)

    if res.returncode != 0:
        if verbose:
            print(f"    h5repack failed again: {res.stderr.strip()}", flush=True)
            print("    falling back to pure-Python deep copy...", flush=True)
        try:
            _python_repack(str(path), str(tmp_out), verbose=verbose)
        except Exception as e:
            print(f"  ERROR: repack failed for {path}: {e}", flush=True)
            try:
                if tmp_out.exists():
                    tmp_out.unlink()
            except Exception:
                pass
            return

    try:
        os.replace(tmp_out, path)
    except Exception as e:
        print(f"  ERROR: could not replace original with repacked file for {path}: {e}", flush=True)

# ----------------------------- Per-file processing -----------------------------
def process_file(path: Path,
                 targets: List[str],
                 cycles: Optional[Set[int]],
                 all_cycles: bool,
                 dry_run=False,
                 prune=False,
                 verbose=False) -> Dict[str, int]:
    stats = {"files_processed": 0, "nodes_deleted": 0, "groups_pruned": 0}

    try:
        with h5py.File(path, "r+") as h5:
            stats["files_processed"] = 1

            deletions = find_matching_deletions(h5, targets, cycles, all_cycles)
            if verbose:
                print(f"  matched {len(deletions)} deletions", flush=True)

            did_delete = False
            for p in deletions:
                if delete_node(h5, p, dry_run=dry_run, verbose=verbose):
                    did_delete = True
                    stats["nodes_deleted"] += 1 if not dry_run else 0

            if prune:
                if dry_run:
                    if verbose:
                        print("  (dry-run) would prune empty groups", flush=True)
                else:
                    stats["groups_pruned"] = prune_empty_groups(h5, verbose=verbose)

    except (OSError, IOError) as e:
        print(f"  ERROR opening {path}: {e}", file=sys.stderr, flush=True)

    return stats

# ----------------------------- Main -----------------------------
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    ap = argparse.ArgumentParser(
        description="Delete user-selected quantities/paths from iPIC3D HDF5 (proc*.hdf or restart*.hdf)."
    )
    ap.add_argument("files", nargs="+", help="HDF5 files (proc*.hdf or restart*.hdf) or @filelist.txt")

    ap.add_argument("--targets", nargs="+", required=True,
                    help=("Target paths (relative or absolute). Examples: "
                          "fields/Ex   fields/Bz/cycle_0   moments/species_2/Jx   moments/species_1/rho/cycle_15000"))

    ap.add_argument("--cycles", nargs="*", default=None,
                    help="Optional cycle filter (integers and/or inclusive ranges like 1000-1100). Only used if targets are groups.")

    ap.add_argument("--all-cycles", action="store_true",
                    help="If a target is a group (e.g. fields/Ex), delete ALL cycle_* children under it.")

    ap.add_argument("--dry-run", action="store_true", help="Show actions without modifying files")
    ap.add_argument("--prune-empty", action="store_true", help="Remove now-empty groups")
    ap.add_argument("--repack", action="store_true", help="Repack files after deletions (shrinks file size)")
    ap.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    ap.add_argument("--progress-percent", type=float, default=10.0, help="Print global progress every X percent (0 disables)")

    args = ap.parse_args()

    if rank == 0:
        files = expand_file_args(args.files)
        if not files:
            print("No input files.", file=sys.stderr, flush=True)
        try:
            label = detect_file_label(files)
        except ValueError as e:
            print(str(e), file=sys.stderr, flush=True)
            files = []
            label = ""

        cycles = None
        if args.cycles is not None and len(args.cycles) > 0:
            cycles = parse_cycle_specs(args.cycles)

        # normalize targets once
        targets = [norm_path(t) for t in args.targets]

        # safety: require either --all-cycles, --cycles, or explicit cycle_* path for group targets
        if cycles is None and (not args.all_cycles):
            # if any target is not an explicit cycle_* path, we are deleting whole subtrees
            has_non_cycle = any(is_cycle_node(t) is None for t in targets)
            if has_non_cycle and not args.dry_run:
                print(
                    "ERROR: You provided group targets without --cycles or --all-cycles.\n"
                    "This would delete entire subtrees. Re-run with --dry-run to inspect, or add --cycles / --all-cycles.",
                    file=sys.stderr, flush=True
                )
                files = []
    else:
        files = None
        label = None
        cycles = None
        targets = None

    files = comm.bcast(files, root=0)
    label = comm.bcast(label, root=0)
    cycles = comm.bcast(cycles, root=0)
    targets = comm.bcast(targets, root=0)

    if not files:
        return

    my_files = files[rank::size]

    if args.verbose:
        print(f"[rank {rank}/{size}] assigned {len(my_files)} files", flush=True)

    my_tot = {"files_assigned": len(my_files),
              "files_processed": 0,
              "nodes_deleted": 0,
              "groups_pruned": 0,
              "bytes_before": 0,
              "bytes_after": 0}

    next_pct = args.progress_percent if args.progress_percent > 0 else None
    processed_local = 0
    
    for idx, f in enumerate(my_files, start=1):
        p = Path(f)
        if args.verbose:
            print(f"[rank {rank}] ==> {p}", flush=True)

        try:
            b0 = p.stat().st_size
        except FileNotFoundError:
            b0 = 0

        stats = process_file(p, targets, cycles, args.all_cycles,
                             dry_run=args.dry_run, prune=args.prune_empty, verbose=args.verbose)

        my_tot["files_processed"] += stats["files_processed"]
        my_tot["nodes_deleted"]   += stats["nodes_deleted"]
        my_tot["groups_pruned"]   += stats["groups_pruned"]

        did_delete = (stats["nodes_deleted"] > 0) and (not args.dry_run)
        if args.repack:
            repack_file(p, did_delete=did_delete, verbose=args.verbose, dry_run=args.dry_run)

        try:
            b1 = p.stat().st_size
        except FileNotFoundError:
            b1 = 0

        my_tot["bytes_before"] += b0
        my_tot["bytes_after"]  += b1

        processed_local += 1

        if args.progress_percent > 0:
            global_done = comm.allreduce(processed_local, op=MPI.SUM)
            global_total = len(files)
            pct = 100.0 * global_done / global_total if global_total else 0.0

            if rank == 0:
                while next_pct is not None and pct >= next_pct - 1e-12:
                    print(f"[global] progress: {next_pct:.1f}% ({global_done}/{global_total} files)", flush=True)
                    next_pct += args.progress_percent
                    if next_pct > 100.0:
                        next_pct = None
                        break

    all_tot = comm.gather(my_tot, root=0)
    comm.Barrier()

    if rank == 0:
        files_assigned   = sum(t["files_assigned"] for t in all_tot)
        files_processed  = sum(t["files_processed"] for t in all_tot)
        nodes_deleted    = sum(t["nodes_deleted"] for t in all_tot)
        groups_pruned    = sum(t["groups_pruned"] for t in all_tot)
        bytes_before     = sum(t["bytes_before"] for t in all_tot)
        bytes_after      = sum(t["bytes_after"] for t in all_tot)
        bytes_saved      = max(0, bytes_before - bytes_after)
        pct_saved        = (100.0 * bytes_saved / bytes_before) if bytes_before > 0 else 0.0

        print("\n============= SUMMARY =============", flush=True)
        print(f"Files assigned      : {files_assigned}", flush=True)
        print(f"Files processed     : {files_processed}", flush=True)
        print(f"Nodes deleted       : {nodes_deleted}", flush=True)
        if args.prune_empty:
            print(f"Groups pruned       : {groups_pruned}", flush=True)
        print()
        print(f"Total size of all {label} before   : {_fmt_bytes(bytes_before)}", flush=True)
        print(f"Total size of all {label} after    : {_fmt_bytes(bytes_after)}", flush=True)
        if args.repack and not args.dry_run:
            print(f"Space saved                          : {_fmt_bytes(bytes_saved)} ({pct_saved:.2f}%)", flush=True)

        if files_assigned == files_processed:
            print(f"\n✅ All {label} processed successfully.", flush=True)
        else:
            print(f"\n⚠️ Some {label} may not have been processed, check logs.", flush=True)

if __name__ == "__main__":
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    main()
