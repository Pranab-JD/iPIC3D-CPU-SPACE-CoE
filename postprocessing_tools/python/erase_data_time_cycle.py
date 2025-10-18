"""
Created on Sat Oct 18 10:48 2025

Description: Delete restart data for user-specified time cycles. 
             Supports both restart*.hdf and proc*.hdf sets.
             No other labels are allowed. This works with (user-defined) "N" MPI tasks.

Usage:
        To erase restart data for time cycles 1200, 1300, and 1500, try
        srun python3 ../postprocessing_tools/python/erase_restart.py /path/restart*.hdf 1200 1300 1500

        To erase restart data for time cycles 1200 to 1500 (including cycles 1200 and 1500), try
        srun python3 ../postprocessing_tools/python/erase_restart.py /path/restart*.hdf 1200-1500

        To erase output data for time cycles 1200 to 1500 (including cycles 1200 and 1500), try
        srun python3 ../postprocessing_tools/python/erase_restart.py /path/proc*.hdf 1000-1100 2000
"""

import argparse
import re, sys, os, subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set

from mpi4py import MPI
import h5py

CYCLE_RX = re.compile(r"(?:^|/)cycle_(\d+)$")
PROC_RX  = re.compile(r"^proc\d+\.hdf$", re.IGNORECASE)
RESTART_RX = re.compile(r"^restart.*\.hdf$", re.IGNORECASE)

def find_all_cycle_datasets(h5: h5py.File) -> List[Tuple[str, int]]:
    found: List[Tuple[str, int]] = []
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            m = CYCLE_RX.search(name)
            if m:
                found.append((f"/{name}" if not name.startswith("/") else name, int(m.group(1))))
    h5.visititems(visitor)
    return found

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

def parse_cycle_specs(specs: List[str]) -> Set[int]:
    cycles: Set[int] = set()
    for tok in specs:
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            try:
                start = int(a); end = int(b)
            except ValueError:
                raise ValueError(f"Invalid cycle range '{tok}'")
            if end < start:
                start, end = end, start
            cycles.update(range(start, end + 1))
        else:
            try:
                cycles.add(int(tok))
            except ValueError:
                raise ValueError(f"Invalid cycle '{tok}'")
    return cycles

def process_file(path: Path, target_cycles: Set[int], dry_run=False, prune=False, verbose=False) -> Dict[str, int]:
    stats = {"datasets_deleted": 0, "groups_pruned": 0, "files_processed": 0}
    try:
        with h5py.File(path, "r+") as h5:
            stats["files_processed"] = 1
            all_cycles = find_all_cycle_datasets(h5)
            if not all_cycles:
                if verbose:
                    print("  No cycle_* datasets found; nothing to do.", flush=True)
                return stats
            to_delete = [(p, c) for (p, c) in all_cycles if c in target_cycles]
            if verbose:
                if to_delete:
                    sel = sorted(set(c for _, c in to_delete))
                    print(f"  Will delete cycles: {sel}", flush=True)
                else:
                    print("  No matching cycles found in this file.", flush=True)
            to_delete.sort(key=lambda t: t[0].count("/"), reverse=True)
            for dspath, cyc in to_delete:
                parent_path = "/" if dspath.rfind("/") == 0 else dspath[: dspath.rfind("/")]
                name = dspath.split("/")[-1]
                if verbose or dry_run:
                    print(f"  delete {dspath} (cycle {cyc})", flush=True)
                if not dry_run:
                    try:
                        del h5[parent_path][name]
                        stats["datasets_deleted"] += 1
                    except KeyError:
                        pass
            if prune:
                if dry_run:
                    if verbose:
                        print("  (dry-run) would prune empty groups", flush=True)
                else:
                    stats["groups_pruned"] = prune_empty_groups(h5, verbose=verbose)
    except (OSError, IOError) as e:
        print(f"  ERROR opening {path}: {e}", file=sys.stderr, flush=True)
    return stats

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

def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if n < 1024 or unit == "PB":
            return f"{n:.2f} {unit}"
        n /= 1024

def _python_repack(src_path, dst_path, verbose=False):
    if verbose:
        print(f"    python repack (deep copy) {src_path} -> {dst_path}", flush=True)
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        for k, v in src.attrs.items():
            dst.attrs[k] = v
        for name in src.keys():
            src.copy(name, dst, name=name)

def repack_file(path: Path, deleted_count: int, verbose=False, dry_run=False):
    if deleted_count <= 0:
        if verbose:
            print(f"  repack skipped (no datasets deleted): {path}", flush=True)
        return
    tmp_out = path.with_suffix(path.suffix + ".repacked")
    if tmp_out.exists():
        try: tmp_out.unlink()
        except Exception as e: print(f"  WARNING: could not remove stale {tmp_out}: {e}", flush=True)
    if verbose or dry_run:
        print(f"  repacking {path} -> {tmp_out}", flush=True)
    if dry_run: return
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
            print(f"  ERROR: repack (all methods) failed for {path}: {e}", flush=True)
            try:
                if tmp_out.exists(): tmp_out.unlink()
            except Exception: pass
            return
    try:
        os.replace(tmp_out, path)
    except Exception as e:
        print(f"  ERROR: could not replace original with repacked file for {path}: {e}", flush=True)

def detect_file_label(file_paths: List[str]) -> str:
    """
    Determine the label to print based on filenames.
    - If ALL basenames match proc\\d+\\.hdf  -> 'data files'
    - If ALL basenames match restart.*\\.hdf -> 'restart files'
    - Otherwise: error (mixed or unknown)
    """
    basenames = [Path(p).name for p in file_paths]
    all_proc = all(PROC_RX.match(bn) for bn in basenames)
    all_restart = all(RESTART_RX.match(bn) for bn in basenames)
    if all_proc and not all_restart:
        return "data files"
    if all_restart and not all_proc:
        return "restart files"
    raise ValueError("Input must be exclusively proc*.hdf or restart*.hdf — no mixing and no other patterns.")

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    ap = argparse.ArgumentParser(
        description="Delete HDF5 datasets named cycle_<n> matching the user-specified cycles; MPI distributes files 1-per-rank."
    )
    ap.add_argument("files", nargs="+", help="HDF5 files (e.g., restart*.hdf or proc*.hdf) or @filelist.txt")
    ap.add_argument("cycles", nargs="+", help="Cycles to delete: integers and/or inclusive ranges like 1000-1100")
    ap.add_argument("--dry-run", action="store_true", help="Show actions without modifying files")
    ap.add_argument("--prune-empty", action="store_true", help="Remove now-empty groups")
    ap.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = ap.parse_args()

    if rank == 0:
        files = expand_file_args(args.files)
        if not files:
            print("No input files.", file=sys.stderr, flush=True)
        try:
            label = detect_file_label(files)  # 'data files' or 'restart files'
        except ValueError as e:
            print(str(e), file=sys.stderr, flush=True)
            files = []
            label = ""
        try:
            target_cycles = sorted(parse_cycle_specs(args.cycles))
        except ValueError as e:
            print(f"Invalid cycle specification: {e}", file=sys.stderr, flush=True)
            target_cycles = []
    else:
        files = None
        target_cycles = None
        label = None

    files = comm.bcast(files, root=0)
    target_cycles = comm.bcast(target_cycles, root=0)
    label = comm.bcast(label, root=0)

    if not files or not target_cycles:
        return

    my_files = files[rank::size]

    if args.verbose:
        print(f"[rank {rank}/{size}] assigned {len(my_files)} files; target cycles: {target_cycles}", flush=True)

    my_totals = {"files_assigned": len(my_files),
                 "files_processed": 0,
                 "datasets_deleted": 0,
                 "groups_pruned": 0,
                 "bytes_before": 0,
                 "bytes_after": 0,}

    for f in my_files:
        p = Path(f)
        if args.verbose:
            print(f"[rank {rank}] ==> {p}", flush=True)
        try:
            size_before = p.stat().st_size
        except FileNotFoundError:
            size_before = 0
        stats = process_file(p, set(target_cycles), dry_run=args.dry_run, prune=args.prune_empty, verbose=args.verbose)
        for k in ("files_processed", "datasets_deleted", "groups_pruned"):
            my_totals[k] += stats.get(k, 0)
        repack_file(p, deleted_count=stats.get("datasets_deleted", 0), verbose=args.verbose, dry_run=args.dry_run)
        try:
            size_after = p.stat().st_size
        except FileNotFoundError:
            size_after = 0
        my_totals["bytes_before"] += size_before
        my_totals["bytes_after"]  += size_after

    all_totals = comm.gather(my_totals, root=0)
    comm.Barrier()

    if rank == 0:
        files_assigned   = sum(t.get("files_assigned", 0)   for t in all_totals)
        files_processed  = sum(t.get("files_processed", 0)  for t in all_totals)
        datasets_deleted = sum(t.get("datasets_deleted", 0) for t in all_totals)
        bytes_before     = sum(t.get("bytes_before", 0)     for t in all_totals)
        bytes_after      = sum(t.get("bytes_after", 0)      for t in all_totals)
        bytes_saved      = max(0, bytes_before - bytes_after)
        pct_saved        = (100.0 * bytes_saved / bytes_before) if bytes_before > 0 else 0.0

        # IMPORTANT: Only the two allowed labels are printed
        print("\n============= SUMMARY =============", flush=True)
        print(f"Files assigned      : {files_assigned}", flush=True)
        print(f"Files processed     : {files_processed}", flush=True)
        print(f"Datasets deleted    : {datasets_deleted}", flush=True)
        print()
        print(f"Total size of all {label} before   : {_fmt_bytes(bytes_before)}", flush=True)
        print(f"Total size of all {label} after    : {_fmt_bytes(bytes_after)}", flush=True)
        print(f"Space saved                          : {_fmt_bytes(bytes_saved)} ({pct_saved:.2f}%)", flush=True)

        if files_assigned == files_processed:
            print("\n✅ All {0} processed successfully.".format(label), flush=True)
        else:
            print("\n⚠️ Some {0} may not have been processed, check logs.".format(label), flush=True)

if __name__ == "__main__":
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    main()
