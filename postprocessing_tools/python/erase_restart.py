"""
Created on Thu Aug 10:48 2025

@author: Pranab JD

Description: This code deletes restart data for all time cycles other than the last one
             using MPI so that each file is handled by exactly one rank. This reduces 
             the memory requirements of the restart files.

Usage: mpirun python3 ../postprocessing_tools/python/erase_restart.py data/restart*.hdf
"""

import argparse
import numpy as np
import re, sys, os, subprocess, shutil, tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from mpi4py import MPI
import h5py

CYCLE_RX = re.compile(r"(?:^|/)cycle_(\d+)$")

def read_last_cycle(h5: h5py.File) -> Optional[int]:
    """Prefer /last_cycle if it exists and is scalar; otherwise None."""
    if "last_cycle" in h5:
        ds = h5["last_cycle"]
        try:
            val = ds[()]
            if getattr(ds, "shape", ()) == ():
                return int(val)
            # 1-element dataset
            if getattr(ds, "shape", None) and ds.shape == (1,):
                return int(ds[0])
        except Exception:
            return None
    return None


def find_all_cycle_datasets(h5: h5py.File) -> List[Tuple[str, int]]:
    """Return list of (path, cycle_int) for datasets whose name ends with cycle_<n>."""
    found: List[Tuple[str, int]] = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            m = CYCLE_RX.search(name)
            if m:
                found.append((f"/{name}" if not name.startswith("/") else name, int(m.group(1))))

    h5.visititems(visitor)
    return found


def prune_empty_groups(h5: h5py.File, verbose=False) -> int:
    """Remove empty groups bottom-up. Returns number of groups removed."""
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
            pass  # might already be gone
    return removed

def process_file(path: Path, dry_run=False, prune=False, verbose=False) -> Dict[str, int]:
    """
    Process a single HDF5 file. Returns stats dict.
    """
    stats = {"datasets_deleted": 0, "groups_pruned": 0, "files_processed": 0}
    try:
        with h5py.File(path, "r+") as h5:
            stats["files_processed"] = 1

            last_cycle = read_last_cycle(h5)
            all_cycles = find_all_cycle_datasets(h5)

            if not all_cycles:
                if verbose:
                    print("  No cycle_* datasets found; nothing to do.", flush=True)
                return stats

            inferred_latest = max(c for _, c in all_cycles)
            latest = last_cycle if last_cycle is not None else inferred_latest
            if verbose:
                src = "/last_cycle" if last_cycle is not None else "inferred"
                print(f"  Latest cycle = {latest} ({src})", flush=True)

            to_delete = [(p, c) for (p, c) in all_cycles if c < latest]
            # Delete deepest paths first
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
    """
    Expand @list files and keep order stable. No globbing here; rely on shell or explicit lists.
    """
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
    # Make absolute, then sort for deterministic partition across ranks
    files = [str(Path(p).expanduser().resolve()) for p in files]
    files = sorted(dict.fromkeys(files))  # de-dup preserving order via dict
    return files

def _fmt_bytes(n: int) -> str:
    # human-readable bytes (no external deps)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if n < 1024 or unit == "PB":
            return f"{n:.2f} {unit}"
        n /= 1024

def _python_repack(src_path, dst_path, verbose=False):
    """Pure-Python repack using h5py deep copy (no options, just a clean copy)."""
    if verbose:
        print(f"    python repack (deep copy) {src_path} -> {dst_path}", flush=True)
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        # Copy file-level attributes
        for k, v in src.attrs.items():
            dst.attrs[k] = v
        # Copy all root members recursively
        for name in src.keys():
            src.copy(name, dst, name=name)  # recursive by default

def repack_file(path: Path, deleted_count: int, verbose=False, dry_run=False):
    """
    Compact an HDF5 file after deletions.
    Tries h5repack (new syntax), then h5repack (-i/-o), then pure-Python deep copy.
    """
    if deleted_count <= 0:
        if verbose:
            print(f"  repack skipped (no datasets deleted): {path}", flush=True)
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

    # 1) h5repack: new syntax
    cmd1 = ["h5repack", "-v", str(path), str(tmp_out)]
    res = run(cmd1)
    if res.returncode != 0:
        if verbose:
            print(f"    h5repack (file1 file2) failed: {res.stderr.strip()}", flush=True)
        # 2) fallback: legacy -i/-o
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
                if tmp_out.exists():
                    tmp_out.unlink()
            except Exception:
                pass
            return

    # Replace original with compacted copy
    try:
        os.replace(tmp_out, path)
    except Exception as e:
        print(f"  ERROR: could not replace original with repacked file for {path}: {e}", flush=True)
        # Leave .repacked for manual inspection

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    ap = argparse.ArgumentParser(description="Delete HDF5 datasets named cycle_<n> older than the latest cycle; MPI distributes files 1-per-rank.")
    ap.add_argument("files", nargs="+", help="HDF5 files (e.g., restart*.hdf) or @filelist.txt")
    ap.add_argument("--dry-run", action="store_true", help="Show actions without modifying files")
    ap.add_argument("--prune-empty", action="store_true", help="Remove now-empty groups")
    ap.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = ap.parse_args()

    # Rank 0 prepares the canonical file list
    if rank == 0:
        files = expand_file_args(args.files)
        if not files:
            print("No input files.", file=sys.stderr, flush=True)
    else:
        files = None

    files = comm.bcast(files, root=0)

    # Partition: each rank takes files[rank::size]
    my_files = files[rank::size]

    if args.verbose:
        print(f"[rank {rank}/{size}] assigned {len(my_files)} files", flush=True)

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

        # Size before
        try:
            size_before = p.stat().st_size
        except FileNotFoundError:
            size_before = 0

        stats = process_file(p, dry_run=args.dry_run, prune=args.prune_empty, verbose=args.verbose)
        for k in ("files_processed", "datasets_deleted", "groups_pruned"):
            my_totals[k] += stats.get(k, 0)

        repack_file(p, deleted_count=stats.get("datasets_deleted", 0), verbose=args.verbose, dry_run=args.dry_run)

        # Size after (post-cleanup/repack or as-is in dry-run)
        try:
            size_after = p.stat().st_size
        except FileNotFoundError:
            size_after = 0

        my_totals["bytes_before"] += size_before
        my_totals["bytes_after"]  += size_after

    # Gather per-rank dicts to rank 0 and summarize there
    all_totals = comm.gather(my_totals, root=0)
    comm.Barrier()

    if rank == 0:
        # Sum defensively (works even if some ranks had zero work)
        files_assigned   = sum(t.get("files_assigned", 0)   for t in all_totals)
        files_processed  = sum(t.get("files_processed", 0)  for t in all_totals)
        datasets_deleted = sum(t.get("datasets_deleted", 0) for t in all_totals)
        bytes_before     = sum(t.get("bytes_before", 0)     for t in all_totals)
        bytes_after      = sum(t.get("bytes_after", 0)      for t in all_totals)
        bytes_saved      = max(0, bytes_before - bytes_after)
        pct_saved        = (100.0 * bytes_saved / bytes_before) if bytes_before > 0 else 0.0

        mode = "DRY-RUN" if args.dry_run else "APPLY"
        print("\n============= SUMMARY =============", flush=True)

        print(f"Files assigned      : {files_assigned}", flush=True)
        print(f"Files processed     : {files_processed}", flush=True)
        print(f"Datasets deleted    : {datasets_deleted}", flush=True)
        print()
        print(f"Total size of all restart files before   : {_fmt_bytes(bytes_before)}", flush=True)
        print(f"Total size of all restart files after    : {_fmt_bytes(bytes_after)}", flush=True)
        print(f"Space saved                              : {_fmt_bytes(bytes_saved)} ({pct_saved:.2f}%)", flush=True)



        if files_assigned == files_processed:
            print("\n✅ All restart files processed successfully.", flush=True)
        else:
            print("\n⚠️ Some files may not have been processed, check logs.", flush=True)


if __name__ == "__main__":
    # Optional: help avoid file locking issues on some shared filesystems
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    main()
