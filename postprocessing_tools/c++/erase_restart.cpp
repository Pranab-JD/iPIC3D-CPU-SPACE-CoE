//
// Description: Delete restart data (datasets named '.../cycle_<n>') older than the
//              latest cycle (prefers scalar /last_cycle if present; otherwise infers
//              from the maximum found). Works with user-defined N MPI tasks, assigning
//              one file per rank. Optionally prunes empty groups and repacks files.
//
// Usage:
//          srun -n N ./erase_restart data/restart*.hdf 
//                  [--dry-run] [--prune-empty] [-v|--verbose]
//
// Notes:
//      - Filenames are expected to be expanded by the shell. You may also pass @list files
//          containing one path per line (comments starting with '#').
//      - Repacking tries the external tool `h5repack`; if unavailable, falls back to a
//          deep-copy repack implemented here.
//
// Compile and run (from the build folder):
//          mpicxx -std=c++17 ../postprocessing_tools/c++/erase_restart.cpp -o erase_restart -lhdf5 -lhdf5_hl
//          srun ./erase_restart /path/restart*.hdf 
//
// ----------------------------------------------------------------------
// erase_restart.cpp (portable, no H5Lvisit3)
// Created: Sat Oct 18 10:48 2025

#include <mpi.h>
#include <hdf5.h>

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <regex>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ===== CLI =====
struct Args {
    std::vector<std::string> files;
    bool dry_run = false, prune_empty = false, verbose = false;
};

static void die_usage(int rank, const char* msg=nullptr){
    if(rank==0){
        if(msg) std::cerr << msg << "\n";
        std::cerr <<
        "Usage:\n"
        "  srun -n N ./erase_restart <files...> [--dry-run] [--prune-empty] [-v|--verbose]\n"
        "  (supports @filelist.txt entries)\n";
    }
    MPI_Abort(MPI_COMM_WORLD, 2);
    std::exit(2);
}

static std::vector<std::string> expand_file_args(const std::vector<std::string>& raw){
    std::vector<std::string> out;
    for(const auto& tok: raw){
        if(!tok.empty() && tok[0]=='@'){
            std::ifstream f(tok.substr(1));
            std::string line;
            while(std::getline(f,line)){
                auto s = line;
                s.erase(0, s.find_first_not_of(" \t\r\n"));
                if(s.empty() || s[0]=='#') continue;
                s.erase(s.find_last_not_of(" \t\r\n")+1);
                if(!s.empty()) out.push_back(s);
            }
        } else out.push_back(tok);
    }
    std::vector<std::string> abs; abs.reserve(out.size());
    for(auto& p: out){
        try { abs.push_back(fs::weakly_canonical(fs::path(p)).string()); }
        catch(...) { abs.push_back(fs::path(p).lexically_normal().string()); }
    }
    std::set<std::string> seen;
    std::vector<std::string> uniq;
    for(auto& p: abs){ if(seen.insert(p).second) uniq.push_back(p); }
    std::sort(uniq.begin(), uniq.end());
    return uniq;
}

static Args parse_args(int argc, char** argv, int rank){
    if(argc<2) die_usage(rank);
    Args a;
    for(int i=1;i<argc;++i){
        std::string s(argv[i]);
        if(s=="--dry-run") a.dry_run = true;
        else if(s=="--prune-empty") a.prune_empty = true;
        else if(s=="-v"||s=="--verbose") a.verbose = true;
        else if(!s.empty() && s[0]=='-') die_usage(rank, ("Unknown option: "+s).c_str());
        else a.files.push_back(s);
    }
    if(a.files.empty()) die_usage(rank, "No input files provided.");
    a.files = expand_file_args(a.files);
    if(a.files.empty()) die_usage(rank, "No input files after expansion.");
    return a;
}

static std::string fmt_bytes(uint64_t n){
    const char* u[]={"B","KB","MB","GB","TB","PB"};
    int i=0; double v=double(n);
    while(v>=1024.0 && i<5){ v/=1024.0; ++i; }
    char buf[64]; 
    std::snprintf(buf,sizeof(buf),"%.2f %s",v,u[i]); 
    return buf;
}

// ===== HDF5 helpers =====
static std::regex CYCLE_RX("(?:^|/)cycle_(\\d+)$");

static bool h5_exists(hid_t loc, const char* path){
    htri_t e = H5Lexists(loc, path, H5P_DEFAULT);
    return e>0;
}

static std::optional<long long> read_last_cycle(hid_t file, bool verbose=false){
    if(!h5_exists(file, "/last_cycle")) return std::nullopt;
    hid_t ds = H5Dopen2(file, "/last_cycle", H5P_DEFAULT);
    if(ds<0) return std::nullopt;
    hid_t space = H5Dget_space(ds);
    if(space<0){ H5Dclose(ds); return std::nullopt; }
    int ndims = H5Sget_simple_extent_ndims(space);
    hsize_t dims[1]={0};
    if(ndims==1) H5Sget_simple_extent_dims(space,dims,nullptr);
    bool scalar_like = (H5Sget_simple_extent_type(space)==H5S_SCALAR) || (ndims==1 && dims[0]==1);
    long long val=-1; herr_t st=-1;
    if(scalar_like){
        hid_t dt = H5Dget_type(ds);
        if(H5Tequal(dt,H5T_NATIVE_LLONG)>0) st = H5Dread(ds,H5T_NATIVE_LLONG,H5S_ALL,H5S_ALL,H5P_DEFAULT,&val);
        else { int tmp=-1; st = H5Dread(ds,H5T_NATIVE_INT,H5S_ALL,H5S_ALL,H5P_DEFAULT,&tmp); if(st>=0) val=tmp; }
        H5Tclose(dt);
    }
    H5Sclose(space); H5Dclose(ds);
    if(st<0) return std::nullopt;
    if(verbose) std::cerr << "  /last_cycle = " << val << "\n";
    return val;
}

struct PathCycle{ std::string path; long long cycle; };

static void collect_groups_and_datasets(hid_t grp, const std::string& abs,
                                        std::vector<std::string>& groups,
                                        std::vector<PathCycle>& datasets)
{
    H5G_info_t gi; if(H5Gget_info(grp,&gi)<0) return;
    for(hsize_t i=0;i<gi.nlinks;++i){
        // get child name
        ssize_t len = H5Lget_name_by_idx(grp, ".", H5_INDEX_NAME, H5_ITER_NATIVE, i, nullptr, 0, H5P_DEFAULT);
        if(len<0) continue;
        std::string name; name.resize(size_t(len)+1);
        H5Lget_name_by_idx(grp, ".", H5_INDEX_NAME, H5_ITER_NATIVE, i, name.data(), name.size(), H5P_DEFAULT);
        name.resize(size_t(len));
        std::string child_abs = (abs=="/") ? ("/"+name) : (abs+"/"+name);

        // object type
        H5O_info_t oi;
#if H5_VERSION_GE(1,12,0)
        if(H5Oget_info_by_name3(grp, name.c_str(), &oi, H5O_INFO_BASIC, H5P_DEFAULT)<0) continue;
#else
        if(H5Oget_info_by_name(grp, name.c_str(), &oi, H5P_DEFAULT)<0) continue;
#endif
        if(oi.type == H5O_TYPE_GROUP){
            groups.push_back(child_abs);
            hid_t sub = H5Gopen2(grp, name.c_str(), H5P_DEFAULT);
            if(sub>=0){ collect_groups_and_datasets(sub, child_abs, groups, datasets); H5Gclose(sub); }
        } else if(oi.type == H5O_TYPE_DATASET){
            std::smatch m;
            if(std::regex_search(child_abs, m, CYCLE_RX)){
                long long cyc = std::stoll(m[1].str());
                datasets.push_back({child_abs, cyc});
            }
        }
    }
}

static bool delete_link(hid_t file, const std::string& abs_path){
    if(abs_path=="/" || abs_path.empty()) return false;
    auto pos = abs_path.rfind('/');
    std::string parent = (pos==0) ? "/" : abs_path.substr(0,pos);
    std::string name = abs_path.substr(pos+1);
    hid_t grp = H5Gopen2(file, parent.c_str(), H5P_DEFAULT);
    if(grp<0) return false;
    herr_t st = H5Ldelete(grp, name.c_str(), H5P_DEFAULT);
    H5Gclose(grp);
    return st>=0;
}

static int prune_empty_groups(hid_t file, bool verbose=false){
    // collect all groups
    std::vector<std::string> groups; std::vector<PathCycle> dummy;
    hid_t root = H5Gopen2(file, "/", H5P_DEFAULT);
    if(root<0) return 0;
    collect_groups_and_datasets(root, "/", groups, dummy);
    H5Gclose(root);

    // deepest-first
    std::sort(groups.begin(), groups.end(),
        [](const std::string& a, const std::string& b){
            auto da = std::count(a.begin(),a.end(),'/');
            auto db = std::count(b.begin(),b.end(),'/');
            if(da!=db) return da>db;
            return a>b;
        });

    int removed=0;
    for(const auto& gpath: groups){
        if(gpath=="/") continue;
        hid_t grp = H5Gopen2(file, gpath.c_str(), H5P_DEFAULT);
        if(grp<0) continue;
        H5G_info_t gi;
        bool empty=false;
        if(H5Gget_info(grp,&gi)>=0) empty = (gi.nlinks==0);
        H5Gclose(grp);
        if(empty){
            auto pos = gpath.rfind('/');
            std::string parent = (pos==0) ? "/" : gpath.substr(0,pos);
            std::string name = gpath.substr(pos+1);
            hid_t par = H5Gopen2(file, parent.c_str(), H5P_DEFAULT);
            if(par>=0){
                if(H5Ldelete(par, name.c_str(), H5P_DEFAULT)>=0){
                    ++removed;
                    if(verbose) std::cerr << "  pruned empty group: " << gpath << "\n";
                }
                H5Gclose(par);
            }
        }
    }
    return removed;
}

// ===== repack =====
// Replace your run_cmd with this version (adds optional silencing)
// static int run_cmd(const std::vector<std::string>& cmd, bool silent) {
//     std::string s;
//     for (size_t i = 0; i < cmd.size(); ++i) {
//         if (i) s += ' ';
//         s += cmd[i];
//     }
//     if (silent) s += " >/dev/null 2>&1";  // swallow stdout+stderr
//     int rc = std::system(s.c_str());
//     if (WIFEXITED(rc)) return WEXITSTATUS(rc);
//     return rc;
// }

// // Replace your repack_file with this version
// static void repack_file(const fs::path& p, int deleted_count, bool verbose, bool dry_run) {
//     if (deleted_count <= 0) {
//         if (verbose) std::cerr << "  repack skipped (no datasets deleted): " << p << "\n";
//         return;
//     }
//     fs::path tmp = p; tmp += ".repacked";
//     std::error_code ec; if (fs::exists(tmp, ec)) fs::remove(tmp, ec);

//     if (verbose || dry_run) std::cerr << "  repacking " << p << " -> " << tmp << "\n";
//     if (dry_run) return;

//     // Only add -v to h5repack if our program is verbose. Also silence output if not verbose.
//     std::vector<std::string> cmd1 = verbose
//         ? std::vector<std::string>{"h5repack", "-v", p.string(), tmp.string()}
//         : std::vector<std::string>{"h5repack", p.string(), tmp.string()};

//     int rc = run_cmd(cmd1, /*silent=*/!verbose);
//     if (rc != 0) {
//         if (verbose) std::cerr << "    h5repack(file1 file2) failed, trying -i/-o\n";
//         std::vector<std::string> cmd2 = verbose
//             ? std::vector<std::string>{"h5repack", "-v", "-i", p.string(), "-o", tmp.string()}
//             : std::vector<std::string>{"h5repack", "-i", p.string(), "-o", tmp.string()};
//         rc = run_cmd(cmd2, /*silent=*/!verbose);
//     }

//     if (rc != 0) {
//         if (verbose) std::cerr << "    h5repack failed again; deep copy...\n";
//         if (!deep_copy_file(p, tmp, verbose)) {
//             if (verbose) std::cerr << "  ERROR: repack failed for " << p << "\n";
//             return;
//         }
//     }

//     fs::rename(tmp, p, ec);
//     if (ec && verbose) {
//         std::cerr << "  ERROR: replacing with repacked file failed for " << p
//                   << " : " << ec.message() << "\n";
//     }
// }

static bool deep_copy_file(const fs::path& src, const fs::path& dst, bool verbose=false) {
    if (verbose) std::cerr << "    deep copy repack " << src << " -> " << dst << "\n";

    hid_t fsrc = H5Fopen(src.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (fsrc < 0) return false;

    // Create a brand new destination file
    hid_t fdst = H5Fcreate(dst.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (fdst < 0) { H5Fclose(fsrc); return false; }

    // Open root groups
    hid_t gsrc = H5Gopen2(fsrc, "/", H5P_DEFAULT);
    hid_t gdst = H5Gopen2(fdst, "/", H5P_DEFAULT);
    if (gsrc < 0 || gdst < 0) {
        if (gsrc >= 0) H5Gclose(gsrc);
        if (gdst >= 0) H5Gclose(gdst);
        H5Fclose(fdst); H5Fclose(fsrc);
        return false;
    }

    // Iterate top-level links under "/" and copy each to destination root
    H5G_info_t gi;
    if (H5Gget_info(gsrc, &gi) >= 0) {
        for (hsize_t i = 0; i < gi.nlinks; ++i) {
            // fetch child name
            ssize_t len = H5Lget_name_by_idx(gsrc, ".", H5_INDEX_NAME, H5_ITER_NATIVE,
                                             i, nullptr, 0, H5P_DEFAULT);
            if (len < 0) continue;
            std::string name; name.resize(static_cast<size_t>(len) + 1);
            H5Lget_name_by_idx(gsrc, ".", H5_INDEX_NAME, H5_ITER_NATIVE,
                               i, name.data(), name.size(), H5P_DEFAULT);
            name.resize(static_cast<size_t>(len));

            // src path is "/<name>", dst link name is "<name>"
            std::string src_obj = "/" + name;

            // Copy the object/link. Destination link must not exist (true for new file).
            herr_t st = H5Ocopy(fsrc, src_obj.c_str(), fdst, name.c_str(),
                                H5P_DEFAULT, H5P_DEFAULT);
            if (st < 0 && verbose) {
                std::cerr << "    WARN: H5Ocopy failed for " << src_obj << "\n";
            }
        }
    }

    H5Gclose(gsrc);
    H5Gclose(gdst);
    H5Fclose(fdst);
    H5Fclose(fsrc);
    return true;
}


static void repack_file(const fs::path& p, int deleted_count, bool verbose, bool dry_run) {
    if (deleted_count <= 0) { if (verbose) std::cerr << "  repack skipped: " << p << "\n"; return; }
    fs::path tmp = p; tmp += ".repacked";
    std::error_code ec; if (fs::exists(tmp, ec)) fs::remove(tmp, ec);
    if (verbose || dry_run) std::cerr << "  repacking (deep copy) " << p << " -> " << tmp << "\n";
    if (dry_run) return;
    if (!deep_copy_file(p, tmp, /*verbose=*/false)) {
        if (verbose) std::cerr << "  ERROR: deep copy repack failed for " << p << "\n";
        return;
    }
    fs::rename(tmp, p, ec);
    if (ec && verbose) std::cerr << "  ERROR: replacing with repacked file failed for " << p
                                 << " : " << ec.message() << "\n";
}

// ===== per-file processing =====
struct Stats{
    int files_assigned=0, files_processed=0, datasets_deleted=0, groups_pruned=0;
    uint64_t bytes_before=0, bytes_after=0;
};

static Stats process_files(const std::vector<std::string>& my_files,
                           bool dry_run, bool prune, bool verbose, int rank)
{
    Stats totals;
    totals.files_assigned = static_cast<int>(my_files.size());

    for (const auto& f : my_files) {
        fs::path p(f);
        if (verbose) std::cerr << "[rank " << rank << "] ==> " << p << "\n";

        std::error_code ec;
        uint64_t size_before = fs::exists(p, ec) ? static_cast<uint64_t>(fs::file_size(p, ec)) : 0;
        int deleted_here = 0;
        int pruned_here  = 0;

        // Try opening file
        hid_t file = H5Fopen(p.string().c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
        if (file < 0) {
            std::cerr << "  ERROR opening " << p << "\n";
        } else {
            totals.files_processed += 1;

            // Collect datasets (cycle_*) and groups
            std::vector<std::string> groups;
            std::vector<PathCycle> datasets;

            hid_t root = H5Gopen2(file, "/", H5P_DEFAULT);
            if (root >= 0) {
                collect_groups_and_datasets(root, "/", groups, datasets);
                H5Gclose(root);
            }

            if (datasets.empty()) {
                if (verbose) std::cerr << "  No cycle_* datasets found; nothing to do.\n";
            } else {
                // Determine latest cycle (prefer /last_cycle)
                auto last_opt = read_last_cycle(file, verbose);
                long long max_cyc = datasets.front().cycle;
                for (const auto& pc : datasets) max_cyc = std::max(max_cyc, pc.cycle);
                long long latest = last_opt.has_value() ? *last_opt : max_cyc;
                if (verbose) {
                    std::cerr << "  Latest cycle = " << latest
                              << (last_opt ? " (/last_cycle)" : " (inferred)") << "\n";
                }

                // Build deletion list (all cycles < latest), deepest-first
                std::vector<PathCycle> to_delete;
                to_delete.reserve(datasets.size());
                for (const auto& pc : datasets) if (pc.cycle < latest) to_delete.push_back(pc);

                std::sort(to_delete.begin(), to_delete.end(),
                          [](const PathCycle& a, const PathCycle& b) {
                              auto da = std::count(a.path.begin(), a.path.end(), '/');
                              auto db = std::count(b.path.begin(), b.path.end(), '/');
                              if (da != db) return da > db;
                              return a.path > b.path;
                          });

                for (const auto& pc : to_delete) {
                    if (verbose || dry_run) {
                        std::cerr << "  delete " << pc.path << " (cycle " << pc.cycle << ")\n";
                    }
                    if (!dry_run) {
                        if (delete_link(file, pc.path)) ++deleted_here;
                    }
                }

                // Optional prune of empty groups
                if (prune) {
                    if (dry_run) {
                        if (verbose) std::cerr << "  (dry-run) would prune empty groups\n";
                    } else {
                        pruned_here = prune_empty_groups(file, verbose);
                    }
                }
            }

            H5Fflush(file, H5F_SCOPE_GLOBAL);
            H5Fclose(file);
        }

        // Repack if anything was deleted in THIS file
        repack_file(p, deleted_here, verbose, dry_run);

        // Size after
        uint64_t size_after = fs::exists(p, ec) ? static_cast<uint64_t>(fs::file_size(p, ec)) : 0;

        // Accumulate totals
        totals.bytes_before += size_before;
        totals.bytes_after  += size_after;
        totals.datasets_deleted += deleted_here;
        totals.groups_pruned  += pruned_here;
    }

    return totals;
}

// ===== main =====
int main(int argc, char** argv){
    setenv("HDF5_USE_FILE_LOCKING","FALSE",0);

    MPI_Init(&argc,&argv);
    int rank=0,size=1; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);

    Args args = parse_args(argc, argv, rank);

    // bcast file list
    std::vector<std::string> files;
    if(rank==0) files = args.files;
    int count = (rank==0)? (int)files.size() : 0;
    MPI_Bcast(&count,1,MPI_INT,0,MPI_COMM_WORLD);
    if(rank!=0) files.resize(count);
    for(int i=0;i<count;++i){
        int len = (rank==0)? (int)files[i].size() : 0;
        MPI_Bcast(&len,1,MPI_INT,0,MPI_COMM_WORLD);
        if(rank!=0) files[i].resize(len);
        if(len>0) MPI_Bcast(files[i].data(), len, MPI_CHAR, 0, MPI_COMM_WORLD);
    }

    // partition
    std::vector<std::string> my_files;
    for(int i=rank;i<count;i+=size) my_files.push_back(files[i]);
    if(args.verbose) std::cerr << "[rank " << rank << "/" << size << "] assigned " << my_files.size() << " files\n";

    Stats my = process_files(my_files, args.dry_run, args.prune_empty, args.verbose, rank);

    struct Packed{ int a,b,c,d; uint64_t e,f; } mine{ my.files_assigned,my.files_processed,my.datasets_deleted,my.groups_pruned,my.bytes_before,my.bytes_after };
    std::vector<Packed> all; if(rank==0) all.resize(size);
    MPI_Gather(&mine, sizeof(Packed), MPI_BYTE, rank==0? all.data():nullptr, sizeof(Packed), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    if(rank==0){
        int files_assigned=0, files_processed=0, datasets_deleted=0, groups_pruned=0;
        uint64_t bytes_before=0, bytes_after=0;
        for(const auto& p: all){ files_assigned+=p.a; files_processed+=p.b; datasets_deleted+=p.c; groups_pruned+=p.d; bytes_before+=p.e; bytes_after+=p.f; }
        uint64_t saved = (bytes_after<bytes_before)? (bytes_before-bytes_after):0;
        double pct = (bytes_before>0)? (100.0*double(saved)/double(bytes_before)):0.0;

        std::cout << "\n============= SUMMARY =============\n";
        std::cout << "Files assigned      : " << files_assigned << "\n";
        std::cout << "Files processed     : " << files_processed << "\n";
        std::cout << "Datasets deleted    : " << datasets_deleted << "\n\n";
        std::cout << "Total size of all restart files before   : " << fmt_bytes(bytes_before) << "\n";
        std::cout << "Total size of all restart files after    : " << fmt_bytes(bytes_after)  << "\n";
        std::cout << "Space saved                              : " << fmt_bytes(saved)
                  << " (" << std::fixed << std::setprecision(2) << pct << "%)\n";
        if(files_assigned==files_processed) std::cout << "\n✅ All restart files processed successfully.\n";
        else std::cout << "\n⚠️ Some files may not have been processed, check logs.\n";
        std::cout.flush();
    }

    MPI_Finalize();
    return 0;
}