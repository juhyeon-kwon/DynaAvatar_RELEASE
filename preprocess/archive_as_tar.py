import argparse, os, tarfile, shutil, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time

INCLUDES = [
    "images",
    "masks",
    "smplx/smplx_params_smooth",
    "smplx/shape_param.json",
    #"bbox_orig.json",
    "cam_params.json",
    "face_bbox.json",
    "face_bbox_targ.json"
]

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root_dir", required=True, help="root directory containing sequence folders to archive.")
    parser.add_argument("--target_dir", required=True, help="destination directory for tars")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing archives")
    parser.add_argument("--max_workers", type=int, default=max(1, (os.cpu_count() or 4) // 4), help="number of parallel processes")
    
    args = parser.parse_args()
    return args

def make_tar(src_dir: Path, dst_tar: Path):
    dst_tar.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_tar.with_name(dst_tar.name + ".part")
    try:
        with tarfile.open(tmp, "w") as tf:
            # start with "{key}/"
            for rel in INCLUDES: 
                p = src_dir / rel
                if (rel != "bbox_orig.json") and (not p.exists()): 
                    raise FileNotFoundError(f"[ERROR] missing {src_dir.name}/{rel}")
                inner_path = f"{src_dir.name}/{rel}".replace(os.sep, "/")
                tf.add(str(p), arcname=inner_path, recursive=True)
        shutil.move(tmp, dst_tar)
        return True, ""
    except Exception as e:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return False, str(e)

def archive_one(uid: str, dataset: str, dest: str, overwrite: bool) -> tuple[str, str, str]:
    dataset = Path(dataset)
    dest = Path(dest)
    src_dir = dataset / uid
    if not src_dir.is_dir():
        return uid, "skip", f"missing: {src_dir}"
    out_name = f"{uid}.tar"
    dst_tar = dest / out_name
    if dst_tar.exists() and not overwrite:
        return uid, "skip", f"exists: {dst_tar}"
    ok, msg = make_tar(src_dir, dst_tar)
    return (uid, "ok", str(dst_tar)) if ok else (uid, "error", msg)

def main():
    print("Archive: ", INCLUDES)
    
    args = arg_parser()
    dataset_root_dir = Path(args.dataset_root_dir).resolve()
    target_dir = Path(args.target_dir).resolve()
    max_workers = args.max_workers

    uid_list = os.listdir(str(dataset_root_dir))
    if 'DNA' in args.dataset_root_dir: 
        uid_list.sort(key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1])))
    elif '4D' in args.dataset_root_dir: 
        uid_list.sort(key=lambda x: (int(x.split('_')[0]), x.split('_')[1], int(x.split('_')[2])))
    else: 
        pass
    
    print("Target uid length:", len(uid_list))
    
    pending = []
    for uid in uid_list:
        out_name = f"{uid}.tar"
        if not (target_dir / out_name).exists() or args.overwrite:
            pending.append(uid)
    if not pending:
        print("[INFO] nothing to do (all exist or filtered)")
        print("Summary: ok=0, skip=%d, error=0" % len(uid_list))
        return
    
    start =time.time()
    ok = 0; skip = 0; err = 0
    results = []
    max_workers = max(1, min(max_workers, len(pending)))
    print("max workers:", max_workers)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(archive_one, uid, str(dataset_root_dir), str(target_dir), args.overwrite): uid
            for uid in pending
        }
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Archiving", dynamic_ncols=True):
            uid, status, msg = fut.result()
            results.append((uid, status, msg))
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                err += 1
    with open("archive_as_tar_log.txt", "w") as f: 
        for uid, status, msg in results:
            if status == "ok":
                print(f"[ok] {msg}")
                f.write(f"[ok] {msg}\n")
            elif status == "skip":
                print(f"[skip] {msg}")
                f.write(f"[skip] {msg}\n")
            else:
                print(f"[error] {uid}: {msg}", file=sys.stderr)
                f.write(f"[error] {uid}: {msg}\n")

    print(f"Summary: ok={ok}, skip={skip}, error={err}")
    print(f"[FINISH] total: {time.time() - start:.2f}s (workers={max_workers})")

if __name__ == "__main__":
    main()
