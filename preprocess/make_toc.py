# build_toc_sqlite_parallel.py
import argparse
import tarfile
import sqlite3
from tqdm import tqdm
from glob import glob
import os
import os.path as osp
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir", type=str, required=True, dest='target_dir', help="directory containing UID.tar")
    parser.add_argument('--max_workers', type=int, default=max(1, (os.cpu_count() or 4) // 4), help="number of parallel processes")
    args = parser.parse_args()
    assert args.target_dir, "Please set target_dir"
    return args

def process_single_tar(tar_path, save_dir): 
    uid = tar_path.split('/')[-1][:-4]
    try:
        BASE = re.escape(f"{uid}")
        IMG_PAT   = re.compile(rf"{BASE}/images/([^/]+)/(\d{{6}})\.(?:jpg|jpeg|png)$", re.I)
        MSK_PAT   = re.compile(rf"{BASE}/masks/([^/]+)/(\d{{6}})\.(?:jpg|jpeg|png)$", re.I)
        SMPLX_PAT = re.compile(rf"{BASE}/smplx/smplx_params_smooth/(\d{{6}})\.json$", re.I)

        db_path = osp.join(save_dir, uid + ".idx.sqlite")
        
        # Delete existing DB
        if osp.exists(db_path): 
            print(f"[DEL] already exists: {db_path}")
            os.remove(db_path)

        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.executescript("""
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        DROP TABLE IF EXISTS files;
        DROP TABLE IF EXISTS images;
        DROP TABLE IF EXISTS masks;
        DROP TABLE IF EXISTS smplx;

        CREATE TABLE files (path TEXT PRIMARY KEY, off INTEGER, size INTEGER);
        CREATE TABLE images(uid TEXT, cam TEXT, frame TEXT, PRIMARY KEY(uid,cam,frame));
        CREATE TABLE masks (uid TEXT, cam TEXT, frame TEXT, PRIMARY KEY(uid,cam,frame));
        CREATE TABLE smplx (uid TEXT, frame TEXT, PRIMARY KEY(uid,frame));

        CREATE INDEX IF NOT EXISTS idx_images_uid_cam ON images(uid, cam);
        CREATE INDEX IF NOT EXISTS idx_masks_uid_cam  ON masks(uid, cam);
        """
        )

        with tarfile.open(tar_path, "r:") as tf:
            for ti in tf:
                if not ti.isfile(): continue
                name = ti.name
                off = getattr(ti, "offset_data", None)
                if off is None:
                    off = ti.offset + 512
                size = int(ti.size)

                cur.execute("INSERT OR REPLACE INTO files(path, off, size) VALUES (?,?,?)", (name, int(off), size))
                
                m = IMG_PAT.match(name)
                if m:
                    cam, frame = m.groups()
                    cur.execute("INSERT OR IGNORE INTO images VALUES (?,?,?)", (uid, str(cam), str(frame)))
                    continue

                m = MSK_PAT.match(name)
                if m:
                    cam, frame = m.groups()
                    cur.execute("INSERT OR IGNORE INTO masks VALUES (?,?,?)", (uid, str(cam), str(frame)))
                    continue

                m = SMPLX_PAT.match(name)
                if m:
                    frame = m.group(1)
                    cur.execute("INSERT OR IGNORE INTO smplx VALUES (?,?)", (uid, str(frame)))
                    continue
                
        con.commit()
        con.close()
        return uid, "ok", db_path

    except Exception as e:
        return uid, "error", str(e)

def main():
    args = parse_args()
    target_dir = args.target_dir

    tar_path_list = glob(osp.join(target_dir, '*.tar'))
    if 'DNA' in target_dir: 
        tar_path_list.sort(key=lambda x: (int(x.split('/')[-1].split('_')[0]), int(x.split('/')[-1].split('_')[1][:-4])))
    elif '4D' in target_dir: 
        tar_path_list.sort(key=lambda x: (int(x.split('/')[-1].split('_')[0]), x.split('/')[-1].split('_')[1], int(x.split('/')[-1].split('_')[2][:-4])))

    results = []
    max_workers = max(1, min(args.max_workers, len(tar_path_list)))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_tar, tar_path, target_dir) for tar_path in tar_path_list]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Building TOCs"):
            res = fut.result()
            results.append(res)
            if res[1] != "ok":
                print(f"[ERROR] UID: {res[0]}, Msg: {res[2]}")
    print("All tasks finished.")

if __name__ == "__main__":
    main()