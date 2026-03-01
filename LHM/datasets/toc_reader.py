# dna_toc_reader.py
import os
import os.path as osp
import io
import sqlite3
import random
from typing import Iterable, Tuple, Optional, List, Dict
from collections import OrderedDict

class TocReader:
    def __init__(self, 
                 tar_path: str, 
                 db_path: Optional[str] = None,
                 sqlite_mmap_size: int = 128 << 20):
        self.tar_path = tar_path
        self.db_path  = db_path
        assert os.path.exists(self.db_path), f"TOC not found: {self.db_path}"

        # raw file handle (무압축 tar면 랜덤 seek 가능)
        self.fh = open(self.tar_path, "rb", buffering=0)
        # SQLite 인덱스: 메모리 거의 0, prefix/정렬 쿼리 용이
        uri = f"file:{self.db_path}?mode=ro&immutable=1"  # read only mode & immutable
        self.con = sqlite3.connect(uri, uri=True, check_same_thread=False)
        self.con.execute("PRAGMA query_only=ON")

    def ping(self) -> bool:
        try:
            self.con.execute("SELECT 1")
            return True
        except Exception:
            return False
    
    # ------------- raw bytes -------------
    def read(self, path: str) -> Optional[bytes]:
        row = self.con.execute("SELECT off,size FROM files WHERE path=?", (path,)).fetchone()
        if not row:
            return None
        off, size = row
        self.fh.seek(off)
        return self.fh.read(size)

    def iter_prefix(self, prefix: str, exts: Tuple[str, ...] = ()) -> Iterable[Tuple[str, bytes]]:
        # prefix로 빠르게 뽑기 (LIKE 인덱스 최적화는 path 앞쪽이 동일할수록 유리)
        like = prefix.replace("%", "%%") + "%"
        q = "SELECT path, off, size FROM files WHERE path LIKE ?"
        for path, off, size in self.con.execute(q, (like,)):
            if exts and not any(path.lower().endswith(e) for e in exts):
                continue
            self.fh.seek(off)
            yield path, self.fh.read(size)

    # ------------- sampling meta -------------
    def list_uids(self) -> List[str]:
        # images와 masks 중 하나라도 있는 uid
        q = "SELECT DISTINCT uid FROM images UNION SELECT DISTINCT uid FROM masks"
        return [r[0] for r in self.con.execute(q).fetchall()]

    def cams_with_images(self, uid: str) -> List[str]:
        q = "SELECT DISTINCT cam FROM images WHERE uid=? ORDER BY cam"
        return [r[0] for r in self.con.execute(q, (uid,)).fetchall()]

    def cams_with_masks(self, uid: str) -> List[str]:
        q = "SELECT DISTINCT cam FROM masks WHERE uid=? ORDER BY cam"
        return [r[0] for r in self.con.execute(q, (uid,)).fetchall()]

    def cams_with_both(self, uid: str) -> List[str]:
        q = """
        SELECT DISTINCT i.cam
        FROM images i
        INNER JOIN masks m ON i.uid=m.uid AND i.cam=m.cam
        WHERE i.uid=? ORDER BY i.cam
        """
        return [r[0] for r in self.con.execute(q, (uid,)).fetchall()]

    def frames_for(self, uid: str, cam: str, kind: str = "images") -> List[str]:
        assert kind in ("images", "masks")
        q = f"SELECT frame FROM {kind} WHERE uid=? AND cam=? ORDER BY frame"
        return [r[0] for r in self.con.execute(q, (uid, cam)).fetchall()]

    def smplx_frames(self, uid: str) -> List[str]:
        q = "SELECT frame FROM smplx WHERE uid=? ORDER BY frame"
        return [r[0] for r in self.con.execute(q, (uid,)).fetchall()]

    def get_min_max_frame_idx(self, uid: str, cam: str | None = None) -> tuple[int | None, int | None]:  # 1004
        if cam is None:
            q = "SELECT MIN(CAST(frame AS INTEGER)), MAX(CAST(frame AS INTEGER)) FROM images WHERE uid=?"
            row = self.con.execute(q, (uid,)).fetchone()
        else:
            q = "SELECT MIN(CAST(frame AS INTEGER)), MAX(CAST(frame AS INTEGER)) FROM images WHERE uid=? AND cam=?"
            row = self.con.execute(q, (uid, cam)).fetchone()
        if not row:
            return None, None
        return row[0], row[1]

    # ------------- high-level helpers -------------
    def choose_two_cams(self, uid: str, rng: Optional[random.Random] = None, require_both=True) -> Tuple[str, str]:
        rng = rng or random
        cams = self.cams_with_both(uid) if require_both else self.cams_with_images(uid)
        if len(cams) < 2:
            raise RuntimeError(f"{uid}: not enough cameras (require_both={require_both})")
        return tuple(rng.sample(cams, 2))

    def choose_frame(self, uid: str, cam: str, rng: Optional[random.Random] = None) -> str:
        rng = rng or random
        frames = self.frames_for(uid, cam, "images")
        if not frames:
            raise RuntimeError(f"{uid}:{cam} has no frames")
        return rng.choice(frames)

    # ------------- convenience reads -------------
    def read_image_bytes(self, uid: str, cam: str, frame: str) -> Optional[bytes]:
        base = f"{uid}/images/{cam}/{frame}"
        for ext in (".jpg", ".jpeg", ".png"):
            b = self.read(base + ext)
            if b:
                return b
        return None
    
    def read_image_face_bytes(self, uid: str, cam: str, frame: str) -> Optional[bytes]:
        base = f"{uid}/images_face/{cam}/{frame}"
        for ext in (".jpg", ".jpeg", ".png"):
            b = self.read(base + ext)
            if b:
                return b
        return None

    def read_mask_bytes(self, uid: str, cam: str, frame: str) -> Optional[bytes]:
        base = f"{uid}/masks/{cam}/{frame}"
        for ext in (".png", ".jpg", ".jpeg"):
            b = self.read(base + ext)
            if b:
                return b
        return None

    def read_cam_param(self, uid: str) -> Optional[bytes]:
        return self.read(f"{uid}/cam_params.json")
    
    def read_face_bbox_bytes(self, uid: str) -> Optional[bytes]:
        return self.read(f"{uid}/face_bbox.json")

    def read_smplx_bytes(self, uid: str, frame: str) -> Optional[bytes]:
        return self.read(f"{uid}/smplx/smplx_params_smooth/{frame}.json"), self.read(f"{uid}/smplx/shape_param.json")

    def read_face_bbox_targ_bytes(self, uid: str) -> Optional[bytes]:
        return self.read(f"{uid}/face_bbox_targ.json")

    def close(self):
        try: 
            self.con.close()
        except: 
            pass
        try: 
            self.fh.close()
        except: 
            pass


class TocLRU:
    """UID별 DNATarTOC 인스턴스를 LRU로 관리해 FD 한도 내에서 운용."""
    def __init__(self, root_dirs: str,
                 capacity: int = 64,
                 sqlite_mmap_size: int = 128 << 20):
        self.root_dirs = root_dirs
        self.cap = int(capacity)
        self.sqlite_mmap_size = sqlite_mmap_size
        self.od: "OrderedDict[str, TocReader]" = OrderedDict()

    def _open(self, uid: str) -> TocReader:
        tar_path = os.path.join(self.root_dirs, f"{uid}.tar")
        toc_path = os.path.join(self.root_dirs, f"{uid}.idx.sqlite")
        return TocReader(tar_path, toc_path,
                         sqlite_mmap_size=self.sqlite_mmap_size)        

    def _ensure_alive(self, uid: str, toc: "TocReader") -> "TocReader":
        if toc and toc.ping():
            return toc
        # 재오픈
        try:
            if toc: toc.close()
        except Exception:
            pass
        return self._open(uid)

    def get(self, uid: str) -> "TocReader":
        toc = self.od.get(uid)
        if toc is not None:
            toc = self._ensure_alive(uid, toc)
            self.od[uid] = toc
            self.od.move_to_end(uid)
            return toc

        # 새로 열기
        toc = self._open(uid)
        self.od[uid] = toc
        self.od.move_to_end(uid)

        # 용량 초과 시 가장 오래된 것부터 닫기
        if len(self.od) > self.cap:
            old_uid, old_toc = self.od.popitem(last=False)
            try: old_toc.close()
            except Exception: pass
        return toc

    def close_all(self):
        while self.od:
            _, toc = self.od.popitem(last=True)
            try: toc.close()
            except Exception: pass