# -*- coding: utf-8 -*-
# Multi-person face-blend inference:
#   Same two-phase design as human_lrm_face_blend.py, but animates N avatars
#   (one per source image) with N per-person motion dirs in a SHARED world and
#   renders them together per frame by concatenating the posed GaussianModels.
#
# Expected motion root layout (cfg.motion_seqs_dir points to <root>):
#   <root>/cam_params.json                   # {"26": {focal, princpt, R, t}}
#   <root>/person_00/smplx_params/%06d.json
#   <root>/person_01/smplx_params/%06d.json
#
# Images in cfg.image_input (dir) are sorted and matched to sorted person dirs.

import glob
import os
import os.path as osp

import numpy as np
import torch
from accelerate.logging import get_logger
from tqdm.auto import tqdm

from LHM.runners import REGISTRY_RUNNERS
from LHM.runners.infer.human_lrm_face_blend import (
    HumanLRMFaceBlendInferrer,
    _GS_ATTRS,
    _SMPLX_KEYS,
    _SOFT_BLEND_ATTRS,
    _create_motion_history,
)
from LHM.utils.ffmpeg_utils import images_to_video

logger = get_logger(__name__)


def _merge_gaussian_models(gs_models, use_rgb):
    """Concatenate several posed GaussianModel objects into one."""
    cls = type(gs_models[0])
    return cls(
        xyz=torch.cat([g.xyz for g in gs_models], dim=0),
        opacity=torch.cat([g.opacity for g in gs_models], dim=0),
        rotation=torch.cat([g.rotation for g in gs_models], dim=0),
        scaling=torch.cat([g.scaling for g in gs_models], dim=0),
        shs=torch.cat([g.shs for g in gs_models], dim=0),
        use_rgb=use_rgb,
    )


@REGISTRY_RUNNERS.register("infer.human_lrm_face_blend_multi")
class HumanLRMFaceBlendMultiInferrer(HumanLRMFaceBlendInferrer):

    # ── helpers ────────────────────────────────────────────────────────────

    def _person_motion_dirs(self):
        # cfg.motion_seqs_dir points at ONE person's smplx_params dir
        # (e.g. <root>/person_00/smplx_params) so that parse_configs' frame
        # glob works; the root and sibling person dirs are derived from it.
        one = self.cfg.motion_seqs_dir.rstrip("/")
        root = osp.dirname(osp.dirname(one))
        dirs = sorted(glob.glob(osp.join(root, "person_*", "smplx_params")))
        assert dirs, f"no person_*/smplx_params under {root}"
        return dirs

    def _renderer(self):
        mdl = self.model
        if hasattr(mdl, "base_model"):  # PEFT wrapper
            mdl = mdl.base_model.model
        return mdl.renderer

    # ── phase 2 (multi) ────────────────────────────────────────────────────

    def _run_dynamic_multi(self, tasks, motion_dirs, dump_video_path, interp="hard"):
        device = "cuda"
        dtype = torch.float32
        self.model.to(dtype)
        renderer = self._renderer()
        P = len(tasks)

        # per-person preparation
        per = []
        for p in range(P):
            t = tasks[p]
            image, src_head_rgb = self._prepare_image_inputs(t["image_path"])
            motion_seq = self._get_motion_seq(motion_dirs[p], t["dump_tmp"])
            smplx_params = {k: v.to(device) for k, v in motion_seq["smplx_params"].items()}
            shape_t = torch.tensor(t["beta"], dtype=dtype).unsqueeze(0).to(device)
            smplx_params["betas"] = shape_t

            motion_history = _create_motion_history(
                smplx_params, self.cfg.n_history_length, motion_dirs[p],
                self.cfg.sampling_stride,
            )

            face_data = np.load(t["face_npz"])
            face_mask = face_data["face_mask"]
            face_attrs = {
                a: torch.from_numpy(face_data[a]).to(device, dtype) for a in _GS_ATTRS
            }
            soft = interp == "soft" and "face_blend_weights" in face_data
            entry = dict(
                image=image, src_head_rgb=src_head_rgb, motion_seq=motion_seq,
                smplx_params=smplx_params, shape_t=shape_t,
                motion_history=motion_history, face_mask=face_mask,
                face_attrs=face_attrs, soft=soft,
            )
            if soft:
                entry["face_w"] = torch.from_numpy(face_data["face_blend_weights"]).to(device, dtype)
                entry["body_w"] = torch.from_numpy(face_data["body_blend_weights"]).to(device, dtype)
                entry["body_nn"] = torch.from_numpy(face_data["body_face_nn_idx"]).long().to(device)
            per.append(entry)

        num_frames = min(len(e["motion_seq"]["motion_seqs"]) for e in per)
        print(f"[Multi] persons={P}, frames={num_frames}")
        batch_size = 8
        frames_out = []

        # shared camera comes from person 0's motion_seq (same cam_params.json)
        cam_seq = per[0]["motion_seq"]
        render_intrs = cam_seq["render_intrs"]
        render_h = int(render_intrs[0, 0, 1, 2] * 2)
        render_w = int(render_intrs[0, 0, 0, 2] * 2)
        print(f"[Multi] render {render_w}x{render_h}")

        for batch_i in range(0, num_frames, batch_size):
            nv = min(batch_size, num_frames - batch_i)
            print(f"[Multi] batch {batch_i // batch_size + 1}/{(num_frames - 1) // batch_size + 1}")
            posed = [[None] * nv for _ in range(P)]

            with torch.no_grad():
                for p, e in enumerate(per):
                    ms = e["motion_seq"]
                    gs_model_list, query_points, transform_mat = self.model.infer_single_view(
                        e["image"].unsqueeze(0).to(device, dtype),
                        e["src_head_rgb"].unsqueeze(0).to(device, dtype),
                        None, None,
                        render_c2ws=ms["render_c2ws"].to(device),
                        render_intrs=ms["render_intrs"].to(device),
                        render_bg_colors=ms["render_bg_colors"].to(device),
                        smplx_params=e["smplx_params"],
                        is_dynamic=True,
                        motion_history=e["motion_history"][batch_i:batch_i + nv],
                    )

                    # face gaussian replacement (same as single-person runner)
                    face_mask = e["face_mask"]
                    body_mask = ~face_mask
                    for gs in gs_model_list:
                        if e["soft"]:
                            fw = e["face_w"]
                            for attr in _SOFT_BLEND_ATTRS:
                                tensor = getattr(gs, attr)
                                base_vals = e["face_attrs"][attr]
                                w = 0.5 + 0.5 * fw.view(-1, *([1] * (base_vals.dim() - 1)))
                                tensor[face_mask] = w * base_vals + (1.0 - w) * tensor[face_mask]
                            gs.rotation[face_mask] = torch.nn.functional.normalize(
                                gs.rotation[face_mask], dim=-1)
                            for attr in _SOFT_BLEND_ATTRS:
                                tensor = getattr(gs, attr)
                                fv = e["face_attrs"][attr][e["body_nn"]]
                                w = 0.5 * e["body_w"].view(-1, *([1] * (fv.dim() - 1)))
                                tensor[body_mask] = w * fv + (1.0 - w) * tensor[body_mask]
                            gs.rotation[body_mask] = torch.nn.functional.normalize(
                                gs.rotation[body_mask], dim=-1)
                        else:
                            for attr, base_vals in e["face_attrs"].items():
                                getattr(gs, attr)[face_mask] = base_vals

                    batch_smplx = {
                        "betas": e["shape_t"],
                        "transform_mat_neutral_pose": transform_mat,
                    }
                    for key in _SMPLX_KEYS:
                        batch_smplx[key] = ms["smplx_params"][key][
                            :, batch_i:batch_i + nv
                        ].to(device)

                    # pose each frame's gaussians WITHOUT rendering
                    for v in range(nv):
                        smpl_sv = renderer.get_single_view_smpl_data(batch_smplx, v)
                        smpl_sb = renderer.get_single_batch_smpl_data(smpl_sv, 0)
                        gs_list, _, _ = renderer.animate_gs_model_face_blend(
                            gs_model_list[v], query_points[v], smpl_sb,
                        )
                        posed[p][v] = gs_list[0]

                # merged render per frame
                for v in range(nv):
                    merged = _merge_gaussian_models(
                        [posed[p][v] for p in range(P)], use_rgb=renderer.gs_net.use_rgb,
                    )
                    fidx = batch_i + v
                    out = renderer.forward_single_batch(
                        [merged],
                        cam_seq["render_c2ws"][0, fidx:fidx + 1].to(device),
                        cam_seq["render_intrs"][0, fidx:fidx + 1].to(device),
                        render_h, render_w,
                        cam_seq["render_bg_colors"][0, fidx:fidx + 1].to(device),
                    )
                    rgb = out["comp_rgb"][0]          # [H, W, 3]
                    mask = out["comp_mask"][0]        # [H, W, 1]
                    mask = (mask >= 0.5).to(rgb.dtype)
                    frame = rgb * mask + (1 - mask) * 1.0
                    frames_out.append(
                        (frame.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
                    )
            torch.cuda.empty_cache()

        rgb = np.stack(frames_out, axis=0)
        os.makedirs(osp.dirname(dump_video_path), exist_ok=True)
        images_to_video(rgb, output_path=dump_video_path,
                        fps=self.cfg.render_fps, gradio_codec=False, verbose=True)
        print(f"[Multi] Saved video -> {dump_video_path}")

    # ── main entry ─────────────────────────────────────────────────────────

    def infer(self):
        motion_dirs = self._person_motion_dirs()

        image_paths = []
        assert os.path.isdir(self.cfg.image_input), "image_input must be a dir with N person images"
        for f in sorted(os.listdir(self.cfg.image_input)):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                image_paths.append(osp.join(self.cfg.image_input, f))
        assert len(image_paths) == len(motion_dirs), (
            f"{len(image_paths)} images vs {len(motion_dirs)} person motion dirs"
        )
        print("[Multi] pairing:")
        for img, md in zip(image_paths, motion_dirs):
            print(f"  {osp.basename(img)}  <->  {md}")

        interp = getattr(self.cfg, "interp", "hard")
        blend_hops = int(getattr(self.cfg, "blend_hops", 10))
        motion_root = osp.dirname(osp.dirname(self.cfg.motion_seqs_dir.rstrip("/")))
        motion_name = osp.basename(motion_root)

        tasks = []
        for img in image_paths:
            uid = osp.basename(img).split(".")[0]
            dump_tmp = osp.join(self.cfg.image_dump, "multi", "tmp_res")
            os.makedirs(dump_tmp, exist_ok=True)
            tasks.append({
                "image_path": img,
                "uid": uid,
                "dump_tmp": dump_tmp,
                "face_npz": osp.join(dump_tmp, f"base_face_gs_{uid}.npz"),
            })

        dump_video_path = osp.join(
            self.cfg.video_dump,
            f"motion_{motion_name}",
            "multi_" + "_".join(t["uid"] for t in tasks) + ".mp4",
        )

        # ── Phase 1: base model face gaussians (per person) ────────────────
        print("\n[Multi] === Phase 1: base face extraction ===")
        self._load_base_model()
        for t, mdir in zip(tqdm(tasks, desc="[Phase 1]"), motion_dirs):
            shape_pose = self.pose_estimator(t["image_path"])
            assert shape_pose.ratio > 0.4, f"body ratio too small: {t['image_path']}"
            t["beta"] = shape_pose.beta
            self._extract_base_face_gaussians(
                t["image_path"], mdir, t["dump_tmp"],
                shape_pose.beta, t["face_npz"],
                interp=interp, blend_hops=blend_hops,
            )
        self._unload_model()

        # ── Phase 2: dynamic multi-person render ───────────────────────────
        print("\n[Multi] === Phase 2: dynamic multi-person render ===")
        self._load_dynamic_model()
        self._run_dynamic_multi(tasks, motion_dirs, dump_video_path, interp=interp)
        self._unload_model()
        print("\n[Multi] Done.")
