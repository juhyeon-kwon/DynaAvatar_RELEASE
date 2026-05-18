# -*- coding: utf-8 -*-
# Face-replace inference runner:
#   Phase 1 – base LHM (is_dynamic=False, from_pretrained only) runs
#              infer_single_view once per subject.  The face gaussians
#              (smplx_model.is_face mask) are extracted and saved to disk.
#   Phase 2 – finetuned dynamic model runs per-batch infer_single_view.
#              Before animation_infer, face gaussian attributes are replaced
#              with the saved Phase-1 values.  Rendering is then done as usual.
#
# Both models are never in GPU memory simultaneously (memory-saving design).

import argparse
import os
import os.path as osp
import json
from collections import deque

import cv2
import numpy as np
import torch
from accelerate.logging import get_logger
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm

from engine.pose_estimation.pose_estimator import PoseEstimator

try:
    from engine.SegmentAPI.SAM import SAM2Seg
except Exception as e:
    print(e)
    from rembg import remove

from LHM.runners import REGISTRY_RUNNERS
from LHM.runners.infer.human_lrm import (
    avaliable_device,
    download_geo_files,
    infer_preprocess_image,
    parse_configs,
    prior_check,
)
from LHM.runners.infer.utils import prepare_motion_seqs_2
from LHM.utils.face_detector import FaceDetector
from LHM.utils.ffmpeg_utils import images_to_video
from LHM.utils.hf_hub import wrap_model_hub
from LHM.utils.logging import configure_logger
from LHM.utils.model_download_utils import AutoModelQuery
from .base_inferrer import Inferrer

logger = get_logger(__name__)

_GS_ATTRS = ['offset_xyz', 'opacity', 'rotation', 'scaling', 'shs']
_SOFT_BLEND_ATTRS = ('offset_xyz', 'opacity', 'rotation', 'scaling', 'shs')


def _compute_face_blend_weights(smplx_model, blend_hops=10):
    """Bidirectional BFS on SMPLX mesh topology for seamless face/body blending.

    Returns:
        face_blend_weights [N_face]:  0=boundary, 1=deep interior  (face side)
        body_blend_weights [N_body]:  1=boundary, 0=deep interior  (body side)
        body_face_nn_idx   [N_body]:  index into face gaussians — nearest face pt
                                      for each body gaussian (used to pull color)
    """
    from pytorch3d.ops import knn_points

    smpl_x = smplx_model.smpl_x
    faces = smpl_x.face_orig          # [F, 3], int64, indices into 10475 vertices
    V = smpl_x.vertex_num             # 10475

    adj = [[] for _ in range(V)]
    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        adj[a].append(b); adj[a].append(c)
        adj[b].append(a); adj[b].append(c)
        adj[c].append(a); adj[c].append(b)

    face_vert_set = set(smpl_x.face_vertex_idx.tolist())
    body_vert_set = set(range(V)) - face_vert_set

    # Template vertex positions for neck-side boundary detection (y-axis = up in SMPLX)
    template_verts_np = smplx_model.smplx_layer.v_template.cpu().numpy()  # [V, 3]
    face_vert_list = list(face_vert_set)
    # Use face centroid y as threshold: vertices below centroid are jaw/neck side
    face_centroid_y = template_verts_np[face_vert_list, 1].mean()

    # ── BFS inward: neck/jaw boundary only → face interior ─────────────────
    # Starting from ALL face boundaries (including hairline) causes upper-face
    # Gaussians to get small depth values and be partially blended with the dynamic
    # model. By starting only from the neck/jaw side, upper face Gaussians are
    # unreachable and default to depth=blend_hops → weight=1.0 (fully base model).
    face_depth = {}
    queue = deque()
    for v in face_vert_set:
        if any(nb not in face_vert_set for nb in adj[v]):
            if template_verts_np[v, 1] < face_centroid_y:  # neck/jaw side only
                face_depth[v] = 0
                queue.append(v)
    while queue:
        v = queue.popleft()
        for nb in adj[v]:
            if nb in face_vert_set and nb not in face_depth:
                face_depth[nb] = face_depth[v] + 1
                queue.append(nb)

    # ── BFS outward: neck-adjacent body boundary → body interior ───────────
    body_depth = {}
    queue = deque()
    for v in body_vert_set:
        face_nbs = [nb for nb in adj[v] if nb in face_vert_set]
        if face_nbs:
            if min(template_verts_np[nb, 1] for nb in face_nbs) < face_centroid_y:
                body_depth[v] = 0
                queue.append(v)
    while queue:
        v = queue.popleft()
        d_next = body_depth[v] + 1
        if d_next >= blend_hops:
            continue
        for nb in adj[v]:
            if nb in body_vert_set and nb not in body_depth:
                body_depth[nb] = d_next
                queue.append(nb)

    # ── Per-vertex weights ──────────────────────────────────────────────────
    # face side: 0 at neck boundary, 1 deep inside (unreachable → defaults to 1)
    smplx_face_w = np.zeros(V, dtype=np.float32)
    for v in face_vert_set:
        # default=blend_hops so hairline/upper face vertices unreachable from neck
        # BFS get weight 1.0 (fully from base model, no blending)
        smplx_face_w[v] = min(face_depth.get(v, blend_hops) / max(blend_hops, 1), 1.0)

    # body side: 1 at neck boundary, 0 deep outside
    smplx_body_w = np.zeros(V, dtype=np.float32)
    # body dense pts whose nearest SMPLX vertex is a face vertex → treat as boundary (1.0)
    for v in face_vert_set:
        smplx_body_w[v] = 1.0
    for v, d in body_depth.items():
        smplx_body_w[v] = max(1.0 - d / max(blend_hops, 1), 0.0)

    # ── Map dense pts → nearest SMPLX template vertex ──────────────────────
    dense_pts = smplx_model.dense_pts.cuda()
    template_verts = smplx_model.smplx_layer.v_template.cuda()
    query_indx = knn_points(
        dense_pts.unsqueeze(0), template_verts.unsqueeze(0), K=1
    ).idx.squeeze(0, -1).cpu().numpy()  # [N_dense]

    is_face = smplx_model.is_face.cpu().numpy()  # [N_dense] bool

    face_blend_weights = smplx_face_w[query_indx][is_face]    # [N_face]
    body_blend_weights = smplx_body_w[query_indx][~is_face]   # [N_body]

    # ── KNN: each body pt → nearest face pt (canonical template space) ─────
    body_pts = dense_pts[torch.from_numpy(~is_face).cuda()]   # [N_body, 3]
    face_pts = dense_pts[torch.from_numpy(is_face).cuda()]    # [N_face, 3]
    body_face_nn_idx = knn_points(
        body_pts.unsqueeze(0), face_pts.unsqueeze(0), K=1
    ).idx.squeeze(0, -1).cpu().numpy()  # [N_body], indices into face gaussian array

    n_face_boundary = (face_blend_weights < 0.1).sum()
    n_body_boundary = (body_blend_weights > 0.0).sum()
    print(f"[FaceBlend] face boundary gaussians (<0.1): {n_face_boundary}, "
          f"body boundary gaussians (>0): {n_body_boundary}")

    return face_blend_weights, body_blend_weights, body_face_nn_idx


_SMPLX_KEYS = [
    "root_pose", "body_pose", "jaw_pose",
    "leye_pose", "reye_pose", "lhand_pose", "rhand_pose",
    "trans", "expr",
]


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

def parse_configs_face_blend():
    """Wraps parse_configs() and additionally resolves base_model_name."""
    cfg, cfg_train = parse_configs()

    query_model = AutoModelQuery()
    if "base_model_name" in cfg:
        cfg.base_model_name = query_model.query(str(cfg.base_model_name))
    else:
        cfg.base_model_name = cfg.model_name

    return cfg, cfg_train


# ──────────────────────────────────────────────────────────────────────────────
# Motion history helper (no self dependency, copied from HumanLRMInferrer)
# ──────────────────────────────────────────────────────────────────────────────

def _create_motion_history(smplx_params, n_history_length, motion_seqs_dir, sampling_stride=1):
    root_pose = smplx_params['root_pose'].reshape(
        smplx_params['root_pose'].shape[0], smplx_params['root_pose'].shape[1], 1, 3
    )
    jaw_pose  = smplx_params['jaw_pose'].reshape(
        smplx_params['jaw_pose'].shape[0], smplx_params['jaw_pose'].shape[1], 1, 3
    )
    leye_pose  = torch.zeros_like(root_pose)
    reye_pose  = torch.zeros_like(root_pose)
    body_pose  = smplx_params['body_pose']
    lhand_pose = smplx_params['lhand_pose']
    rhand_pose = smplx_params['rhand_pose']
    transl     = smplx_params['trans'].squeeze(0)

    if '4D' in motion_seqs_dir or 'actorshq' in motion_seqs_dir:
        from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
        device = root_pose.device
        rot = torch.tensor([[1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],
                           dtype=torch.float32, device=device).unsqueeze(0)
        rp = axis_angle_to_matrix(root_pose.squeeze(2).view(-1, 3))
        B  = rp.shape[0]
        root_pose = matrix_to_axis_angle(torch.bmm(rot.expand(B,-1,-1), rp)).view(
            smplx_params['root_pose'].shape[0], smplx_params['root_pose'].shape[1], 1, 3
        )
        transl = torch.bmm(rot.expand(B,-1,-1), transl[:,:,None]).view(-1, 3)

    elif 'I3D' in motion_seqs_dir:
        from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
        device = root_pose.device
        rot = torch.tensor([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]],
                           dtype=torch.float32, device=device).unsqueeze(0)
        rp = axis_angle_to_matrix(root_pose.squeeze(2).view(-1, 3))
        B  = rp.shape[0]
        root_pose = matrix_to_axis_angle(torch.bmm(rot.expand(B,-1,-1), rp)).view(
            smplx_params['root_pose'].shape[0], smplx_params['root_pose'].shape[1], 1, 3
        )
        transl = torch.bmm(rot.expand(B,-1,-1), transl[:,:,None]).view(-1, 3)

    full_pose = torch.cat(
        [root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, leye_pose, reye_pose], dim=2
    ).squeeze(0)

    transl = transl.unsqueeze(1)
    full_pose_with_transl = torch.cat([full_pose, transl], dim=1)

    F      = full_pose_with_transl.shape[0]
    device = full_pose_with_transl.device

    effective_history = n_history_length * sampling_stride
    pad      = full_pose_with_transl[0].unsqueeze(0).repeat(effective_history, 1, 1)
    padded   = torch.cat([pad, full_pose_with_transl], dim=0)
    base_idx = torch.arange(F, device=device) + effective_history
    hist_offsets = torch.arange(n_history_length, -1, -1, device=device) * sampling_stride
    all_idx  = base_idx[:, None] - hist_offsets[None, :]
    return padded[all_idx].flip(1)  # (F, window_size, 56, 3)


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────

@REGISTRY_RUNNERS.register("infer.human_lrm_face_blend")
class HumanLRMFaceBlendInferrer(Inferrer):
    """
    Two-phase inference:
      Phase 1  – base pretrained LHM (is_dynamic=False):
                 run infer_single_view once per subject, extract face gaussian
                 attributes (selected by smplx_model.is_face), save to .npz.
      Phase 2  – finetuned dynamic model (is_dynamic=True):
                 for every batch call infer_single_view → replace face gaussian
                 attributes in gs_model_list with Phase-1 values → animation_infer.
    Models are loaded/unloaded sequentially so only one lives in GPU at a time.
    """

    EXP_TYPE: str = "human_lrm_sapdino_bh_sd3_5"

    def __init__(self):
        super().__init__()

        self.cfg, self.cfg_train = parse_configs_face_blend()
        self.cfg.is_dynamic      = self.cfg_train['model'].is_dynamic
        self.cfg.n_history_length = self.cfg_train['model'].n_history_length
        configure_logger(stream_level=self.cfg.logger, log_level=self.cfg.logger)
        prior_check()

        self.facedetect = FaceDetector(
            "./pretrained_models/gagatracker/vgghead/vgg_heads_l.trcd",
            device=avaliable_device(),
        )
        self.pose_estimator = PoseEstimator(
            "./pretrained_models/human_model_files/", device=avaliable_device()
        )
        try:
            self.parsingnet = SAM2Seg()
        except Exception:
            self.parsingnet = None

        self.model      = None   # loaded on demand
        self.motion_dict = dict()

    # ── abstract method stubs ──────────────────────────────────────────────

    def _build_model(self, cfg):
        return None

    def infer_single(self, *args, **kwargs):
        raise NotImplementedError("Use infer() for this runner")

    # ── model loaders ──────────────────────────────────────────────────────

    def _load_base_model(self):
        from LHM.models import model_dict
        hf_cls = wrap_model_hub(model_dict[self.EXP_TYPE])
        self.model = hf_cls.from_pretrained(self.cfg.base_model_name).to(self.device)
        # Safety: enforce static-LHM flags regardless of what config.json says.
        # base_model_name should point to LHM-500M-HF (is_dynamic=false), but
        # if it doesn't, unconstrained offset_xyz would make face splats fly.
        self.model.renderer.gs_net.restrict_offset = True
        self.model.renderer.gs_net.use_skinning_offset = False
        self.model.renderer.use_skinning_offset = False
        logger.info("[FaceBlend] Base model loaded (static mode enforced: restrict_offset=True).")

    def _load_dynamic_model(self):
        from LHM.models import model_dict
        from peft import LoraConfig, get_peft_model

        hf_cls = wrap_model_hub(model_dict[self.EXP_TYPE])
        model  = hf_cls.from_pretrained(self.cfg.model_name)

        lora_target_modules = [
            "to_q", "to_k", "to_v",
            "add_q_proj", "add_k_proj", "add_v_proj",
            "to_add_out", "proj", "linear",
        ]
        modules_to_save = [
            f"transformer.dynamic_layers.{i}.dynamic_dit"
            for i, layer in enumerate(model.transformer.dynamic_layers)
            if hasattr(layer, "dynamic_dit")
        ] + ["fine_encoder", "encoder"]

        lora_config = LoraConfig(
            r=32, lora_alpha=64,
            target_modules=lora_target_modules,
            lora_dropout=0.1, bias="none",
            modules_to_save=modules_to_save,
        )
        self.model = get_peft_model(model, lora_config).to(self.device)
        self.load_model_(
            osp.join(self.cfg_train["saver"].load_model, "model.safetensors")
        )
        logger.info("[FaceBlend] Dynamic model loaded.")

    def _unload_model(self):
        del self.model
        self.model = None
        torch.cuda.empty_cache()
        logger.info("[FaceBlend] Model unloaded.")

    # ── shared preprocessing ───────────────────────────────────────────────

    @torch.no_grad()
    def _parse_mask(self, img_path):
        if self.parsingnet is not None:
            out = self.parsingnet(img_path=img_path, bbox=None)
            return (out.masks * 255).astype(np.uint8)
        img_np    = cv2.imread(img_path)
        remove_np = remove(img_np)
        return remove_np[..., 3]

    def _prepare_image_inputs(self, image_path):
        parsing_mask = self._parse_mask(image_path)
        image, _, _ = infer_preprocess_image(
            image_path, mask=parsing_mask, intr=None,
            pad_ratio=0, bg_color=1.0, max_tgt_size=896,
            aspect_standard=5.0 / 3, enlarge_ratio=[1.0, 1.0],
            render_tgt_size=self.cfg.source_size, multiply=14, need_mask=True,
        )
        try:
            rgb   = np.array(Image.open(image_path))
            rgb_t = torch.from_numpy(rgb).permute(2, 0, 1)
            bbox  = self.facedetect(rgb_t)
            head  = rgb_t[:, int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            head  = head.permute(1, 2, 0).numpy()
        except Exception:
            head = np.zeros((112, 112, 3), dtype=np.uint8)
        try:
            head = cv2.resize(head, (self.cfg.src_head_size, self.cfg.src_head_size),
                              interpolation=cv2.INTER_AREA)
        except Exception:
            head = np.zeros((self.cfg.src_head_size, self.cfg.src_head_size, 3), dtype=np.uint8)

        src_head_rgb = (
            torch.from_numpy(head / 255.0).float().permute(2, 0, 1).unsqueeze(0)
        )
        return image, src_head_rgb

    def _get_motion_seq(self, motion_seqs_dir, dump_tmp_dir):
        motion_name = osp.basename(osp.dirname(motion_seqs_dir.rstrip("/")))
        if motion_name in self.motion_dict:
            return self.motion_dict[motion_name]

        cam_param_path = osp.join(motion_seqs_dir, "..", "..", "cam_params.json")
        with open(cam_param_path) as f:
            camera_name_set = set(json.load(f).keys())

        camera_name = self.cfg.camera_name if self.cfg.camera_name is not None else "26"
        # hard-code to match human_lrm.py line 1023
        camera_name = "26"
        print("Cam:", camera_name)

        motion_seq = prepare_motion_seqs_2(
            motion_seqs_dir, None,
            save_root=dump_tmp_dir,
            fps=self.cfg.motion_video_read_fps,
            bg_color=1.0,
            aspect_standard=5.0 / 3,
            enlarge_ratio=[1.0, 1.0],
            render_image_res=self.cfg.render_size,
            multiply=16,
            need_mask=self.cfg.get("motion_img_need_mask", False),
            vis_motion=self.cfg.get("vis_motion", False),
            motion_st_idx=self.cfg.motion_st_idx,
            motion_size=self.cfg.motion_size,
            cam_param_path=cam_param_path,
            camera_name=camera_name,
        )
        self.motion_dict[motion_name] = motion_seq
        return motion_seq

    # ── Phase 1: extract base face gaussians ──────────────────────────────

    def _extract_base_face_gaussians(self, image_path, motion_seqs_dir, dump_tmp_dir,
                                     shape_param, save_path, interp='hard', blend_hops=10):
        """
        Run base model infer_single_view (is_dynamic=False) once for this subject.
        Extract face gaussian attributes using smplx_model.is_face mask and save to .npz.
        """
        device = "cuda"
        dtype  = torch.float32

        image, src_head_rgb = self._prepare_image_inputs(image_path)
        motion_seq = self._get_motion_seq(motion_seqs_dir, dump_tmp_dir)

        smplx_params = {k: v.to(device) for k, v in motion_seq["smplx_params"].items()}
        shape_t = torch.tensor(shape_param, dtype=dtype).unsqueeze(0).to(device)
        smplx_params["betas"] = shape_t

        self.model.to(dtype)
        with torch.no_grad():
            gs_model_list, query_points, transform_mat = self.model.infer_single_view(
                image.unsqueeze(0).to(device, dtype),
                src_head_rgb.unsqueeze(0).to(device, dtype),
                None, None,
                render_c2ws=motion_seq["render_c2ws"].to(device),
                render_intrs=motion_seq["render_intrs"].to(device),
                render_bg_colors=motion_seq["render_bg_colors"].to(device),
                smplx_params=smplx_params,
                is_dynamic=False,
            )

        # face mask: bool [N_verts], same for every subject (SMPLX topology)
        face_mask = self.model.renderer.smplx_model.is_face.cpu().numpy()  # [N] bool

        gs = gs_model_list[0]  # batch_size=1 for base model
        save_data = {"face_mask": face_mask}
        for attr in _GS_ATTRS:
            val = getattr(gs, attr)
            save_data[attr] = val[face_mask].detach().cpu().numpy()

        if interp == 'soft':
            print(f"[FaceBlend] Computing bidirectional blend weights (blend_hops={blend_hops}) ...")
            face_w, body_w, body_face_nn = _compute_face_blend_weights(
                self.model.renderer.smplx_model, blend_hops=blend_hops
            )
            save_data['face_blend_weights'] = face_w
            save_data['body_blend_weights'] = body_w
            save_data['body_face_nn_idx']   = body_face_nn

        np.savez(save_path, **save_data)
        print(f"[FaceBlend] Saved base face gaussians → {save_path}")
        print(f"            face gaussian count: {face_mask.sum()}")
        # --- debug ---
        face_t = torch.as_tensor(face_mask)
        qp_face = query_points[0][face_t]  # [N_face, 3] canonical mesh face verts
        off_face = gs.offset_xyz[face_t]   # [N_face, 3] delta from canonical mesh
        print(f"[DBG Phase1] query_points[face] mean/std: {qp_face.mean(0).tolist()} / {qp_face.std(0).tolist()}")
        print(f"[DBG Phase1] offset_xyz[face] norm mean: {off_face.norm(dim=-1).mean().item():.4f}  max: {off_face.norm(dim=-1).max().item():.4f}")
        print(f"[DBG Phase1] mean_3d[face] = query+offset mean: {(qp_face+off_face).mean(0).tolist()}")
        del qp_face, off_face, face_t
        # --- end debug ---

    # ── Phase 2: dynamic inference with face replacement ──────────────────

    def _run_dynamic_with_face_replace(self, image_path, motion_seqs_dir, dump_tmp_dir,
                                       shape_param, face_npz_path, dump_video_path,
                                       interp='hard'):
        """
        Run dynamic model per-batch.  Before animation_infer, replace face gaussian
        attributes in each gs_model_list item with the base-model values.
        """
        device = "cuda"
        dtype  = torch.float32

        # Load base face gaussians
        face_data = np.load(face_npz_path)
        face_mask = face_data["face_mask"]  # [N] bool (numpy)
        face_attrs = {
            attr: torch.from_numpy(face_data[attr]).to(device, dtype)
            for attr in _GS_ATTRS
        }

        face_blend_w_t = None
        body_blend_w_t = None
        body_face_nn_t = None
        if interp == 'soft' and 'face_blend_weights' in face_data:
            face_blend_w_t = torch.from_numpy(face_data['face_blend_weights']).to(device, dtype)  # [N_face]
            body_blend_w_t = torch.from_numpy(face_data['body_blend_weights']).to(device, dtype)  # [N_body]
            body_face_nn_t = torch.from_numpy(face_data['body_face_nn_idx']).long().to(device)     # [N_body]
            print(f"[FaceBlend] soft interp: face_w [{face_blend_w_t.min():.3f}, {face_blend_w_t.max():.3f}], "
                  f"body boundary count: {(body_blend_w_t > 0).sum().item()}")
        elif interp == 'soft':
            print("[FaceBlend] WARNING: interp=soft requested but blend weights not in npz; falling back to hard replace.")

        image, src_head_rgb = self._prepare_image_inputs(image_path)
        motion_seq = self._get_motion_seq(motion_seqs_dir, dump_tmp_dir)

        smplx_params = {k: v.to(device) for k, v in motion_seq["smplx_params"].items()}
        shape_t = torch.tensor(shape_param, dtype=dtype).unsqueeze(0).to(device)
        smplx_params["betas"] = shape_t

        motion_history = _create_motion_history(
            smplx_params,
            self.cfg.n_history_length,
            motion_seqs_dir,
            self.cfg.sampling_stride,
        )

        self.model.to(dtype)

        camera_size = len(motion_seq["motion_seqs"])
        batch_size  = 8
        batch_list  = []

        for batch_i in range(0, camera_size, batch_size):
            print(f"[Dynamic+Replace] batch {batch_i // batch_size + 1}/"
                  f"{(camera_size - 1) // batch_size + 1}")
            with torch.no_grad():
                gs_model_list, query_points, transform_mat = self.model.infer_single_view(
                    image.unsqueeze(0).to(device, dtype),
                    src_head_rgb.unsqueeze(0).to(device, dtype),
                    None, None,
                    render_c2ws=motion_seq["render_c2ws"].to(device),
                    render_intrs=motion_seq["render_intrs"].to(device),
                    render_bg_colors=motion_seq["render_bg_colors"].to(device),
                    smplx_params=smplx_params,
                    is_dynamic=True,
                    motion_history=motion_history[batch_i:batch_i + batch_size],
                )

                # ── Replace face gaussians with base-model values ──────────
                body_mask = ~face_mask  # [N_dense] bool numpy
                for gs in gs_model_list:
                    if interp == 'soft' and face_blend_w_t is not None:
                        # ── Face side: soft blend all attrs ────────────────
                        # face_blend_w_t: 0(boundary)→1(interior), scaled to 0.5→1.0
                        for attr in _SOFT_BLEND_ATTRS:
                            tensor = getattr(gs, attr)
                            base_vals = face_attrs[attr]
                            w_raw = face_blend_w_t.view(-1, *([1] * (base_vals.dim() - 1)))
                            w = 0.5 + 0.5 * w_raw
                            tensor[face_mask] = w * base_vals + (1.0 - w) * tensor[face_mask]
                        # quaternion must remain unit-norm after linear blend
                        gs.rotation[face_mask] = torch.nn.functional.normalize(
                            gs.rotation[face_mask], dim=-1
                        )
                        # ── Body side: soft blend all attrs (boundary only) ─
                        # body_blend_w_t: 1(boundary)→0(interior), scaled to 0.5→0.0
                        for attr in _SOFT_BLEND_ATTRS:
                            tensor = getattr(gs, attr)
                            face_vals_for_body = face_attrs[attr][body_face_nn_t]  # [N_body, ...]
                            w_raw = body_blend_w_t.view(-1, *([1] * (face_vals_for_body.dim() - 1)))
                            w = 0.5 * w_raw
                            tensor[body_mask] = w * face_vals_for_body + (1.0 - w) * tensor[body_mask]
                        gs.rotation[body_mask] = torch.nn.functional.normalize(
                            gs.rotation[body_mask], dim=-1
                        )
                    else:
                        for attr, base_vals in face_attrs.items():
                            getattr(gs, attr)[face_mask] = base_vals
                # ──────────────────────────────────────────────────────────

                # --- debug (first batch only) ---
                if batch_i == 0:
                    face_t   = torch.as_tensor(face_mask, device=device)
                    gs0      = gs_model_list[0]
                    qp_face  = query_points[0][face_t]          # [N_face, 3]
                    off_face = gs0.offset_xyz[face_t].float()   # [N_face, 3]
                    try:
                        mdl = self.model
                        if hasattr(mdl, 'base_model'):
                            mdl = mdl.base_model.model
                        print(f"[DBG Ph2] model.is_dynamic={mdl.is_dynamic}")
                        print(f"[DBG Ph2] restrict_offset={mdl.renderer.gs_net.restrict_offset}")
                    except Exception as e:
                        print(f"[DBG Ph2] model attr error: {e}")
                    print(f"[DBG Ph2] query_points[0][face].mean={qp_face.mean(0).cpu().tolist()}")
                    print(f"[DBG Ph2] offset_xyz[face] norm  mean={off_face.norm(dim=-1).mean().item():.4f}  max={off_face.norm(dim=-1).max().item():.4f}")
                    print(f"[DBG Ph2] canonical face pos  mean={(qp_face+off_face).mean(0).cpu().tolist()}")
                    # check replacement
                    expected = face_attrs['offset_xyz'].float()
                    max_err  = (off_face - expected).abs().max().item()
                    print(f"[DBG Ph2] replacement max_err={max_err:.6f}  (0 means perfect)")
                    del qp_face, off_face, face_t, expected
                # --- end debug ---

                batch_smplx = {
                    "betas": shape_t,
                    "transform_mat_neutral_pose": transform_mat,
                }
                for key in _SMPLX_KEYS:
                    batch_smplx[key] = motion_seq["smplx_params"][key][
                        :, batch_i:batch_i + batch_size
                    ].to(device)

                res = self.model.animation_infer_face_blend(
                    gs_model_list, query_points, batch_smplx,
                    render_c2ws=motion_seq["render_c2ws"][:, batch_i:batch_i + batch_size].to(device),
                    render_intrs=motion_seq["render_intrs"][:, batch_i:batch_i + batch_size].to(device),
                    render_bg_colors=motion_seq["render_bg_colors"][:, batch_i:batch_i + batch_size].to(device),
                )

            comp_rgb  = res["comp_rgb"]
            comp_mask = res["comp_mask"]
            comp_mask[comp_mask < 0.5] = 0.0
            batch_rgb = comp_rgb * comp_mask + (1 - comp_mask) * 1.0
            batch_rgb = (batch_rgb.clamp(0, 1) * 255).to(torch.uint8).detach().cpu().numpy()
            batch_list.append(batch_rgb)
            del res
            torch.cuda.empty_cache()

        rgb = np.concatenate(batch_list, axis=0)
        os.makedirs(osp.dirname(dump_video_path), exist_ok=True)
        images_to_video(rgb, output_path=dump_video_path,
                        fps=self.cfg.render_fps, gradio_codec=False, verbose=True)
        print(f"[FaceBlend] Saved video → {dump_video_path}")

    # ── main entry point ───────────────────────────────────────────────────

    def infer(self):
        # ── collect image paths ────────────────────────────────────────────
        image_paths = []
        if os.path.isfile(self.cfg.image_input):
            omit_prefix = os.path.dirname(self.cfg.image_input)
            image_paths.append(self.cfg.image_input)
        else:
            omit_prefix = self.cfg.image_input
            for root, _, files in os.walk(self.cfg.image_input):
                for f in files:
                    if f.endswith((".jpg", ".jpeg", ".png", ".webp", ".JPG")):
                        image_paths.append(osp.join(root, f))
            image_paths.sort()

        image_paths = image_paths[
            self.accelerator.process_index::self.accelerator.num_processes
        ]

        # ── build per-image task metadata ─────────────────────────────────
        motion_seqs_dir = self.cfg.motion_seqs_dir
        tasks = []
        for image_path in image_paths:
            uid      = osp.basename(image_path).split(".")[0]
            subdir   = osp.dirname(image_path).replace(omit_prefix, "").lstrip("/")
            dump_tmp = osp.join(self.cfg.image_dump, subdir, "tmp_res")
            dump_video = osp.join(
                self.cfg.video_dump,
                f"motion_{motion_seqs_dir.split('/')[-3]}",
                f"image_{uid}.mp4",
            )
            face_npz = osp.join(dump_tmp, f"base_face_gs_{uid}.npz")

            os.makedirs(dump_tmp, exist_ok=True)
            os.makedirs(osp.dirname(dump_video), exist_ok=True)

            tasks.append({
                "image_path": image_path,
                "uid":        uid,
                "dump_tmp":   dump_tmp,
                "dump_video": dump_video,
                "face_npz":   face_npz,
            })

        # ══════════════════════════════════════════════════════════════════
        # Phase 1 – Base model: extract face gaussians
        # ══════════════════════════════════════════════════════════════════
        interp = getattr(self.cfg, 'interp', 'hard')
        blend_hops = int(getattr(self.cfg, 'blend_hops', 10))
        print(f"\n[FaceBlend] interp={interp}, blend_hops={blend_hops}")

        print("\n[FaceBlend] ═══ Phase 1: Base model — extract face gaussians ═══")
        self._load_base_model()

        for task in tqdm(tasks, desc="[Phase 1] Base face extract",
                         disable=not self.accelerator.is_local_main_process):
            shape_pose = self.pose_estimator(task["image_path"])
            if shape_pose.ratio <= 0.4:
                print(f"[FaceBlend] Body ratio too small, skipping: {task['image_path']}")
                np.savez(task["face_npz"])  # empty sentinel
                continue

            self._extract_base_face_gaussians(
                task["image_path"], motion_seqs_dir, task["dump_tmp"],
                shape_pose.beta, task["face_npz"],
                interp=interp, blend_hops=blend_hops,
            )

        self._unload_model()

        # ══════════════════════════════════════════════════════════════════
        # Phase 2 – Dynamic model: inference with face gaussian replacement
        # ══════════════════════════════════════════════════════════════════
        print("\n[FaceBlend] ═══ Phase 2: Dynamic model + face gaussian replace ═══")
        self._load_dynamic_model()

        for task in tqdm(tasks, desc="[Phase 2] Dynamic + replace",
                         disable=not self.accelerator.is_local_main_process):
            face_npz = task["face_npz"]
            if not osp.exists(face_npz):
                print(f"[FaceBlend] No face gaussians for {task['image_path']}, skipping.")
                continue

            # Check the saved npz actually has face_mask (empty sentinel = skipped above)
            data = np.load(face_npz)
            if "face_mask" not in data:
                print(f"[FaceBlend] Empty face gaussian file, skipping: {task['image_path']}")
                continue

            shape_pose = self.pose_estimator(task["image_path"])
            if shape_pose.ratio <= 0.4:
                print(f"[FaceBlend] Body ratio too small, skipping: {task['image_path']}")
                continue

            self._run_dynamic_with_face_replace(
                task["image_path"], motion_seqs_dir, task["dump_tmp"],
                shape_pose.beta, face_npz, task["dump_video"],
                interp=interp,
            )

        self._unload_model()
        print("\n[FaceBlend] Done.")
