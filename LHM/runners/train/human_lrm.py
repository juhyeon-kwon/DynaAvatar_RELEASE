# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import math
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from accelerate.logging import get_logger

from .base_trainer import Trainer
from LHM.utils.profiler import DummyProfiler
from LHM.runners import REGISTRY_RUNNERS

from LHM.utils.hf_hub import wrap_model_hub
from LHM.models.modeling_human_lrm import ModelHumanLRM


from LHM.utils.face_detector import FaceDetector
import torchvision.transforms.functional as F

import cv2
from peft import LoraConfig, get_peft_model


logger = get_logger(__name__)



@REGISTRY_RUNNERS.register('train.human_lrm')
class HumanLRMTrainer(Trainer):

    EXP_TYPE: str = "human_lrm_sapdino_bh_sd3_5"

    def __init__(self):
        super().__init__()
        
        self.model = self._build_model(self.cfg)
        self.optimizer = self._build_optimizer(self.model, self.cfg)

        self.train_loader, self.val_loader = self._build_dataloader(self.cfg)
        self.scheduler = self._build_scheduler(self.optimizer, self.cfg)
        self.pixel_loss_fn, self.perceptual_loss_fn, self.tv_loss_fn, self.asap_loss_fn, self.acap_loss_fn,       self.lap_loss_fn, self.glue_loss_fn = self._build_loss_fn(self.cfg)
    
        

        # qw00n; preprocessed
        #self.facedetect = FaceDetector(
        #    "./pretrained_models/gagatracker/vgghead/vgg_heads_l.trcd",
        #    device=self.accelerator.device,
        #) 

    # --------------------------------------------------------------------------
    # Initializers
    # --------------------------------------------------------------------------
    # qw00n; from pretrained OR from scratch / submission version : post-dynamic-layers
    def _build_model(self, cfg):
        from LHM.models import model_dict

        # start from pretrained LHM
        hf_model_cls = wrap_model_hub(model_dict[self.EXP_TYPE])
        
        model = hf_model_cls.from_pretrained(cfg.model_name)

        # qw00n; LoRA finetuning
        print('[VCAI] ======== Build LoRA Model ========')
        lora_target_modules_list = [
            "to_q", "to_k", "to_v",
            "add_q_proj", "add_k_proj", "add_v_proj",
            "to_add_out",
            "proj",
            "linear"
        ]
        modules_to_save = [
            f"transformer.dynamic_layers.{i}.dynamic_dit"
            for i, layer in enumerate(model.transformer.dynamic_layers)
            if hasattr(layer, 'dynamic_dit') 
        ]
        
        modules_to_save.append("fine_encoder")
        modules_to_save.append("encoder")
        
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=lora_target_modules_list,  
            lora_dropout=0.1,
            bias="none",
            modules_to_save=modules_to_save,
        )

        lora_model = get_peft_model(model, lora_config)

        print("[VCAI] Re-freezing modules intended to be fully frozen...")
        modules_to_force_freeze = ['fine_encoder', 'encoder']
        for name, param in lora_model.named_parameters():
            for module_name in modules_to_force_freeze:
                if f'base_model.model.{module_name}' in name:
                    param.requires_grad = False
                    break 

        trainable_modules_names = [
            'norm_layer',
            'motion_projection',
            'renderer' # gs decoder
        ]
        
        for name, module in lora_model.named_modules():
            if any(module_name in name for module_name in trainable_modules_names):
                for param in module.parameters():
                    param.requires_grad = True

        lora_model.print_trainable_parameters()
        
        print("\n[VCAI] ======== Trainable Parameter Breakdown ========")
        
        trainable_param_summary = {
            "LoRA Adapters": 0,
            "Dynamic DiT (Full)": 0,
            "Norm Layers": 0,
            "Motion Projection": 0,
            "Renderer": 0,
            "Other Trainable": 0
        }
        total_trainable_check = 0

        for name, param in lora_model.named_parameters():
            if param.requires_grad:
                num_params = param.numel()
                total_trainable_check += num_params
                
                if 'lora_' in name:
                    trainable_param_summary["LoRA Adapters"] += num_params
                elif 'dynamic_dit' in name:
                    # dynamic_dit 모듈 (LoRA가 아닌 full parameter)
                    trainable_param_summary["Dynamic DiT (Full)"] += num_params
                elif 'norm_layer' in name:
                    trainable_param_summary["Norm Layers"] += num_params
                elif 'motion_projection' in name:
                    trainable_param_summary["Motion Projection"] += num_params
                elif 'renderer' in name:
                    trainable_param_summary["Renderer"] += num_params
                else:
                    # 위 분류에 속하지 않는 나머지 (예: LoRA의 bias 등)
                    trainable_param_summary["Other Trainable"] += num_params

        for module_name, count in trainable_param_summary.items():
            if count > 0: 
                print(f"  > {module_name:<20}: {count:>12,}") 
        
        print("--------------------------------------------------")
        print(f"  > {'Total (from breakdown)':<20}: {total_trainable_check:>12,}")
        print("[VCAI] ==================================================\n")
        

        return lora_model
        
    # qw00n; copied from OpenLRM
    def _build_optimizer(self, model: nn.Module, cfg):
        decay_params, no_decay_params = [], []

        # add all bias and LayerNorm params to no_decay_params
        for name, module in model.named_modules():
            if isinstance(module, nn.LayerNorm):
                no_decay_params.extend([p for p in module.parameters()])
            elif hasattr(module, 'bias') and module.bias is not None:
                no_decay_params.append(module.bias)

        # add remaining parameters to decay_params
        _no_decay_ids = set(map(id, no_decay_params))
        decay_params = [p for p in model.parameters() if id(p) not in _no_decay_ids]

        # filter out parameters with no grad
        decay_params = list(filter(lambda p: p.requires_grad, decay_params))
        no_decay_params = list(filter(lambda p: p.requires_grad, no_decay_params))

        # monitor this to make sure we don't miss any parameters
        logger.info("======== Weight Decay Parameters ========")
        logger.info(f"Total: {len(decay_params)}")
        logger.info("======== No Weight Decay Parameters ========")
        logger.info(f"Total: {len(no_decay_params)}")

        # Optimizer
        opt_groups = [
            {'params': decay_params, 'weight_decay': cfg.train.optim.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        optimizer = torch.optim.AdamW(
            opt_groups,
            lr=cfg.train.optim.lr,
            betas=(cfg.train.optim.beta1, cfg.train.optim.beta2),
        )

        return optimizer

    # qw00n; copied from OpenLRM
    def _build_dataloader(self, cfg):
        from LHM.datasets import MixerDataset

        # build dataset class
        train_dataset = MixerDataset(
            split="train",
            subsets=cfg.dataset.subsets,
            sample_side_views=cfg.dataset.sample_side_views,
            render_image_res_low=cfg.dataset.render_image.low,
            render_image_res_high=cfg.dataset.render_image.high,
            render_region_size=cfg.dataset.render_image.region,
            source_image_res=cfg.dataset.source_image_res,
            #normalize_camera=cfg.dataset.normalize_camera,
            #normed_dist_to_center=cfg.dataset.normed_dist_to_center,
        )
        val_dataset = MixerDataset(
            split="val",
            subsets=cfg.dataset.subsets,
            sample_side_views=cfg.dataset.sample_side_views,
            render_image_res_low=cfg.dataset.render_image.low,
            render_image_res_high=cfg.dataset.render_image.high,
            render_region_size=cfg.dataset.render_image.region,
            source_image_res=cfg.dataset.source_image_res,
            #normalize_camera=cfg.dataset.normalize_camera,
            #normed_dist_to_center=cfg.dataset.normed_dist_to_center,
        )

        # build dataloader
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=cfg.dataset.num_train_workers,
            pin_memory=cfg.dataset.pin_mem,
            persistent_workers=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.val.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=cfg.dataset.num_val_workers,
            pin_memory=cfg.dataset.pin_mem,
            persistent_workers=False,
        )

        return train_loader, val_loader

    # qw00n; copied from OpenLRM
    def _build_scheduler(self, optimizer, cfg):
        local_batches_per_epoch = math.floor(len(self.train_loader) / self.accelerator.num_processes)
        total_global_batches = cfg.train.epochs * math.ceil(local_batches_per_epoch / self.cfg.train.accum_steps)
        effective_warmup_iters = cfg.train.scheduler.warmup_real_iters
        logger.debug(f"======== Scheduler effective max iters: {total_global_batches} ========")
        logger.debug(f"======== Scheduler effective warmup iters: {effective_warmup_iters} ========")
        
        
        if cfg.train.scheduler.type == 'cosine':
            from LHM.utils.scheduler import CosineWarmupScheduler
            scheduler = CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_iters=effective_warmup_iters,
                max_iters=total_global_batches,
            )
        else:
            raise NotImplementedError(f"Scheduler type {cfg.train.scheduler.type} not implemented")
        return scheduler
        
    def _build_loss_fn(self, cfg):
        from LHM.losses import ASAP_Loss, ACAP_Loss, LPIPSLoss, PixelLoss, TVLoss, WanVAELoss, LaplacianRegLoss, FeatureMatchingLoss
        pixel_loss_fn = PixelLoss(option='l1')
        with self.accelerator.main_process_first():
            perceptual_loss_fn = LPIPSLoss(device=self.device, prefech=True)
            #video_vae_loss_fn = WanVAELoss(device=self.device)
            glue_loss_fn = FeatureMatchingLoss(device=self.device)
        tv_loss_fn = TVLoss()

        # qw00n;
        asap_loss_fn, acap_loss_fn = ASAP_Loss(), ACAP_Loss()

        lap_loss_fn = LaplacianRegLoss(self.model.renderer.smplx_model.dense_pts)

        return pixel_loss_fn, perceptual_loss_fn, tv_loss_fn, asap_loss_fn, acap_loss_fn,        lap_loss_fn, glue_loss_fn

    def register_hooks(self):
        pass


    # implemented by vcai
    @torch.no_grad()
    def crop_face(self, image_tensor):
        bboxes = self.facedetect((image_tensor * 255.0).to(torch.uint8))
        
        cropped_heads_np = []
        
        for i in range(image_tensor.shape[0]):
            single_image = image_tensor[i]
            single_bbox = bboxes[i]
            print(single_bbox)

            head_rgb_tensor = single_image[:, int(single_bbox[1]) : int(single_bbox[3]), int(single_bbox[0]) : int(single_bbox[2])]
            
            head_rgb_numpy = (head_rgb_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            try:
                resized_head_rgb = cv2.resize(
                    head_rgb_numpy,
                    dsize=(self.cfg.dataset.src_head_size, self.cfg.dataset.src_head_size),
                    interpolation=cv2.INTER_AREA,
                )
            except Exception as e:
                print(f"[qw00n] ERROR: Failed to resize cropped face: {e}")
                resized_head_rgb = np.zeros(
                    (self.cfg.dataset.src_head_size, self.cfg.dataset.src_head_size, 3), dtype=np.uint8
                )
            
            cropped_heads_np.append(resized_head_rgb)

        return np.stack(cropped_heads_np, axis=0)
    

    # implemented by vcai
    def forward_train(self, data):
        B, T, V = data['gt_img'].shape[:3]

        # (B, T, V, ...) -> (B*T, V, ...)
        # (B, T, ...) -> (B*T, ...)
        for key, value in data.items():
            if not isinstance(value, (torch.Tensor, np.ndarray)):
                continue
            
            s = value.shape
            
            # (B, T, V, ...) -> (B*T, V, ...)
            if len(s) >= 3 and s[0] == B and s[1] == T and s[2] == V:
                new_shape = (-1, V, *s[3:])
                if isinstance(value, torch.Tensor):
                    data[key] = value.reshape(new_shape)
                else: 
                    data[key] = value.reshape(new_shape)
            
            # (B, T, ...) -> (B*T, ...)
            elif len(s) >= 2 and s[0] == B and s[1] == T:
                new_shape = (-1, *s[2:])
                if isinstance(value, torch.Tensor):
                    data[key] = value.reshape(new_shape)
                else: 
                    data[key] = value.reshape(new_shape)

        
        image = data['img'] * data['mask'] + (1.0 - data['mask'])

        outputs = self.model(
            image=image, # [B*T, 1, 3, 1024, 856]
            head_image=data['face_img'].detach(),     # [B*T, 1, 3, 112, 112]
            render_c2ws=data['extri'], # [B*T, V, 4, 4]
            render_intrs=data['intri'], # [B*T, V, 4, 4]
            
            render_bg_colors=torch.ones((image.shape[0], image.shape[1], 3), device=image.device),
            
            smplx_params = {
                'root_pose':  data['target_pose'][:,0:1],         
                'body_pose':  data['target_pose'][:,1:22].unsqueeze(1), # [B*T, 1, 21, 3]
                'lhand_pose': data['target_pose'][:,22:37].unsqueeze(1),     
                'rhand_pose': data['target_pose'][:,37:52].unsqueeze(1),

                'jaw_pose':   data['target_pose'][:,52:53],       
                'leye_pose':  data['target_pose'][:,53:54],       
                'reye_pose':  data['target_pose'][:,54:55],       
   
                'expr':       data['target_exprs'], 
                'trans':      data['target_transl'].unsqueeze(1),     # [B*T, 1, 3]
                
                'betas':      data['target_betas'],      
            },
            motion_history = data['motion_history'] # [B*T, t, K, 3]
        )
        # ['latent_points', 'offset_output', 'scaling_output', 'comp_rgb', 'comp_rgb_bg', 'comp_mask', 'comp_depth', '3dgs']

        return outputs
    
    # implemented by vcai   
    def train_epoch(self, pbar: tqdm, loader: torch.utils.data.DataLoader, profiler: torch.profiler.profile):
        #self.model.train()
        
        local_step_losses = []
        global_step_losses = []

        logger.debug(f"======== Starting epoch {self.current_epoch} ========")

        # batch
        for data in loader:
            logger.debug(f"======== Starting global step {self.global_step} ========")
            with self.accelerator.accumulate(self.model):
                B, T, V, C, H, W = data['gt_img'].shape
                
                # 1. Forward & calc loss
                outputs = self.forward_train(data) # in this function, B, F -> B*F
                
                gt_image = data['gt_img'] * data['gt_mask'] + (1.0 - data['gt_mask'])

                rgb_loss = self.pixel_loss_fn(outputs['comp_rgb'], gt_image)

                mask_loss = self.pixel_loss_fn(outputs['comp_mask'], data['gt_mask'], is_mask=True) * 0.5 * 1.5 # B*F, V, 1, H, W
                
                lpips_loss = self.perceptual_loss_fn(outputs['comp_rgb'], gt_image)


                acap_loss = self.acap_loss_fn(outputs['hand_offset_output']) * 50

                mask_dist_loss = self.pixel_loss_fn(outputs['dist_comp_mask'], data['gt_mask'], is_mask=True) * 0.5 * 1.5


                lap_scale_loss = self.lap_loss_fn(outputs['scaling_output'], None).mean() * 10000 * 5
                lap_rotation_loss = self.lap_loss_fn(outputs['rotation_output'], None).mean() * 100

                # qw00n; only for hand, face
                lap_mean_loss = self.lap_loss_fn(outputs['query_points'] + outputs['offset_output'], outputs['query_points'])[:, outputs['lap_mean_mask']].mean() * 1000

                # qw00n;
                #video_vae_loss = self.video_vae_loss_fn(outputs['comp_rgb'].reshape(B, T, V, C, H, W), gt_image.reshape(B, T, V, C, H, W)) * 0.5 * 0
                    
                # qw00n;
                if self.global_step > 20000:
                    glue_loss = self.glue_loss_fn(outputs['comp_rgb'], gt_image, outputs['proj_comp_rgb'], outputs['proj_comp_mask']) * 0.1
                else:
                    glue_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

                loss = rgb_loss + mask_loss + lpips_loss + acap_loss + mask_dist_loss       + lap_scale_loss + lap_rotation_loss + lap_mean_loss + glue_loss
               
                
                # 2. backward
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients and self.cfg.train.optim.clip_grad_norm > 0.:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.cfg.train.optim.clip_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # track local losses
                local_step_losses.append(torch.stack([
                    _loss.detach() if _loss is not None else torch.tensor(float('nan'), device=self.device)
                    for _loss in [loss, rgb_loss, mask_loss, lpips_loss, acap_loss, mask_dist_loss,       lap_scale_loss, lap_rotation_loss, lap_mean_loss, glue_loss] # TODO add more losses
                ]))

            # track global step
            if self.accelerator.sync_gradients:
                profiler.step()
                self.scheduler.step()
                logger.debug(f"======== Scheduler step ========")
                self.global_step += 1
                global_step_loss = self.accelerator.gather(torch.stack(local_step_losses)).mean(dim=0).cpu()
                loss, loss_pixel, loss_mask, loss_lpips, loss_acap, loss_mask_dist,         loss_lap_scale, loss_lap_rotation, loss_lap_mean, loss_glue = global_step_loss.unbind()
                loss_kwargs = {
                    'loss': loss.item(),
                    'loss_pixel': loss_pixel.item(),
                    'loss_mask': loss_mask.item(),
                    'loss_lpips': loss_lpips.item(),
                    'loss_acap': loss_acap.item(),
                    'loss_mask_dist': loss_mask.item(),
                    #'loss_video_vae': loss_video_vae.item(),
                    'loss_lap_scale': loss_lap_scale.item(),
                    'loss_lap_rotation': loss_lap_rotation.item(),
                    'loss_lap_mean': loss_lap_mean.item(),
                    'loss_glue': loss_glue.item()
                }
                self.log_scalar_kwargs(
                    step=self.global_step, split='train',
                    **loss_kwargs
                )
                self.log_optimizer(step=self.global_step, attrs=['lr'], group_ids=[0, 1])
                local_step_losses = []
                global_step_losses.append(global_step_loss)

                # manage display
                pbar.update(1)
                description = {
                    **loss_kwargs,
                    'lr': self.optimizer.param_groups[0]['lr'],
                }
                description = '[TRAIN STEP]' + \
                    ', '.join(f'{k}={tqdm.format_num(v)}' for k, v in description.items() if not math.isnan(v))
                pbar.set_description(description)

                # periodic actions
                if self.global_step % self.cfg.saver.checkpoint_global_steps == 0:
                    self.save_checkpoint()
                #if self.global_step % self.cfg.val.global_step_period == 0:
                #    self.evaluate()
                #    self.model.train()
                if self.global_step % self.cfg.logger.image_monitor.train_global_steps == 0:

                    self.log_image_monitor(
                        step=self.global_step, split='train',
                        #renders_static=outputs['comp_rgb'].detach()[:self.cfg.logger.image_monitor.samples_per_log].cpu(),
                        renders=outputs['comp_rgb'].detach()[:self.cfg.logger.image_monitor.samples_per_log].unsqueeze(0).cpu(),
                        gts=gt_image[:self.cfg.logger.image_monitor.samples_per_log].unsqueeze(0).cpu(),
                        #residual=residual.detach()[:self.cfg.logger.image_monitor.samples_per_log].cpu()
                    )

                # progress control
                if self.global_step >= self.N_max_global_steps:
                    self.accelerator.set_trigger()
                    break

        # track epoch
        self.current_epoch += 1
        epoch_losses = torch.stack(global_step_losses).mean(dim=0)
        epoch_loss, epoch_loss_pixel, epoch_loss_mask, epoch_loss_lpips, epoch_loss_acap, epoch_loss_mask_dist,      epoch_loss_lap_scale, epoch_loss_lap_rotation, epoch_loss_lap_mean, epoch_loss_glue = epoch_losses.unbind()
        epoch_loss_dict = {
            'loss': epoch_loss.item(),
            'loss_pixel': epoch_loss_pixel.item(),
            'loss_mask': epoch_loss_mask.item(),
            'loss_lpips': epoch_loss_lpips.item(),
            'loss_acap': epoch_loss_acap.item(),
            'loss_mask_dist': epoch_loss_mask_dist.item(),
            #'loss_video_vae': epoch_loss_video_vae.item(),
            'loss_lap_scale': epoch_loss_lap_scale.item(),
            'loss_lap_rotation': epoch_loss_lap_rotation.item(),
            'loss_lap_mean': epoch_loss_lap_mean.item(),
            'loss_glue': epoch_loss_glue.item()
        }
        self.log_scalar_kwargs(
            epoch=self.current_epoch, split='train',
            **epoch_loss_dict,
        )
        logger.info(
            f'[TRAIN EPOCH] {self.current_epoch}/{self.cfg.train.epochs}: ' + \
                ', '.join(f'{k}={tqdm.format_num(v)}' for k, v in epoch_loss_dict.items() if not math.isnan(v))
        )

    def train(self):
        # qw00n; for resume
        starting_local_step_in_epoch = self.global_step_in_epoch * self.cfg.train.accum_steps
        skipped_loader = self.accelerator.skip_first_batches(self.train_loader, starting_local_step_in_epoch)
        logger.info(f"======== Skipped {starting_local_step_in_epoch} local batches ========")

        with tqdm(
            range(0, self.N_max_global_steps),
            initial=self.global_step,
            disable=(not self.accelerator.is_main_process),
        ) as pbar:
            
            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(
                    wait=10, warmup=10, active=100,
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(
                    self.cfg.logger.tracker_root,
                    self.cfg.experiment.parent, self.cfg.experiment.child,
                )),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) if self.cfg.logger.enable_profiler else DummyProfiler()

            # qw00n; training loop 
            with profiler:
                self.optimizer.zero_grad()

                for _ in range(self.current_epoch, self.cfg.train.epochs):
                    loader = skipped_loader or self.train_loader
                    skipped_loader = None
                    self.train_epoch(pbar=pbar, loader=loader, profiler=profiler)
                    if self.accelerator.check_trigger():
                        break

            logger.info(f"======== Training finished at global step {self.global_step} ========")

        # final checkpoint and evaluation
        self.save_checkpoint()
        self.evaluate()

    @torch.no_grad()
    @torch.compiler.disable
    def evaluate(self):
        pass

    @Trainer.control('on_main_process')
    def log_image_monitor(
        self, epoch: int = None, step: int = None, split: str = None,
        renders: torch.Tensor = None, gts: torch.Tensor = None,
    ):
        """
        Logs image grids for each item in the batch.
        
        Args:
            renders, gts: Input tensors of shape (B, T, V, C, H, W)
        """
        
        # Get dimensions
        if renders is None or gts is None:
            print("Renders or GTs tensor is missing.")
            return
            
        try:
            B, T, V, C, H, W = renders.shape
            assert gts.shape == (B, T, V, C, H, W)
        except (AttributeError, ValueError, AssertionError) as e:
            print(f"Error: Input tensors have incorrect shape. Expected (B, T, V, C, H, W).")
            print(f"Renders shape: {renders.shape if renders is not None else 'None'}")
            print(f"GTS shape: {gts.shape if gts is not None else 'None'}")
            print(f"Original error: {e}")
            return

        # How many batch items to log (e.g., log up to 4 items)
        num_samples_to_log = min(4, B)

        log_type, log_progress = self._get_str_progress(epoch, step)
        split_str = f'/{split}' if split else ''

        # Iterate over each item in the batch to create a separate grid
        for i in range(num_samples_to_log):
            # Get the i-th item from the batch.
            # Shapes are now (T, V, C, H, W)
            render_item = renders[i]
            gt_item = gts[i]

            # --- 1. Renders Grid ---
            # Flatten T and V dimensions to (T*V, C, H, W)
            # This orders images as: [t0_v0, t0_v1, ..., t0_vV, t1_v0, t1_v1, ...]
            render_flat = render_item.reshape(T * V, C, H, W)
            
            # Create a grid. nrow=V means V columns.
            # This results in a (T_rows, V_cols) grid.
            render_grid = make_grid(render_flat, nrow=V)

            # --- 2. GTs Grid ---
            # Do the same for the ground truth images
            gt_flat = gt_item.reshape(T * V, C, H, W)
            gt_grid = make_grid(gt_flat, nrow=V)

            # --- 3. Merged Grid (Render/GT side-by-side) ---
            # We want a grid where each row is a time step T,
            # and columns are (render_v0, gt_v0, render_v1, gt_v1, ...)
            
            # Stack along a new dimension (dim=2) -> (T, V, 2, C, H, W)
            # This groups (render, gt) pairs together
            merged_item = torch.stack([render_item, gt_item], dim=2)
            
            # Reshape to flatten T, V, and the new (render/gt) dim
            # Shape becomes (T * V * 2, C, H, W)
            merged_flat = merged_item.reshape(T * V * 2, C, H, W)
            
            # Make a grid with T rows and (V*2) columns
            # This arranges the flat tensor into:
            # [r0,g0, r1,g1, ...]
            # [r0,g0, r1,g1, ...]
            merged_grid = make_grid(merged_flat, nrow=V * 2)

            # Log this item's grids
            # We add a unique key for each batch item (e.g., /item_0, /item_1)
            # .unsqueeze(0) adds a batch dim: [1, C, H_grid, W_grid]
            # This is often required by logging libraries like TensorBoard
            self.log_images({
                f'Images_split{split_str}/rendered': render_grid.unsqueeze(0),
                f'Images_split{split_str}/gt': gt_grid.unsqueeze(0),
                f'Images_merged{split_str}': merged_grid.unsqueeze(0),
            }, log_progress)