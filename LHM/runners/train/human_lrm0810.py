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
        self.pixel_loss_fn, self.perceptual_loss_fn, self.tv_loss_fn, self.asap_loss_fn, self.acap_loss_fn = self._build_loss_fn(self.cfg)
    
        # qw00n; TODO: preprocess
        self.facedetect = FaceDetector(
            "./pretrained_models/gagatracker/vgghead/vgg_heads_l.trcd",
            device=self.accelerator.device,
        ) 

    # --------------------------------------------------------------------------
    # Initializers
    # --------------------------------------------------------------------------
    # qw00n; from pretrained OR from scratch
    def _build_model(self, cfg):
        from LHM.models import model_dict

        # start from pretrained LHM
        hf_model_cls = wrap_model_hub(model_dict[self.EXP_TYPE])
        
        model = hf_model_cls.from_pretrained(cfg.model_name)
        return model
        
        '''
        # if from scratch ?
        model = model_dict[self.EXP_TYPE](**cfg.model)
        return model
        '''

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
        
    # qw00n; have to check Heuristic ASAP_Loss, ACAP_Loss
    # qw00n; have to implement additional loss func if needed
    def _build_loss_fn(self, cfg):
        from LHM.losses import ASAP_Loss, ACAP_Loss, LPIPSLoss, PixelLoss, TVLoss
        pixel_loss_fn = PixelLoss()
        with self.accelerator.main_process_first():
            perceptual_loss_fn = LPIPSLoss(device=self.device, prefech=True)
        tv_loss_fn = TVLoss()

        # qw00n;
        asap_loss_fn, acap_loss_fn = ASAP_Loss(), ACAP_Loss()

        return pixel_loss_fn, perceptual_loss_fn, tv_loss_fn, asap_loss_fn, acap_loss_fn

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
            #print(single_bbox)

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
    def forward_single(self, data):
        # qw00n; 1. mask the image using the preprocessed one
        image = data['img'] * data['mask'] + (1.0 - data['mask'])

        # qw00n; 2. crop the head (TODO: SHOULD BE PREPROCESSED)
        try:
            src_head_rgb = self.crop_face(data['img'])
        except:
            print("w/o head input!")
            exit()
            src_head_rgb = np.zeros((data['mask'].shape[0], 112, 112, 3), dtype=np.uint8)
        src_head_rgb = torch.from_numpy(src_head_rgb).float().permute(0, 3, 1, 2)/ 255.0
        
        # qw00n; 3. forward
        outputs = self.model(
            image=image.unsqueeze(1), # [B, 1, 3, 1024, 856]
            head_image=src_head_rgb.unsqueeze(1).detach(), # [B, 1, 3, 112, 112]
            render_c2ws=data['extri'], 
            render_intrs=data['intri'],
            render_bg_colors=torch.ones((image.shape[0], 1, 3)),
            smplx_params = {
                'root_pose':  data['target_pose'][:,0:1],         
                'body_pose':  data['target_pose'][:,1:22].unsqueeze(1),       
                'jaw_pose':   data['target_pose'][:,22:23],       
                'leye_pose':  data['target_pose'][:,23:24],       
                'reye_pose':  data['target_pose'][:,24:25],       
                'lhand_pose': data['target_pose'][:,25:40].unsqueeze(1),     
                'rhand_pose': data['target_pose'][:,40:55].unsqueeze(1),      
                
                'expr':       torch.zeros_like(data['target_betas']), 
                'trans':      data['target_transl'].unsqueeze(1),     # [B, 1, 3]
                
                'betas':      data['target_betas'],      
            }
        )
        # ['latent_points', 'offset_output', 'scaling_output', 'comp_rgb', 'comp_rgb_bg', 'comp_mask', 'comp_depth', '3dgs']

        # outputs['latent_points'].shape, outputs['offset_output'].shape, outputs['scaling_output'].shape : B, 40000, 1024  B, 40000, 3
        # outputs['comp_rgb'].shape -> B, 1, 3, H, W

        # debug
        '''
        single_rgb_image = outputs['comp_rgb'][2, 0]
        rgb_image_np = (single_rgb_image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        rgb_image_bgr = cv2.cvtColor(rgb_image_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite("debug_rgb.png", rgb_image_bgr)
        print("RGB Saved: debug_rgb.png")
        '''
        
        return outputs
    
    # implemented by vcai   
    def train_epoch(self, pbar: tqdm, loader: torch.utils.data.DataLoader, profiler: torch.profiler.profile):
        self.model.train()
        
        local_step_losses = []
        global_step_losses = []

        logger.debug(f"======== Starting epoch {self.current_epoch} ========")

        # batch
        for data in loader:
            logger.debug(f"======== Starting global step {self.global_step} ========")
            with self.accelerator.accumulate(self.model):
                # 1. Forward & calc loss
                outputs = self.forward_single(data)

                # TODO: collect all loss calculations into unified function
                gt_image = data['gt_img'] * data['gt_mask'] + (1.0 - data['gt_mask'])
                rgb_loss = self.pixel_loss_fn(outputs['comp_rgb'], (gt_image).unsqueeze(1))

                mask_loss = self.pixel_loss_fn(outputs['comp_mask'], data['gt_mask'].unsqueeze(1)) * 0.5
                
                lpips_loss = self.perceptual_loss_fn(outputs['comp_rgb'], (gt_image).unsqueeze(1))

                acap_loss = self.acap_loss_fn(outputs['offset_output']) * 50

                loss = rgb_loss + mask_loss + lpips_loss + acap_loss

                # 2. backward
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients and self.cfg.train.optim.clip_grad_norm > 0.:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.cfg.train.optim.clip_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # track local losses
                local_step_losses.append(torch.stack([
                    _loss.detach() if _loss is not None else torch.tensor(float('nan'), device=self.device)
                    for _loss in [loss, rgb_loss, mask_loss, lpips_loss, acap_loss] # TODO add more losses
                ]))

            # track global step
            if self.accelerator.sync_gradients:
                profiler.step()
                self.scheduler.step()
                logger.debug(f"======== Scheduler step ========")
                self.global_step += 1
                global_step_loss = self.accelerator.gather(torch.stack(local_step_losses)).mean(dim=0).cpu()
                loss, loss_pixel, loss_mask, loss_lpips, loss_acap = global_step_loss.unbind()
                loss_kwargs = {
                    'loss': loss.item(),
                    'loss_pixel': loss_pixel.item(),
                    'loss_mask': loss_mask.item(),
                    'loss_lpips': loss_lpips.item(),
                    'loss_acap': loss_acap.item(),
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
                if self.global_step % self.cfg.val.global_step_period == 0:
                    self.evaluate()
                    self.model.train()
                if self.global_step % self.cfg.logger.image_monitor.train_global_steps == 0:
                    self.log_image_monitor(
                        step=self.global_step, split='train',
                        renders=outputs['comp_rgb'].detach()[:self.cfg.logger.image_monitor.samples_per_log].cpu(),
                        gts=data['gt_img'].unsqueeze(1)[:self.cfg.logger.image_monitor.samples_per_log].cpu(),
                    )

                # progress control
                if self.global_step >= self.N_max_global_steps:
                    self.accelerator.set_trigger()
                    break

        # track epoch
        self.current_epoch += 1
        epoch_losses = torch.stack(global_step_losses).mean(dim=0)
        epoch_loss, epoch_loss_pixel, epoch_loss_mask, epoch_loss_lpips, epoch_loss_acap = epoch_losses.unbind()
        epoch_loss_dict = {
            'loss': epoch_loss.item(),
            'loss_pixel': epoch_loss_pixel.item(),
            'loss_mask': epoch_loss_mask.item(),
            'loss_lpips': epoch_loss_lpips.item(),
            'loss_acap': epoch_loss_acap.item(),
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
        M = renders.shape[1]
        merged = torch.stack([renders, gts], dim=1)[0].view(-1, *renders.shape[2:])
        renders, gts = renders.reshape(-1, *renders.shape[2:]), gts.reshape(-1, *gts.shape[2:])
        renders, gts, merged = make_grid(renders, nrow=M), make_grid(gts, nrow=M), make_grid(merged, nrow=M)
        log_type, log_progress = self._get_str_progress(epoch, step)
        split = f'/{split}' if split else ''
        self.log_images({
            f'Images_split{split}/rendered': renders.unsqueeze(0),
            f'Images_split{split}/gt': gts.unsqueeze(0),
            f'Images_merged{split}': merged.unsqueeze(0),
        }, log_progress)