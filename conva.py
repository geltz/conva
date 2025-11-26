import gc
import sys
import os
import math
import random
import time
import glob
import re
import copy
from typing import List, Optional
import importlib.util

import torch
import torch.nn as nn
import torch.nn.functional as F

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QFileDialog, QFrame, QDoubleSpinBox,
    QGraphicsDropShadowEffect, QSlider, QSplitter, QSplitterHandle,
    QComboBox, QPlainTextEdit, QSizePolicy, QCheckBox, QGraphicsOpacityEffect
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRectF, QPointF, QTimer
from PyQt6.QtGui import (
    QColor, QFont, QPainter, QPen, QBrush, QPainterPath, 
    QPixmap, QIcon, QRadialGradient
)

from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from safetensors.torch import load_file, save_file
from PIL import Image
from torchvision import transforms

def apply_ggpo_perturbation(network, sigma=0.03, beta=0.01):
    """
    Applies Gradient-Guided Perturbation Optimization (GGPO).
    Based on arXiv:2502.14538v1, Eq 6 & 8.
    Calculates row-wise noise based on weight and gradient norms.
    """
    with torch.no_grad():
        for param in network.parameters():
            if param.requires_grad and param.grad is not None:
                # Normalize shapes for convolution vs linear
                # If 4D (Conv2d), flatten to [out, in_pixels] for row-wise calc
                original_shape = param.shape
                if param.ndim > 2:
                    flat_param = param.view(param.shape[0], -1)
                    flat_grad = param.grad.view(param.shape[0], -1)
                else:
                    flat_param = param
                    flat_grad = param.grad

                n = flat_param.shape[1] # Input dimension

                # Eq 7 & 8: Compute squared norms per row (filter)
                w_norm_sq = torch.sum(flat_param ** 2, dim=1, keepdim=True)
                g_norm_sq = torch.sum(flat_grad ** 2, dim=1, keepdim=True)

                # Eq 6: Calculate covariance/variance scale
                # Variance = (sigma^2 / n) * (||W||^2 + beta * ||G||^2)
                variance = (sigma ** 2 / n) * (w_norm_sq + beta * g_norm_sq)
                std_dev = torch.sqrt(variance + 1e-10) # 1e-10 for numerical stability

                # Generate noise and apply
                epsilon = torch.randn_like(flat_param) * std_dev
                
                # Add to weights in place
                flat_param.add_(epsilon)

class WarmupCosineSchedule:
    """
    Warmup with cubic Bezier, then cosine decay.
    All in normalized [0, 1] LR space, then scaled to [lr_min, lr_max].
    """
    def __init__(
        self,
        total_steps: int,
        lr_max: float,
        lr_min: float = 0.0,
        warmup_fraction: float = 0.1,
        bezier_p1: float = 0.15,
        bezier_p2: float = 0.85,
    ):
        self.total_steps = max(1, int(total_steps))
        self.lr_max = float(lr_max)
        self.lr_min = float(lr_min)
        self.warmup_fraction = max(0.0, min(0.5, warmup_fraction))  # cap at 50%
        self.warmup_steps = max(1, int(self.total_steps * self.warmup_fraction))
        self.decay_steps = max(1, self.total_steps - self.warmup_steps)

        # Bezier control points in normalized LR space
        self.b0 = 0.0
        self.b1 = float(bezier_p1)
        self.b2 = float(bezier_p2)
        self.b3 = 1.0

    def _bezier_cubic(self, u: float) -> float:
        """Cubic Bezier in 1D."""
        u = max(0.0, min(1.0, u))
        v = 1.0 - u
        return (
            v * v * v * self.b0
            + 3.0 * v * v * u * self.b1
            + 3.0 * v * u * u * self.b2
            + u * u * u * self.b3
        )

    def lr_at(self, step: float) -> float:
        """
        Continuous step input is allowed. Values outside [0, total_steps]
        are clamped.
        """
        if self.lr_max <= self.lr_min:
            return self.lr_min

        step = max(0.0, min(float(step), float(self.total_steps)))

        if step <= 0.0:
            norm = 0.0
        elif step >= self.total_steps:
            norm = 0.0  # end at lr_min
        elif step < self.warmup_steps:
            # Bezier warmup
            u = step / float(self.warmup_steps)
            norm = self._bezier_cubic(u)  # in [0, 1]
        else:
            # Cosine decay
            t = (step - self.warmup_steps) / float(max(1, self.decay_steps))
            # standard half cosine, t in [0,1]
            norm = 0.5 * (1.0 + math.cos(math.pi * t))

        return self.lr_min + norm * (self.lr_max - self.lr_min)

from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineLRScheduler(_LRScheduler):
    """
    Torch scheduler wrapper around WarmupCosineSchedule.
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_fraction: float = 0.1,
        lr_min: float = 0.0,
        last_epoch: int = -1,
    ):
        self.total_steps = max(1, int(total_steps))
        self.warmup_fraction = warmup_fraction
        self.lr_min = lr_min

        # --------------------------------------------------------
        # IMPORTANT: build schedules BEFORE calling super().__init__
        # --------------------------------------------------------
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.schedules = [
            WarmupCosineSchedule(
                total_steps=self.total_steps,
                lr_max=base_lr,
                lr_min=self.lr_min,
                warmup_fraction=self.warmup_fraction,
            )
            for base_lr in self.base_lrs
        ]

        # Now safe to call super(), which will run _initial_step()
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        # last_epoch is step index
        step = self.last_epoch
        return [sched.lr_at(step) for sched in self.schedules]

# ==============================================================================
# 1. EMBEDDED RAVEN OPTIMIZER
# ==============================================================================
class RavenAdamW(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.05,
        eps: float = 1e-8,
        debias_strength: float = 1.0,
        use_grad_centralization: bool = False,
        gc_alpha: float = 1.0,
    ):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps, 
                        debias_strength=debias_strength, use_grad_centralization=use_grad_centralization, 
                        gc_alpha=gc_alpha)
        super(RavenAdamW, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            wd = group['weight_decay']
            eps = group['eps']

            for p in group["params"]:
                if p.grad is None: continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                
                if wd != 0: p.mul_(1 - lr * wd)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1
                
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

class TimestepSamplingConfig:
    METHODS = [
        "Random Integer",
        "Uniform Continuous",
        "Uniform LogSNR",
        "Logit Normal",
        "Dynamic Windowed"
    ]
    
    def __init__(self):
        self.method = "Random Integer"
        self.use_log_snr = False
        self.ts_min = 0
        self.ts_max = 999
        self.logit_mean = 0.0
        self.logit_std = 1.0

class SchedulerConfig:
    TYPES = [
        "DDPMScheduler",
        "DDIMScheduler", 
        "EulerDiscreteScheduler",
        "FlowMatchEulerDiscreteScheduler"
    ]
    
    PREDICTION_TYPES = [
        "epsilon",
        "v_prediction",
        "flow_matching"
    ]
    
    def __init__(self):
        self.scheduler_type = "DDPMScheduler"
        self.prediction_type = "epsilon"
        self.use_zero_terminal_snr = False
        self.flow_shift = 3.0

# ==============================================================================
# 2. INTERNAL LoCon/LoRA IMPLEMENTATION (Fixed)
# ==============================================================================
class LoConModule(nn.Module):
    def __init__(self, lora_name, org_module: nn.Module, multiplier=1.0, lora_dim=4, alpha=1):
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim
        self.multiplier = multiplier
        self.org_module = org_module 
        self.register_buffer("alpha", torch.tensor(alpha))
        self.scale = alpha / lora_dim

        if isinstance(org_module, nn.Conv2d):
            self.is_conv = True
            in_dim = org_module.in_channels
            k_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            out_dim = org_module.out_channels
            self.lora_down = nn.Conv2d(in_dim, lora_dim, k_size, stride, padding, bias=False)
            self.lora_up = nn.Conv2d(lora_dim, out_dim, 1, 1, bias=False)
        elif isinstance(org_module, nn.Linear):
            self.is_conv = False
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = nn.Linear(in_dim, lora_dim, bias=False)
            self.lora_up = nn.Linear(lora_dim, out_dim, bias=False)
        else:
            raise ValueError("Only Linear and Conv2d are supported")

        # Initialize with same dtype as original
        target_dtype = org_module.weight.dtype
        self.lora_down = self.lora_down.to(dtype=target_dtype)
        self.lora_up = self.lora_up.to(dtype=target_dtype)
        
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

    def forward(self, input):
        # Standard LoRA addition
        return self.lora_up(self.lora_down(input)) * self.multiplier * self.scale

class LoConNetwork(nn.Module):
    def __init__(self, unet: nn.Module, text_encoder1: nn.Module, text_encoder2: nn.Module, 
                 multiplier=1.0, dim=4, alpha=1, conv_dim=4, conv_alpha=1):
        super().__init__()
        self.multiplier = multiplier
        self.lora_modules = nn.ModuleList()

        def create_modules(root_module):
            for name, module in root_module.named_modules():
                if module.__class__.__name__ in ["Linear", "Conv2d"]:
                    if "resnet" in name or "downsamplers" in name or "upsamplers" in name: 
                        continue 
                    
                    is_linear = isinstance(module, nn.Linear)
                    current_dim = dim if is_linear else conv_dim
                    current_alpha = alpha if is_linear else conv_alpha
                    if current_dim == 0: continue

                    lora = LoConModule(name, module, multiplier, current_dim, current_alpha)
                    self.lora_modules.append(lora)

        create_modules(unet)

    def apply_to(self, unet, te1, te2):
        for lora in self.lora_modules:
            original = lora.org_module
            if not hasattr(original, '_original_forward'):
                original._original_forward = original.forward
            
            # Simple hook: Original + LoRA
            def make_forward(lora_module, orig_forward):
                def new_forward(input):
                    return orig_forward(input) + lora_module(input)
                return new_forward
            
            original.forward = make_forward(lora, original._original_forward)

    def save_weights(self, file_path, dtype, metadata):
        state_dict = {}
        for lora in self.lora_modules:
            key_base = lora.lora_name.replace(".", "_")
            state_dict[f"{key_base}.lora_up.weight"] = lora.lora_up.weight.to(dtype)
            state_dict[f"{key_base}.lora_down.weight"] = lora.lora_down.weight.to(dtype)
            state_dict[f"{key_base}.alpha"] = lora.alpha.to(dtype)
        metadata["ss_trained_with"] = "conva"
        save_file(state_dict, file_path, metadata)

# ==============================================================================
# 3. DATASET & TRAINING PIPELINE
# ==============================================================================
class SimpleDataset(Dataset):
    def __init__(self, img_dir, size, tokenizer1, tokenizer2):
        self.img_dir = img_dir
        self.size = size
        self.tokenizer1 = tokenizer1
        self.tokenizer2 = tokenizer2

        self.image_paths = []
        raw_paths = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.PNG", "*.JPG", "*.JPEG", "*.WEBP"]:
            raw_paths.extend(glob.glob(os.path.join(img_dir, ext)))
        
        # Validation Step: Only keep images that can actually open
        for p in raw_paths:
            try:
                # fast verify without decoding pixel data
                with Image.open(p) as img:
                    img.verify() 
                self.image_paths.append(p)
            except:
                print(f"Skipping corrupt image: {p}")
        
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new("RGB", (self.size, self.size))
        
        # Caption loading
        cap_path = os.path.splitext(img_path)[0] + ".txt"
        caption = ""
        if os.path.exists(cap_path):
            with open(cap_path, "r", encoding='utf-8') as f:
                caption = f.read().strip()
        
        img_tensor = self.transform(image)
        
        # Tokenize
        def tokenize(tok, cap):
            return tok(
                cap, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
            ).input_ids[0]

        ids1 = tokenize(self.tokenizer1, caption)
        ids2 = tokenize(self.tokenizer2, caption)
        
        return {
            "pixel_values": img_tensor,
            "input_ids1": ids1,
            "input_ids2": ids2
        }

# ==============================================================================
# 4. INTERNAL TRAINER (Fixed: Quantization + Gradient Checkpointing)
# ==============================================================================
class InternalTrainer(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int, float, float)
    finished_signal = pyqtSignal()

    def __init__(self, config, ts_config, sched_config):
        super().__init__()
        self.cfg = config
        self.ts_cfg = ts_config
        self.sched_cfg = sched_config
        self.running = True

    def log(self, msg):
        self.log_signal.emit(msg)

    def run(self):
        try:
            self.train_process()
        except Exception as e:
            import traceback
            self.log(f"Critical error:\n{traceback.format_exc()}")
        finally:
            self.finished_signal.emit()

    def train_process(self):
        try:
            # 1. Device & Dtype Setup
            def choose_dtype():
                if not torch.cuda.is_available(): return torch.float32
                major, _ = torch.cuda.get_device_capability()
                return torch.bfloat16 if major >= 8 else torch.float16

            model_dtype = choose_dtype()
            precision = "bf16" if model_dtype == torch.bfloat16 else "fp16"
            
            accelerator = Accelerator(
                gradient_accumulation_steps=self.cfg['grad'],
                mixed_precision=precision
            )
            
            self.log(f"Device: {accelerator.device} | Dtype: {model_dtype}")
            self.log("Initializing Training (Gradient Checkpointing Enabled)...")

            # 2. Check for BitsAndBytes (Huge memory saver)
            try:
                import bitsandbytes as bnb
                HAS_BNB = True
                self.log("BitsAndBytes detected: 8-bit Optimizer enabled.")
            except ImportError:
                HAS_BNB = False
                self.log("BitsAndBytes not found. Using standard Optimizer.")

            # 3. Load Pipeline
            from diffusers import StableDiffusionXLPipeline
            
            self.log("Loading SDXL Pipeline...")
            # Load to CPU initially to control memory
            pipeline = StableDiffusionXLPipeline.from_single_file(
                self.cfg['model'], use_safetensors=True
            )
            
            vae = pipeline.vae
            text_encoder1 = pipeline.text_encoder
            text_encoder2 = pipeline.text_encoder_2
            tokenizer1 = pipeline.tokenizer
            tokenizer2 = pipeline.tokenizer_2
            unet = pipeline.unet
            del pipeline
            
            # 4. Prepare Components
            # Move VAE/Text Encoders to GPU in FP32/BF16
            vae.to(accelerator.device, dtype=torch.float32)
            text_encoder1.to(accelerator.device, dtype=torch.float32)
            text_encoder2.to(accelerator.device, dtype=torch.float32)
            
            vae.requires_grad_(False)
            text_encoder1.requires_grad_(False)
            text_encoder2.requires_grad_(False)
            
            # --- UNET OPTIMIZATION ---
            unet.requires_grad_(False)
            # CRITICAL: Enable Gradient Checkpointing. 
            # This trades a bit of speed for massive VRAM savings (activations).
            unet.enable_gradient_checkpointing() 
            
            # Move UNet to GPU. 
            # Note: If this OOMs, you need bitsandbytes quantization or a smaller batch size.
            unet.to(accelerator.device, dtype=model_dtype)
            
            # 5. Setup LoRA Network
            self.log("Creating LoCon network...")
            network = LoConNetwork(
                unet, text_encoder1, text_encoder2, 
                multiplier=1.0, 
                dim=self.cfg['dim'], alpha=self.cfg['alpha'],
                conv_dim=self.cfg['cdim'], conv_alpha=self.cfg['calpha']
            )
            network.apply_to(unet, text_encoder1, text_encoder2)
            
            # Move Trainable LoRA params to GPU
            network.to(accelerator.device)
            
            trainable_params = list(network.lora_modules.parameters())
            total_trainable = sum(p.numel() for p in trainable_params)
            self.log(f"Trainable params: {total_trainable/1e6:.2f}M")
            
            # 6. Optimizer
            lr = float(self.cfg['optim']['lr'])
            wd = float(self.cfg['optim']['wd'])
            betas = eval(self.cfg['optim']['betas'])
            
            if HAS_BNB:
                # Use 8-bit optimizer if available (Saves ~3GB VRAM)
                optimizer = bnb.optim.AdamW8bit(trainable_params, lr=lr, betas=betas, weight_decay=wd)
            else:
                opt_type = self.cfg['optim']['type']
                if opt_type == "raven":
                    optimizer = RavenAdamW(trainable_params, lr=lr, betas=betas, weight_decay=wd)
                else:
                    optimizer = torch.optim.AdamW(trainable_params, lr=lr, betas=betas, weight_decay=wd)

            # 7. Scheduler & Data
            total_steps = self.cfg['total_steps']
            lr_scheduler = WarmupCosineLRScheduler(optimizer, total_steps=total_steps)
            
            dataset = SimpleDataset(self.cfg['data'], self.cfg['res'], tokenizer1, tokenizer2)
            dataloader = DataLoader(dataset, batch_size=self.cfg['batch'], shuffle=True, num_workers=0)
            
            # Setup Noise Scheduler
            from diffusers import DDPMScheduler, DDIMScheduler, EulerDiscreteScheduler, FlowMatchEulerDiscreteScheduler
            SCHED_MAP = {
                "DDPMScheduler": DDPMScheduler, "DDIMScheduler": DDIMScheduler,
                "EulerDiscreteScheduler": EulerDiscreteScheduler, "FlowMatchEulerDiscreteScheduler": FlowMatchEulerDiscreteScheduler,
            }
            sched_cls = SCHED_MAP[self.sched_cfg.scheduler_type]
            
            # Configure Scheduler args
            sched_args = {}
            if self.sched_cfg.prediction_type == "flow_matching":
                 sched_args["shift"] = self.sched_cfg.flow_shift
            else:
                 sched_args["num_train_timesteps"] = 1000

            noise_scheduler = sched_cls(**sched_args)
            
            # Accelerator Prepare
            network, optimizer, dataloader = accelerator.prepare(network, optimizer, dataloader)
            
            # 8. Training Loop
            self.log("Starting training...")
            network.train()
            global_step = 0
            
            epochs = self.cfg['epoch']
            
            for epoch in range(epochs):
                if not self.running: break
                
                for step, batch in enumerate(dataloader):
                    with accelerator.accumulate(network):
                        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                            # Move Data
                            pixel_values = batch["pixel_values"].to(accelerator.device)
                            input_ids1 = batch["input_ids1"].to(accelerator.device)
                            input_ids2 = batch["input_ids2"].to(accelerator.device)
                            
                            # Encode Latents
                            # We encode in float32 for precision then cast
                            with torch.no_grad():
                                latents = vae.encode(pixel_values.to(dtype=torch.float32)).latent_dist.sample()
                                latents = latents * 0.13025
                                latents = latents.to(model_dtype)
                                
                                out1 = text_encoder1(input_ids1, output_hidden_states=True)
                                out2 = text_encoder2(input_ids2, output_hidden_states=True)
                                
                                prompt_embeds = torch.concat([out1.hidden_states[-2], out2.hidden_states[-2]], dim=-1)
                                pooled_prompt_embeds = out2.text_embeds
                                
                                # Time IDs
                                def compute_time_ids(res):
                                    add_time_ids = list((res, res) + (0, 0) + (res, res))
                                    add_time_ids = torch.tensor([add_time_ids], dtype=torch.float32)
                                    return add_time_ids.to(accelerator.device).repeat(latents.shape[0], 1)

                                add_time_ids = compute_time_ids(self.cfg['res'])
                            
                            # Noise
                            noise = torch.randn_like(latents)
                            
                            # Sample Timesteps
                            if self.ts_cfg.method == "Uniform Continuous":
                                t = torch.rand(latents.shape[0], device=latents.device)
                                timesteps = (t * (noise_scheduler.config.num_train_timesteps - 1)).long()
                            else:
                                timesteps = torch.randint(0, 1000, (latents.shape[0],), device=latents.device).long()

                            # Add Noise
                            if self.sched_cfg.prediction_type == "flow_matching":
                                t_continuous = timesteps.float() / 1000.0
                                t_expanded = t_continuous.view(-1, 1, 1, 1)
                                noisy_latents = (1 - t_expanded) * latents + t_expanded * noise
                                target = noise - latents 
                            else:
                                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                                target = noise

                            added_cond_kwargs = {
                                "text_embeds": pooled_prompt_embeds.to(model_dtype), 
                                "time_ids": add_time_ids.to(model_dtype)
                            }
                            
                            # UNet Forward
                            model_pred = unet(
                                noisy_latents, timesteps, 
                                encoder_hidden_states=prompt_embeds.to(model_dtype), 
                                added_cond_kwargs=added_cond_kwargs
                            ).sample
                            
                            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    
                    # 1. Backward Pass
                    accelerator.backward(loss)

                    # 2. GGPO Injection (New)
                    # We use default sigma=0.03, beta=0.01 from paper section 4.1
                    if self.cfg['optim'].get('use_ggpo', False):
                        apply_ggpo_perturbation(network, sigma=0.03, beta=0.01)

                    # 3. Optimizer Step
                    optimizer.step()
                    optimizer.zero_grad()
                
                if accelerator.sync_gradients:
                        lr_scheduler.step()
                        global_step += 1
                        if global_step % 10 == 0:
                            self.progress_signal.emit(global_step, total_steps, loss.item(), lr_scheduler.get_last_lr()[0])
                            self.log(f"Step {global_step} | Loss: {loss.item():.4f}")

                if not self.running: break
                
            # Save
            if self.running:
                os.makedirs(self.cfg['out'], exist_ok=True)
                timestamp = int(time.time())
                model_name = os.path.splitext(os.path.basename(self.cfg['model']))[0]
                out_path = os.path.join(self.cfg['out'], f"conva_{model_name}_{timestamp}.safetensors")
                
                metadata = {"ss_base_model": self.cfg['model'], "ss_training_steps": str(global_step)}
                network.save_weights(out_path, torch.bfloat16, metadata)
                self.log(f"Training Finished. Model saved to: {out_path}")

        except Exception as e:
            import traceback
            self.log(f"Error: {traceback.format_exc()}")
        finally:
            # Logic Change: Explicit Memory Cleanup
            self.log("Cleaning up resources...")
            
            # 1. Delete heavy references safely
            # Check if they exist in locals() before deleting to avoid UnboundLocalError
            if 'vae' in locals(): del vae
            if 'text_encoder1' in locals(): del text_encoder1
            if 'text_encoder2' in locals(): del text_encoder2
            if 'unet' in locals(): del unet
            if 'network' in locals(): del network
            if 'optimizer' in locals(): del optimizer
            if 'pipeline' in locals(): del pipeline
            
            # 2. Force Python Garbage Collection
            import gc
            gc.collect()
            
            # 3. Clear CUDA Cache
            torch.cuda.empty_cache()
            
            self.log("Resources cleaned.")

    def stop(self):
        self.running = False

# ==============================================================================
# 5. GUI THEME & STYLING
# ==============================================================================
THEME = {
    "bg": "#F8FDFF",           
    "panel": "#FFFFFF",        
    "text": "#546E7A",         
    "title_main": "#4FC3F7",   
    "title_sub": "#B3E5FC",    
    "sub_text": "#91999E",     
    "accent": "#B3E5FC",       
    "accent_darker": "#81D4FA",
    "border": "#E1F5FE",
    "input_bg": "#FFFFFF",
    "log_text": "#455A64"
}

STYLESHEET = f"""
    QMainWindow {{ background-color: {THEME['bg']}; }}
    QWidget {{ font-family: "Segoe UI", sans-serif; font-size: 12px; color: {THEME['text']}; }}
    
    QFrame#mainPanel {{
        background-color: {THEME['panel']};
        border: 1px solid {THEME['border']};
        border-radius: 12px;
    }}

    QLineEdit, QComboBox, QDoubleSpinBox {{
        background-color: {THEME['input_bg']};
        border: 1px solid {THEME['border']};
        border-radius: 6px;
        padding: 4px 8px;
        color: {THEME['text']};
        selection-background-color: {THEME['accent']};
    }}
    QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus {{ border: 1px solid {THEME['accent_darker']}; }}
    
    QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 0px; border: none; }} 
    
    /* ComboBox Styling */
    QComboBox::drop-down {{ 
        border: none; 
        width: 20px; 
        background: transparent; /* Remove blue rectangle */
    }}
    QComboBox::down-arrow {{
        width: 0; height: 0;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 6px solid {THEME['title_main']};
        margin-right: 6px;
    }}
    
    /* CheckBox Styling */
        QCheckBox {{
        color: {THEME['text']};
        spacing: 8px;
    }}
    QCheckBox::indicator {{
        width: 16px;
        height: 16px;
        border-radius: 4px;
        border: 1px solid {THEME['border']};
        background-color: {THEME['input_bg']};
    }}
    QCheckBox::indicator:checked {{
        background-color: {THEME['accent']};
        border: 1px solid {THEME['accent_darker']};
    }}
    QCheckBox::indicator:hover {{
        border: 1px solid {THEME['accent_darker']};
    }}
    
    QPlainTextEdit {{
        background-color: transparent; 
        color: {THEME['log_text']};
        border: none;
        padding: 15px;
        font-family: "Consolas", monospace;
        font-size: 11px;
    }}
    
    QSplitter::handle {{ background: none; }}
"""

# ==============================================================================
# 6. UI COMPONENTS
# ==============================================================================
class LRScheduleGraph(QWidget):
    """
    Visual warmup Bezier + cosine decay LR schedule.
    Updated with top padding to prevent line clipping.
    """
    def __init__(self, warmup_fraction: float = 0.1, lr_min: float = 0.0, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(80)
        self.total_steps = 1000
        self.peak_lr = 1e-4
        self.lr_min = lr_min
        self.warmup_fraction = warmup_fraction

        self.current_step = 0
        self.current_lr = self.peak_lr
        self.is_training = False
        self.hover_step = -1

        self.schedule = WarmupCosineSchedule(
            total_steps=self.total_steps,
            lr_max=self.peak_lr,
            lr_min=self.lr_min,
            warmup_fraction=self.warmup_fraction,
        )

        self.setMouseTracking(True)

    def update_data(self, steps: int, lr_max: float):
        self.total_steps = max(1, int(steps))
        self.peak_lr = float(lr_max)
        self.schedule = WarmupCosineSchedule(
            total_steps=self.total_steps,
            lr_max=self.peak_lr,
            lr_min=self.lr_min,
            warmup_fraction=self.warmup_fraction,
        )
        self.update()

    def set_progress(self, step: int, lr: float | None = None):
        self.current_step = max(0, min(int(step), self.total_steps))
        if lr is not None:
            self.current_lr = float(lr)
        else:
            self.current_lr = self.schedule.lr_at(self.current_step)
        self.update()

    def mouseMoveEvent(self, event):
        w = self.width()
        if w <= 0 or self.total_steps <= 0:
            return super().mouseMoveEvent(event)

        x = max(0, min(event.pos().x(), w))
        t = x / float(max(1, w))
        self.hover_step = int(round(t * self.total_steps))

        if not self.is_training:
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self.hover_step = -1
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        w = self.width()
        h = self.height()
        if w < 5 or h < 5:
            return

        # --- Layout Metrics ---
        pad_t = 12  # Top padding prevents clipping
        pad_b = 15  # Bottom padding for text
        
        base_y = h - pad_b        # The visual "zero" line
        graph_h = base_y - pad_t  # The actual height available for the curve

        lr_max = self.peak_lr
        steps = max(1, self.total_steps)

        # Keypoints
        warmup_t = 0.10  
        warmup_lr = 1.0  # Normalized peak

        lr_end = self.schedule.lr_at(steps)

        # -----------------------------
        # Build smooth path
        # -----------------------------
        path = QPainterPath()

        # Start at LR=0 (bottom-left)
        path.moveTo(0, base_y)

        # 1. Warmup cubic Bezier
        x1 = warmup_t * w
        # Y calculation: base_y - (value * graph_h)
        y1 = base_y - warmup_lr * graph_h

        cx1 = x1 * 0.33
        cy1 = base_y - (warmup_lr * graph_h) * 0.25

        cx2 = x1 * 0.66
        cy2 = base_y - (warmup_lr * graph_h) * 0.95

        path.cubicTo(cx1, cy1, cx2, cy2, x1, y1)

        # 2. Cosine decay cubic Bezier
        x2 = w
        ratio_end = (lr_end / lr_max) if lr_max > 0 else 0
        y2 = base_y - ratio_end * graph_h

        decay_cx1 = x1 + (x2 - x1) * 0.33
        decay_cy1 = y1 + (y2 - y1) * 0.05

        decay_cx2 = x1 + (x2 - x1) * 0.66
        decay_cy2 = y1 + (y2 - y1) * 0.85

        path.cubicTo(decay_cx1, decay_cy1, decay_cx2, decay_cy2, x2, y2)

        # -----------------------------
        # Fill under curve
        # -----------------------------
        fill = QPainterPath(path)
        fill.lineTo(w, base_y)
        fill.lineTo(0, base_y)
        fill.closeSubpath()

        fill_col = QColor(THEME["accent"])
        fill_col.setAlpha(55)
        painter.fillPath(fill, QBrush(fill_col))

        # Stroke
        painter.setPen(QPen(QColor(THEME["title_main"]), 2))
        painter.drawPath(path)

        # -----------------------------
        # Progress indicator
        # -----------------------------
        if self.is_training:
            t = self.current_step / steps
            x = t * w
            lr = self.current_lr
            ratio = (lr / lr_max) if lr_max > 0 else 0
            y = base_y - ratio * graph_h

            guide = QPen(QColor(THEME["text"]))
            guide.setStyle(Qt.PenStyle.DotLine)
            guide.setWidthF(0.6)
            painter.setPen(guide)
            painter.drawLine(QPointF(x, 0), QPointF(x, h))

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#2980b9"))
            painter.drawEllipse(QPointF(x, y), 4, 4)

        # -----------------------------
        # Hover preview
        # -----------------------------
        elif self.hover_step >= 0:
            t = self.hover_step / steps
            x = t * w
            lr = self.schedule.lr_at(self.hover_step)
            ratio = (lr / lr_max) if lr_max > 0 else 0
            y = base_y - ratio * graph_h

            guide = QPen(QColor(THEME["sub_text"]).lighter(150))
            guide.setStyle(Qt.PenStyle.DotLine)
            guide.setWidthF(0.6)
            painter.setPen(guide)
            painter.drawLine(QPointF(x, 0), QPointF(x, h))

            painter.setPen(Qt.PenStyle.NoPen)
            dot = QColor("#2980b9")
            dot.setAlpha(160)
            painter.setBrush(dot)
            painter.drawEllipse(QPointF(x, y), 4, 4)

            painter.setPen(QColor(THEME["sub_text"]))
            painter.drawText(
                QRectF(x - 30, 2, 60, 12),
                Qt.AlignmentFlag.AlignCenter,
                f"Step: {self.hover_step}",
            )

        # -----------------------------
        # Labels
        # -----------------------------
        painter.setPen(QColor(THEME["sub_text"]))
        painter.drawText(
            QRectF(0, h - 12, w, 12),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"LR: {lr_max:.1e}",
        )
        painter.drawText(
            QRectF(0, h - 12, w, 12),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            f"Steps: {steps}",
        )

class NeumorphicButton(QPushButton):
    def __init__(self, text, is_primary=False, height=32, width=None):
        super().__init__(text)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(height)
        if width: self.setFixedWidth(width)
        
        bg_stop0 = "#FFFFFF"
        bg_stop1 = "#F4F9FB"
        if is_primary:
            bg_stop0 = "#E1F5FE"
            bg_stop1 = "#B3E5FC"
            
        fg = THEME['text']
        border_col = THEME['accent'] if is_primary else THEME['border']
        
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {bg_stop0}, stop:1 {bg_stop1});
                color: {fg}; border: 1px solid {border_col};
                border-radius: 6px; font-weight: 600; padding-bottom: 1px;
            }}
            QPushButton:hover {{ background-color: {THEME['accent_darker']} if {is_primary} else '#FFFFFF'; }}
            QPushButton:pressed {{ background-color: {THEME['accent']}; }}
            QPushButton:disabled {{ background-color: #EEEEEE; color: #BBBBBB; border-color: #DDDDDD; }}
        """)
        
        if is_primary:
            shadow = QGraphicsDropShadowEffect(self)
            shadow.setBlurRadius(12)
            shadow.setColor(QColor(179, 229, 252, 120)) 
            shadow.setOffset(0, 3)
            self.setGraphicsEffect(shadow)

class CompactFilePicker(QWidget):
    path_changed = pyqtSignal(str)
    def __init__(self, placeholder, is_dir=False):
        super().__init__()
        self.is_dir = is_dir
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.input = QLineEdit()
        self.input.setPlaceholderText(placeholder)
        self.input.textChanged.connect(lambda t: self.path_changed.emit(t))
        layout.addWidget(self.input)
        self.btn = NeumorphicButton("...", height=26, width=32)
        self.btn.clicked.connect(self.browse)
        layout.addWidget(self.btn)
    def browse(self):
        if self.is_dir: path = QFileDialog.getExistingDirectory(self, "Select Folder")
        else: path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "Safetensors (*.safetensors);;All (*)")
        if path: self.input.setText(path)

class PerfectSlider(QSlider):
    def __init__(self, orientation=Qt.Orientation.Horizontal):
        super().__init__(orientation)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(16)
        self.handle_radius = 6   
        self.grab_radius = 12    
        # Logic Change: Visual margin is now tighter (based on handle, not grab area)
        self.visual_margin = self.handle_radius 
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        groove_h = 3
        cy = rect.height() / 2
        groove_y = cy - (groove_h / 2)
        
        # Logic Change: Use tighter margin for drawing
        margin = self.visual_margin 
        groove_rect = QRectF(rect.left() + margin, groove_y, rect.width() - (2 * margin), groove_h)
        
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(THEME['border']))
        painter.drawRoundedRect(groove_rect, 1.5, 1.5)
        
        min_v, max_v, val = self.minimum(), self.maximum(), self.value()
        pos_ratio = 0 if max_v == min_v else (val - min_v) / (max_v - min_v)
        
        active_w = pos_ratio * groove_rect.width()
        active_rect = QRectF(groove_rect.left(), groove_y, active_w, groove_h)
        
        painter.setBrush(QColor(THEME['accent']))
        painter.drawRoundedRect(active_rect, 1.5, 1.5)
        
        handle_cx = groove_rect.left() + active_w
        painter.setBrush(QColor("#FFFFFF"))
        painter.setPen(QPen(QColor(THEME['accent']), 1.2))
        painter.drawEllipse(QPointF(handle_cx, cy), self.handle_radius, self.handle_radius)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            groove_left = self.visual_margin
            groove_width = self.width() - 2 * groove_left
            min_v, max_v, val = self.minimum(), self.maximum(), self.value()
            pos_ratio = 0 if max_v == min_v else (val - min_v) / (max_v - min_v)
            handle_x = groove_left + pos_ratio * groove_width
            
            if abs(event.pos().x() - handle_x) <= self.grab_radius:
                self.setSliderDown(True)
                self.mouseMoveEvent(event)
            else:
                self.setValue(self._pixelPosToValue(event.pos().x()))
                self.setSliderDown(True)
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        if self.isSliderDown():
            self.setValue(self._pixelPosToValue(event.pos().x()))
        super().mouseMoveEvent(event)
    
    def _pixelPosToValue(self, x):
        # Logic Change: specific mapping using the tighter visual margin
        groove_left = self.visual_margin
        groove_width = self.width() - 2 * groove_left
        min_v, max_v = self.minimum(), self.maximum()
        x = max(groove_left, min(x, self.width() - groove_left))
        ratio = (x - groove_left) / groove_width
        value = min_v + ratio * (max_v - min_v)
        return int(value)

class CompactSlider(QWidget):
    valueChanged = pyqtSignal(float)
    def __init__(self, label, min_v, max_v, def_v, step=1, display_scale=1.0, precision=0, scientific=False):
        super().__init__()
        self.display_scale = display_scale
        self.precision = precision
        self.step_val = step
        self.scientific = scientific
        layout = QHBoxLayout(self)
        # Logic Change: Reduced margins and spacing to 4 (was 8) for wider look
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4) 
        
        lbl = QLabel(label)
        lbl.setFixedWidth(85) 
        lbl.setStyleSheet(f"color: {THEME['sub_text']}; font-weight: 600; font-size: 11px;")
        
        self.slider = PerfectSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(int(min_v), int(max_v))
        self.slider.setValue(int(def_v))
        self.slider.setSingleStep(step)
        self.slider.setPageStep(step)
        
        # Logic Change: Ensure slider takes all available space
        self.slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
        self.val_lbl = QLabel()
        self.val_lbl.setFixedWidth(45)
        self.val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.val_lbl.setStyleSheet("font-size: 11px;")
        
        self.slider.valueChanged.connect(self.on_change)
        layout.addWidget(lbl)
        layout.addWidget(self.slider)
        layout.addWidget(self.val_lbl)
        self.on_change(int(def_v))
    
    def on_change(self, val):
        if self.step_val > 1:
            remainder = val % self.step_val
            if remainder != 0:
                val = round(val / self.step_val) * self.step_val
                self.slider.blockSignals(True)
                self.slider.setValue(val)
                self.slider.blockSignals(False)
        real_val = val * self.display_scale
        
        # Use scientific notation if enabled
        if self.scientific:
            self.val_lbl.setText(f"{real_val:.1e}")
        else:
            fmt = f"{{:.{self.precision}f}}"
            self.val_lbl.setText(fmt.format(real_val))
        
        self.valueChanged.emit(real_val)
    
    def value(self): 
        return self.slider.value() * self.display_scale

class DualInput(QWidget):
    def __init__(self, label="betas"):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        lbl = QLabel(label)
        lbl.setFixedWidth(85) 
        lbl.setStyleSheet(f"color: {THEME['sub_text']}; font-weight: 600;")
        layout.addWidget(lbl)
        self.b1 = QDoubleSpinBox()
        self.b1.setRange(0.0, 1.0); self.b1.setDecimals(3); self.b1.setValue(0.9)
        self.b1.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self.b2 = QDoubleSpinBox()
        self.b2.setRange(0.0, 1.0); self.b2.setDecimals(3); self.b2.setValue(0.99)
        self.b2.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        layout.addWidget(self.b1)
        layout.addWidget(self.b2)
    def get_tuple_str(self): return f"({self.b1.value()},{self.b2.value()})"

class CustomSplitterHandle(QSplitterHandle):
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        mid_x = self.rect().width() // 2
        mid_y = self.rect().height() // 2
        painter.setPen(QPen(QColor(THEME['border'])))
        painter.drawLine(mid_x, 0, mid_x, self.rect().height())
        painter.setBrush(QBrush(QColor("#FFFFFF")))
        painter.setPen(QPen(QColor(THEME['accent']), 2))
        painter.drawEllipse(mid_x - 3, mid_y - 3, 6, 6)

class CustomSplitter(QSplitter):
    def createHandle(self): return CustomSplitterHandle(self.orientation(), self)

# ==============================================================================
# 7. OPTIMIZER PANEL WITH ADAPTIVE SETTINGS
# ==============================================================================
class OptimizerPanel(QWidget):
    schedule_update_req = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 5, 0, 0)
        self.layout.setSpacing(9)

        # Selection
        row_sel = QHBoxLayout()
        row_sel.setContentsMargins(0,0,0,0)
        lbl_type = QLabel("type:")
        lbl_type.setStyleSheet("font-weight: 600; color: " + THEME['sub_text'])
        lbl_type.setFixedWidth(85)
        self.cb_opt = QComboBox()
        self.cb_opt.addItems(["raven", "adamw", "adafactor", "prodigy"])
        self.cb_opt.currentTextChanged.connect(self.on_opt_change)
        row_sel.addWidget(lbl_type)
        row_sel.addWidget(self.cb_opt)
        self.layout.addLayout(row_sel)

        # Params
        self.s_lr = CompactSlider("peak lr", 1, 50, 10, step=1, display_scale=1e-5, scientific=True)
        self.s_wd = CompactSlider("decay", 0, 100, 10, step=1, display_scale=0.001, precision=3)
        self.betas = DualInput("betas")
        
        self.chk_ggpo = QCheckBox("enable ggpo")
        self.chk_ggpo.setToolTip("Gradient-Guided Perturbation Optimization\nMitigates double descent.")
        # Removed margin to ensure it fits in the column
        self.chk_ggpo.setStyleSheet(f"color: {THEME['sub_text']};")
        
        self.s_lr.slider.valueChanged.connect(self.emit_update)
        
        self.layout.addWidget(self.s_lr)
        self.layout.addWidget(self.s_wd)
        self.layout.addWidget(self.betas)
        self.layout.addWidget(self.chk_ggpo)
        
        # Initialize state
        self.on_opt_change("raven")

    def emit_update(self): self.schedule_update_req.emit()

    def on_opt_change(self, text):
        # Adaptive UI Logic
        if text == "raven":
            self.s_wd.setVisible(True)
            self.betas.setVisible(True)
            self.s_lr.slider.setValue(10) # 1e-4
            self.s_wd.slider.setValue(0)
        elif text == "adamw":
            self.s_wd.setVisible(True)
            self.betas.setVisible(True)
            self.s_lr.slider.setValue(10)
        elif text == "adafactor":
            self.s_wd.setVisible(False) # No weight decay typical for adafactor via this arg
            self.betas.setVisible(False)
            self.s_lr.slider.setValue(100) # 1e-3
        elif text == "prodigy":
            self.s_wd.setVisible(True)
            self.betas.setVisible(False)
            self.s_lr.slider.setValue(100) # d_coef 1.0
        
        self.schedule_update_req.emit()

    def get_config(self):
        return {
            "type": self.cb_opt.currentText(),
            "lr": f"{self.s_lr.value():.2e}",
            "wd": self.s_wd.value(),
            "betas": self.betas.get_tuple_str(),
            "use_ggpo": self.chk_ggpo.isChecked()
        }

class SamplingPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 2, 0, 0) # Reduced top margin
        self.layout.setSpacing(4) # Tight spacing
        
        # Method selection Row
        row_method = QHBoxLayout()
        row_method.setContentsMargins(0,0,0,0)
        row_method.setSpacing(5)
        
        lbl_method = QLabel("method:")
        lbl_method.setStyleSheet("font-weight: 600; color: " + THEME['sub_text'])
        lbl_method.setFixedWidth(85)
        
        self.cb_method = QComboBox()
        self.cb_method.addItems(TimestepSamplingConfig.METHODS)
        self.cb_method.currentTextChanged.connect(self.on_method_change)
        
        # SQUASH: Inline LogSNR Checkbox
        self.chk_log_snr = QCheckBox("log snr") 
        self.chk_log_snr.setToolTip("Enable Uniform LogSNR")
        self.chk_log_snr.setStyleSheet(f"color: {THEME['sub_text']}; margin-left: 5px;")
        
        row_method.addWidget(lbl_method)
        row_method.addWidget(self.cb_method, stretch=1)
        row_method.addWidget(self.chk_log_snr) # Adds to same row
        
        self.layout.addLayout(row_method)
        
        # Timestep range (Vertical stack but tight)
        self.s_ts_min = CompactSlider("ts min", 0, 999, 0, step=1)
        self.s_ts_max = CompactSlider("ts max", 0, 999, 999, step=1)
        self.layout.addWidget(self.s_ts_min)
        self.layout.addWidget(self.s_ts_max)
        
        # Logit normal params
        self.s_logit_mean = CompactSlider("logit mean", -30, 30, 0, step=1, display_scale=0.1, precision=1)
        self.s_logit_std = CompactSlider("logit std", 1, 30, 10, step=1, display_scale=0.1, precision=1)
        self.layout.addWidget(self.s_logit_mean)
        self.layout.addWidget(self.s_logit_std)
        
        self.on_method_change("Random Integer")
    
    def on_method_change(self, method):
        # Show/hide relevant controls based on method
        is_logit = "Logit" in method
        
        self.s_logit_mean.setVisible(is_logit)
        self.s_logit_std.setVisible(is_logit)
        self.s_ts_min.setVisible(not is_logit)
        self.s_ts_max.setVisible(not is_logit)
        
        is_log_snr = (method == "Uniform LogSNR")
        self.chk_log_snr.setVisible(is_log_snr)
        self.chk_log_snr.setChecked(is_log_snr)
    
    def get_config(self):
        return TimestepSamplingConfig()  # Will be filled by main window

class SchedulerPanel(QWidget):
    prediction_changed = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 2, 0, 0) # Reduced top margin
        self.layout.setSpacing(4) # Tight spacing
        
        # Scheduler type
        row_sched = QHBoxLayout()
        row_sched.setContentsMargins(0,0,0,0)
        lbl_sched = QLabel("scheduler:")
        lbl_sched.setStyleSheet("font-weight: 600; color: " + THEME['sub_text'])
        lbl_sched.setFixedWidth(85)
        self.cb_noise_scheduler = QComboBox()
        self.cb_noise_scheduler.addItems(SchedulerConfig.TYPES)
        row_sched.addWidget(lbl_sched)
        row_sched.addWidget(self.cb_noise_scheduler)
        self.layout.addLayout(row_sched)
        
        # Prediction type & Zero SNR (Merged Row)
        row_pred = QHBoxLayout()
        row_pred.setContentsMargins(0,0,0,0)
        lbl_pred = QLabel("prediction:")
        lbl_pred.setStyleSheet("font-weight: 600; color: " + THEME['sub_text'])
        lbl_pred.setFixedWidth(85)
        
        self.cb_prediction = QComboBox()
        self.cb_prediction.addItems(SchedulerConfig.PREDICTION_TYPES)
        
        # SQUASH: Inline Zero SNR Checkbox
        self.chk_zero_snr = QCheckBox("zero snr") 
        self.chk_zero_snr.setToolTip("Enforce Zero Terminal SNR")
        self.chk_zero_snr.setStyleSheet(f"color: {THEME['sub_text']}; margin-left: 5px;")
        
        row_pred.addWidget(lbl_pred)
        row_pred.addWidget(self.cb_prediction, stretch=1)
        row_pred.addWidget(self.chk_zero_snr) # Adds to same row
        
        self.layout.addLayout(row_pred)
        
        # Flow shift (only for flow matching)
        self.s_flow_shift = CompactSlider("flow shift", 10, 100, 30, step=1, display_scale=0.1, precision=1)
        self.s_flow_shift.setVisible(False)
        self.layout.addWidget(self.s_flow_shift)
        
        self.cb_prediction.currentTextChanged.connect(self.on_prediction_change)
    
    def on_prediction_change(self, pred_type):
        if pred_type == "v_prediction":
            self.chk_zero_snr.setChecked(True)
        else:
            self.chk_zero_snr.setChecked(False)
        
        is_flow = pred_type == "flow_matching"
        self.s_flow_shift.setVisible(is_flow)
        
        if is_flow and self.cb_noise_scheduler.currentText() != "FlowMatchEulerDiscreteScheduler":
            idx = self.cb_noise_scheduler.findText("FlowMatchEulerDiscreteScheduler")
            if idx >= 0:
                self.cb_noise_scheduler.setCurrentIndex(idx)
        
        self.prediction_changed.emit(pred_type)
    
    def get_config(self):
        cfg = SchedulerConfig()
        cfg.scheduler_type = self.cb_noise_scheduler.currentText()
        cfg.prediction_type = self.cb_prediction.currentText()
        cfg.use_zero_terminal_snr = self.chk_zero_snr.isChecked()
        cfg.flow_shift = self.s_flow_shift.value()
        return cfg

# ==============================================================================
# 8. MAIN WINDOW & APP LOGIC
# ==============================================================================
class ConvaWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("conva")
        self._apply_app_icon()
        self.resize(1100, 600)
        self.setMinimumWidth(1100)
        self.setMinimumHeight(600)
        self.setStyleSheet(STYLESHEET)
        
        self.splitter = CustomSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(12)
        self.setCentralWidget(self.splitter)
        
        # LEFT (Controls)
        left_widget = QWidget()
        left_widget.setMinimumWidth(600) 
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(15, 10, 15, 15) 
        left_layout.setSpacing(6)
        
        # Header
        header_container = QWidget()
        header_container.setFixedHeight(35)
        header_layout = QHBoxLayout(header_container)
        header_layout.setContentsMargins(15, 0, 15, 10)  # ← Match panel margins
        header_layout.setSpacing(8)

        # Title
        lbl_title = QLabel("conva")
        lbl_title.setStyleSheet(f"font-family: 'Segoe UI Light'; font-size: 12px; color: {THEME['title_main']};")
        lbl_suffix = QLabel("low-rank sdxl trainer")
        lbl_suffix.setStyleSheet(f"font-family: 'Segoe UI Light'; font-size: 12px; color: {THEME['title_sub']}; margin-left: 4px;")
        header_layout.addWidget(lbl_title)
        header_layout.addWidget(lbl_suffix)

        # Push everything else to the right
        header_layout.addStretch()

        # Config buttons (right side)
        self.btn_save_cfg = NeumorphicButton("save", height=24, width=50)
        self.btn_save_cfg.clicked.connect(self.save_config_toml)
        self.btn_load_cfg = NeumorphicButton("load", height=24, width=50)
        self.btn_load_cfg.clicked.connect(self.load_config_toml)
        header_layout.addWidget(self.btn_save_cfg)
        header_layout.addWidget(self.btn_load_cfg)

        left_layout.addWidget(header_container)
        
        # Panel - Horizontal split layout
        self.panel = QFrame()
        self.panel.setObjectName("mainPanel")
        self.panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        main_panel_layout = QHBoxLayout(self.panel)
        main_panel_layout.setSpacing(15)
        main_panel_layout.setContentsMargins(15, 15, 15, 15)

        def header(t):
            lbl = QLabel(t)
            lbl.setStyleSheet(f"color: {THEME['sub_text']}; font-weight: 700; font-size: 9px; text-transform: uppercase; letter-spacing: 1px;")
            return lbl
        
        # LEFT COLUMN
        left_column = QWidget()
        left_column.setMinimumWidth(200)  # Reduced from 220
        left_column_layout = QVBoxLayout(left_column)
        left_column_layout.setSpacing(6)
        left_column_layout.setContentsMargins(0, 0, 0, 0)
        
        # Source
        left_column_layout.addWidget(header("source"))
        self.pk_model = CompactFilePicker("base model (.safetensors)")
        self.pk_data = CompactFilePicker("dataset folder (images and text)", is_dir=True)
        self.pk_out = CompactFilePicker("output folder", is_dir=True)
        left_column_layout.addWidget(self.pk_model)
        left_column_layout.addWidget(self.pk_data)
        left_column_layout.addWidget(self.pk_out)
        
        # Training
        left_column_layout.addWidget(header("training"))
        self.s_batch = CompactSlider("batch", 1, 16, 1)
        self.s_grad = CompactSlider("grad acc", 1, 64, 1)
        self.s_res = CompactSlider("resolution", 512, 1024, 1024, step=64)
        self.s_epoch = CompactSlider("epochs", 1, 50, 10)
        left_column_layout.addWidget(self.s_batch)
        left_column_layout.addWidget(self.s_grad)
        left_column_layout.addWidget(self.s_res)
        left_column_layout.addWidget(self.s_epoch)

        # Structure
        left_column_layout.addWidget(header("structure"))
        self.s_dim = CompactSlider("dim", 4, 128, 32, step=4)
        self.s_alp = CompactSlider("alpha", 4, 128, 16, step=4)
        self.s_cdim = CompactSlider("conv dim", 4, 64, 32, step=4)
        self.s_calp = CompactSlider("conv alpha", 4, 64, 16, step=4)
        left_column_layout.addWidget(self.s_dim)
        left_column_layout.addWidget(self.s_alp)
        left_column_layout.addWidget(self.s_cdim)
        left_column_layout.addWidget(self.s_calp)
        
        # RIGHT COLUMN
        right_column = QWidget()
        right_column.setMinimumWidth(200)  # Reduced from 220
        right_column_layout = QVBoxLayout(right_column)
        right_column_layout.setSpacing(6)
        right_column_layout.setContentsMargins(0, 0, 0, 0)

        # Optimizer
        right_column_layout.addWidget(header("optimizer"))
        self.opt_panel = OptimizerPanel()
        right_column_layout.addWidget(self.opt_panel)

        # Sampling
        right_column_layout.addWidget(header("sampling"))
        self.sampling_panel = SamplingPanel()
        right_column_layout.addWidget(self.sampling_panel)

        # Scheduler
        right_column_layout.addWidget(header("scheduler"))
        self.scheduler_panel = SchedulerPanel()
        right_column_layout.addWidget(self.scheduler_panel)
        
        # Add columns to main panel with fixed proportions
        main_panel_layout.addWidget(left_column, stretch=1)
        
        # Subtle vertical line separator matching header text color
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.NoFrame) # Disable native 3D look
        separator.setFixedWidth(1)
        # Use THEME['sub_text'] directly and set border:none to prevent overrides
        separator.setStyleSheet(f"background-color: {THEME['sub_text']}; border: none;")
        
        # Add a slight opacity effect via rgba if desired, or keep solid sub_text color
        # Here we apply a generic opacity to the widget to make it subtle
        opacity_effect = QGraphicsOpacityEffect(self)
        opacity_effect.setOpacity(0.3)
        separator.setGraphicsEffect(opacity_effect)
        
        main_panel_layout.addWidget(separator, stretch=0)
        
        main_panel_layout.addWidget(right_column, stretch=1)

        left_layout.addWidget(self.panel)
        
        # Bottom - UPDATED LAYOUT
        bottom_layout = QHBoxLayout()
        self.graph = LRScheduleGraph()
        bottom_layout.addWidget(self.graph)
        
        # Button container
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(10)
        
        self.btn_run = NeumorphicButton("start training", True, 40)
        self.btn_run.clicked.connect(self.start_training)
        
        self.btn_stop = NeumorphicButton("stop training", False, 40)
        self.btn_stop.clicked.connect(self.stop_training)
        self.btn_stop.setEnabled(False)  # Initially disabled
        
        button_layout.addWidget(self.btn_run)
        button_layout.addWidget(self.btn_stop)
        
        left_layout.addLayout(bottom_layout)

        left_layout.addWidget(button_container)
        
        # RIGHT (Log)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Ready.\nSelect model, data and output before training.")
        
        self.splitter.addWidget(left_widget)
        self.splitter.addWidget(self.log_view)
        self.splitter.setSizes([480, 500]) # Match the new minimum width
        self.splitter.setCollapsible(0, False)
        
        # Connections
        self.pk_data.path_changed.connect(self.recalc)
        self.s_batch.valueChanged.connect(lambda v: self.recalc())
        self.s_grad.valueChanged.connect(lambda v: self.recalc())
        self.s_epoch.valueChanged.connect(lambda v: self.recalc())
        self.opt_panel.schedule_update_req.connect(self.recalc)
        
        self.trainer_thread = None
    
    def _apply_app_icon(self):
        """Generates a blue neumorphic dot icon in memory."""
        size = 64
        pix = QPixmap(size, size)
        pix.fill(Qt.GlobalColor.transparent)
        
        p = QPainter(pix)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 1. Drop Shadow (Soft dark offset)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 40))
        p.drawEllipse(QRectF(6, 8, 52, 52))
        
        # 2. Main Body (Radial Gradient for 3D feel)
        # Center the light source slightly top-left
        grad = QRadialGradient(24, 24, 50) 
        grad.setColorAt(0.0, QColor("#81D4FA")) # Light Blue (Accent)
        grad.setColorAt(1.0, QColor("#0277BD")) # Deep Blue
        
        p.setBrush(grad)
        p.drawEllipse(QRectF(4, 4, 52, 52))
        
        p.end()
        self.setWindowIcon(QIcon(pix))

    def save_config_toml(self):
        import toml
        from datetime import datetime
        
        config = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "version": "1.0"
            },
            "source": {
                "model": self.pk_model.input.text(),
                "data": self.pk_data.input.text(),
                "output": self.pk_out.input.text()
            },
            "training": {
                "batch": int(self.s_batch.value()),
                "grad_acc": int(self.s_grad.value()),
                "resolution": int(self.s_res.value()),
                "epochs": int(self.s_epoch.value())
            },
            "structure": {
                "dim": int(self.s_dim.value()),
                "alpha": int(self.s_alp.value()),
                "conv_dim": int(self.s_cdim.value()),
                "conv_alpha": int(self.s_calp.value())
            },
            "optimizer": self.opt_panel.get_config(),
            "sampling": {
                "method": self.sampling_panel.cb_method.currentText(),
                "use_log_snr": self.sampling_panel.chk_log_snr.isChecked(),
                "ts_min": int(self.sampling_panel.s_ts_min.value()),
                "ts_max": int(self.sampling_panel.s_ts_max.value()),
                "logit_mean": self.sampling_panel.s_logit_mean.value(),
                "logit_std": self.sampling_panel.s_logit_std.value()
            },
            "scheduler": {
                "type": self.scheduler_panel.cb_noise_scheduler.currentText(),
                "prediction_type": self.scheduler_panel.cb_prediction.currentText(),
                "use_zero_terminal_snr": self.scheduler_panel.chk_zero_snr.isChecked(),
                "flow_shift": self.scheduler_panel.s_flow_shift.value()
            }
        }
        
        path, _ = QFileDialog.getSaveFileName(self, "Save Config", "", "TOML (*.toml)")
        if path:
            with open(path, 'w') as f:
                toml.dump(config, f)
            self.log(f"Config saved: {path}")

    def load_config_toml(self):
        import toml
        
        path, _ = QFileDialog.getOpenFileName(self, "Load Config", "", "TOML (*.toml)")
        if not path:
            return
        
        try:
            with open(path, 'r') as f:
                config = toml.load(f)
            
            # Source
            if "source" in config:
                self.pk_model.input.setText(config["source"].get("model", ""))
                self.pk_data.input.setText(config["source"].get("data", ""))
                self.pk_out.input.setText(config["source"].get("output", ""))
            
            # Training
            if "training" in config:
                self.s_batch.slider.setValue(config["training"].get("batch", 1))
                self.s_grad.slider.setValue(config["training"].get("grad_acc", 1))
                self.s_res.slider.setValue(config["training"].get("resolution", 1024))
                self.s_epoch.slider.setValue(config["training"].get("epochs", 10))
            
            # Structure
            if "structure" in config:
                self.s_dim.slider.setValue(config["structure"].get("dim", 16))
                self.s_alp.slider.setValue(config["structure"].get("alpha", 8))
                self.s_cdim.slider.setValue(config["structure"].get("conv_dim", 8))
                self.s_calp.slider.setValue(config["structure"].get("conv_alpha", 1))
            
            # Optimizer
            if "optimizer" in config:
                opt = config["optimizer"]
                idx = self.opt_panel.cb_opt.findText(opt.get("type", "raven"))
                if idx >= 0:
                    self.opt_panel.cb_opt.setCurrentIndex(idx)
                # Set LR by converting scientific notation
                lr_str = opt.get("lr", "1.00e-04")
                lr_val = float(lr_str)
                self.opt_panel.s_lr.slider.setValue(int(lr_val / 1e-5))
                # Load GGPO setting
                self.opt_panel.chk_ggpo.setChecked(opt.get("use_ggpo", False))
            
            # Sampling
            if "sampling" in config:
                samp = config["sampling"]
                idx = self.sampling_panel.cb_method.findText(samp.get("method", "Random Integer"))
                if idx >= 0:
                    self.sampling_panel.cb_method.setCurrentIndex(idx)
                self.sampling_panel.chk_log_snr.setChecked(samp.get("use_log_snr", False))
                self.sampling_panel.s_ts_min.slider.setValue(samp.get("ts_min", 0))
                self.sampling_panel.s_ts_max.slider.setValue(samp.get("ts_max", 999))
                self.sampling_panel.s_logit_mean.slider.setValue(int(samp.get("logit_mean", 0.0) * 10))
                self.sampling_panel.s_logit_std.slider.setValue(int(samp.get("logit_std", 1.0) * 10))
            
            # Scheduler
            if "scheduler" in config:
                sched = config["scheduler"]
                idx = self.scheduler_panel.cb_noise_scheduler.findText(sched.get("type", "DDPMScheduler"))
                if idx >= 0:
                    self.scheduler_panel.cb_noise_scheduler.setCurrentIndex(idx)
                idx = self.scheduler_panel.cb_prediction.findText(sched.get("prediction_type", "epsilon"))
                if idx >= 0:
                    self.scheduler_panel.cb_prediction.setCurrentIndex(idx)
                self.scheduler_panel.chk_zero_snr.setChecked(sched.get("use_zero_terminal_snr", False))
                self.scheduler_panel.s_flow_shift.slider.setValue(int(sched.get("flow_shift", 3.0) * 10))
            
            self.log(f"Config loaded: {path}")
            self.recalc()
        except Exception as e:
            self.log(f"Error loading config: {e}")
    
    def log(self, msg):
        self.log_view.appendPlainText(msg)
        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

    def get_img_count(self):
        path = self.pk_data.input.text()
        if not os.path.isdir(path): return 0
        return len(glob.glob(os.path.join(path, "*.*")))

    def recalc(self):
        cnt = self.get_img_count()
        if cnt == 0: 
            self.graph.update_data(0, 0)
            return
        b = max(1, self.s_batch.value())
        g = max(1, self.s_grad.value())
        e = self.s_epoch.value()
        steps = math.ceil(cnt / b / g) * e
        self.graph.update_data(steps, self.opt_panel.s_lr.value())

    def start_training(self):
        if not self.pk_model.input.text(): return self.log("Error: Model missing")
        if not self.pk_data.input.text(): return self.log("Error: Data missing")
        if not self.pk_out.input.text(): return self.log("Error: Output missing")
        
        # Update button states
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        
        self.recalc()
        
        # Update Graph State
        self.graph.is_training = True
        self.graph.set_progress(0)
        self.graph.setMinimumHeight(80)
        
        config = {
            "model": self.pk_model.input.text(),
            "data": self.pk_data.input.text(),
            "out": self.pk_out.input.text(),
            "batch": int(self.s_batch.value()),
            "grad": int(self.s_grad.value()),
            "res": int(self.s_res.value()),
            "epoch": int(self.s_epoch.value()),
            "dim": int(self.s_dim.value()),
            "alpha": int(self.s_alp.value()),
            "cdim": int(self.s_cdim.value()),
            "calpha": int(self.s_calp.value()),
            "optim": self.opt_panel.get_config(),
            "total_steps": self.graph.total_steps
        }
        
        # Timestep sampling config
        ts_config = TimestepSamplingConfig()
        ts_config.method = self.sampling_panel.cb_method.currentText()
        ts_config.use_log_snr = self.sampling_panel.chk_log_snr.isChecked()
        ts_config.ts_min = int(self.sampling_panel.s_ts_min.value())
        ts_config.ts_max = int(self.sampling_panel.s_ts_max.value())
        ts_config.logit_mean = self.sampling_panel.s_logit_mean.value()
        ts_config.logit_std = self.sampling_panel.s_logit_std.value()
        
        # Scheduler config
        sched_config = self.scheduler_panel.get_config()
        
        self.trainer_thread = InternalTrainer(config, ts_config, sched_config)
        self.trainer_thread.log_signal.connect(self.log)
        self.trainer_thread.progress_signal.connect(self.on_progress)
        self.trainer_thread.finished_signal.connect(self.on_finished)
        self.trainer_thread.start()

    def stop_training(self):
        if self.trainer_thread and self.trainer_thread.isRunning():
            self.log("Stopping training... please wait")
            self.btn_stop.setEnabled(False)
            self.trainer_thread.stop()

    def on_progress(self, step: int, total: int, loss: float, lr: float):
        # Update LR graph
        self.graph.is_training = True
        self.graph.set_progress(step, lr)
        self.graph.update()

        # Update any loss/progress UI you have
        if hasattr(self, "loss_label"):
            self.loss_label.setText(f"loss: {loss:.4f}")

        # Update progress text
        if hasattr(self, "step_label"):
            self.step_label.setText(f"{step}/{total}")

    def on_finished(self):
        # Reset button states
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        
        self.graph.is_training = False
        self.graph.update()
        self.log("Process Finished.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    win = ConvaWindow()
    win.show()
    sys.exit(app.exec())