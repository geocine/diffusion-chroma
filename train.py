# MIT License
# Copyright tdrussell
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Original source: https://github.com/tdrussell/diffusion-pipe/blob/34249f23cb744a38f0c6fbe4f5418ce8ebe52a4b/train.py
# 
# Modifications by geocine

import argparse
import os
from datetime import datetime, timezone
import shutil
import glob
import time
import random
import json
import inspect
from pathlib import Path

import toml
import deepspeed
from deepspeed import comm as dist
from deepspeed.runtime.pipe import module as ds_pipe_module
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import multiprocess as mp
import numpy as np

from utils import dataset as dataset_util
from utils import common
from utils.common import is_main_process, get_rank, DTYPE_MAP, empty_cuda_cache
import utils.saver
from utils.isolate_rng import isolate_rng
from utils.patches import apply_patches
from utils.unsloth_utils import unsloth_checkpoint
from utils.pipeline import ManualPipelineModule

from torchvision.utils import make_grid, save_image
import torch.nn.functional as F
from PIL import Image

TIMESTEP_QUANTILES_FOR_EVAL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Path to TOML configuration file.')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
parser.add_argument('--resume_from_checkpoint', nargs='?', const=True, default=None,
                    help='resume training from checkpoint. If no value is provided, resume from the most recent checkpoint. If a folder name is provided, resume from that specific folder.')
parser.add_argument('--regenerate_cache', action='store_true', default=None, help='Force regenerate cache. Useful if none of the files have changed but their contents have, e.g. modified captions.')
parser.add_argument('--cache_only', action='store_true', default=None, help='Cache model inputs then exit.')
# REMOVED: parser.add_argument('--i_know_what_i_am_doing', action='store_true', default=None, help="Skip certain checks.")
parser.add_argument('--master_port', type=int, default=29500, help='Master port for distributed training')
# REMOVED: parser.add_argument('--dump_dataset', type=Path, default=None, help='Decode cached latents and dump the dataset to this directory.')
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()


# Monkeypatch this so it counts all layer parameters, not just trainable parameters.
# This helps it divide the layers between GPUs more evenly when training a LoRA.
def _count_all_layer_params(self):
    param_counts = [0] * len(self._layer_specs)
    for idx, layer in enumerate(self._layer_specs):
        if isinstance(layer, ds_pipe_module.LayerSpec):
            l = layer.build()
            param_counts[idx] = sum(p.numel() for p in l.parameters())
        elif isinstance(layer, nn.Module):
            param_counts[idx] = sum(p.numel() for p in layer.parameters())
    return param_counts
ds_pipe_module.PipelineModule._count_layer_params = _count_all_layer_params


def set_config_defaults(config):
    # Force the user to set this. If we made it a default of 1, it might use a lot of disk space.
    assert 'save_every_n_epochs' in config

    config.setdefault('pipeline_stages', 1)
    config.setdefault('activation_checkpointing', False)
    config['reentrant_activation_checkpointing'] = (config['activation_checkpointing'] == 'unsloth')
    config.setdefault('warmup_steps', 0)
    if 'save_dtype' in config:
        config['save_dtype'] = DTYPE_MAP[config['save_dtype']]

    model_config = config['model']
    model_dtype_str = model_config['dtype']
    model_config['dtype'] = DTYPE_MAP[model_dtype_str]
    if 'transformer_dtype' in model_config:
        model_config['transformer_dtype'] = DTYPE_MAP[model_config['transformer_dtype']]
    model_config.setdefault('guidance', 1.0)

    if 'adapter' in config:
        adapter_config = config['adapter']
        adapter_type = adapter_config['type']
        if adapter_config['type'] == 'lora':
            if 'alpha' in adapter_config:
                raise NotImplementedError(
                    'This script forces alpha=rank to make the saved LoRA format simpler and more predictable with downstream inference programs. Please remove alpha from the config.'
                )
            adapter_config['alpha'] = adapter_config['rank']
            adapter_config.setdefault('dropout', 0.0)
            adapter_config.setdefault('dtype', model_dtype_str)
            adapter_config['dtype'] = DTYPE_MAP[adapter_config['dtype']]
        else:
            raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')

    config.setdefault('logging_steps', 1)
    config.setdefault('eval_datasets', [])
    config.setdefault('eval_gradient_accumulation_steps', 1)
    config.setdefault('eval_every_n_steps', None)
    config.setdefault('eval_every_n_epochs', None)
    config.setdefault('eval_before_first_step', True)


def get_most_recent_run_dir(output_dir):
    return list(sorted(glob.glob(os.path.join(output_dir, '*'))))[-1]


def print_model_info(model):
    # No need for is_main_process check
    print(model)
    for name, module in model.named_modules():
        print(f'{type(module)}: {name}')
        for pname, p in module.named_parameters(recurse=False):
            print(pname)
            print(p.dtype)
            print(p.device)
            print(p.requires_grad)
            print()


# Need to preload all micro batches since pulling from the dataloader does IPC between the
# first and last stage. Can't do that during the train or inference pipeline schedule execution
# because it conflicts with the send / recv steps.
def get_data_iterator_for_step(dataloader, engine, num_micro_batches=None):
    num_micro_batches = num_micro_batches or engine.micro_batches
    # Simplified: With world_size=1, it's always the first and last stage.
    # The check 'if not (engine.is_first_stage() or engine.is_last_stage()): return None' is removed.
    dataloader_iter = iter(dataloader)
    items = [next(dataloader_iter) for _ in range(num_micro_batches)]
    return iter(items)


def evaluate_single(model_engine, eval_dataloader, eval_gradient_accumulation_steps, quantile, pbar=None):
    eval_dataloader.set_eval_quantile(quantile)
    total_loss = 0
    count = 0
    while True:
        model_engine.reset_activation_shape()
        iterator = get_data_iterator_for_step(eval_dataloader, model_engine, num_micro_batches=eval_gradient_accumulation_steps)
        loss = model_engine.eval_batch(iterator, num_micro_batches=eval_gradient_accumulation_steps).item()
        eval_dataloader.sync_epoch()
        if pbar: # pbar might be None if called outside _evaluate
             pbar.update(1)
        total_loss += loss
        count += 1
        if eval_dataloader.epoch == 2:
            break

    eval_dataloader.reset()
    return total_loss / count


def _evaluate(model_engine, eval_dataloaders, tb_writer, step, eval_gradient_accumulation_steps):
    pbar_total = 0
    for eval_dataloader in eval_dataloaders.values():
        pbar_total += len(eval_dataloader) * len(TIMESTEP_QUANTILES_FOR_EVAL) // eval_gradient_accumulation_steps

    # Always main process
    print('Running eval')
    pbar = tqdm(total=pbar_total)

    start = time.time()
    for name, eval_dataloader in eval_dataloaders.items():
        losses = []
        for quantile in TIMESTEP_QUANTILES_FOR_EVAL:
            loss = evaluate_single(model_engine, eval_dataloader, eval_gradient_accumulation_steps, quantile, pbar=pbar)
            losses.append(loss)
            # Always main process
            tb_writer.add_scalar(f'{name}/loss_quantile_{quantile:.2f}', loss, step)
        avg_loss = sum(losses) / len(losses)
        # Always main process
        tb_writer.add_scalar(f'{name}/loss', avg_loss, step)

    duration = time.time() - start
    # Always main process
    tb_writer.add_scalar('eval/eval_time_sec', duration, step)
    pbar.close()

def generate_sample_images(model, config, step=0, tb_writer=None):
    """
    Generate sample images during training for visualization.
    Enhanced version with better denoising and text conditioning.
    """
    # Print the actual configuration for debugging
    print("=== Inference Configuration ===")
    for key, value in config.get('inference', {}).items():
        print(f"{key}: {value}")
    print("==============================")
    
    # Extract inference settings from config
    inference_config = config.get('inference', {})
    steps = inference_config.get('steps', 30)
    guidance = inference_config.get('guidance', 0)
    cfg = inference_config.get('cfg', 7.5)  # Increased CFG for stronger guidance
    first_n_steps_wo_cfg = inference_config.get('first_n_steps_wo_cfg', 0)
    width, height = inference_config.get('image_dim', [512, 512])
    prompts = inference_config.get('prompts', ["A beautiful landscape"])
    t5_max_length = inference_config.get('t5_max_length', 512)
    seed = inference_config.get('seed')
    
    # Prepare output directory
    output_dir = config['output_dir']
    run_dirs = sorted(glob.glob(os.path.join(output_dir, "*")))
    if run_dirs:
        run_dir = run_dirs[-1]  # Get the most recent run directory
    else:
        run_dir = output_dir
    
    samples_dir = os.path.join(run_dir, inference_config.get('inference_folder', "samples"))
    os.makedirs(samples_dir, exist_ok=True)
    print(f"Will save samples to: {samples_dir}")
    
    # Prepare for inference
    model.prepare_block_swap_inference(disable_block_swap=True)
    
    # Check model state
    model_was_in_train = False
    if hasattr(model, 'training'):
        model_was_in_train = model.training
        model.eval()
    elif hasattr(model, 'transformer'):
        if hasattr(model.transformer, 'training'):
            model_was_in_train = model.transformer.training
            model.transformer.eval()
    
    # Get model dtype and device
    model_dtype = next(model.transformer.parameters()).dtype
    device = next(model.transformer.parameters()).device
    
    # Process each prompt
    all_images = []
    for prompt_idx, prompt in enumerate(prompts):
        print(f"Generating image for prompt: {prompt}")
        
        try:
            # Set seed for reproducibility
            current_seed = seed if seed is not None else random.randint(0, 999999)
            torch.manual_seed(current_seed)
            np.random.seed(current_seed)
            random.seed(current_seed)
            print(f"Using seed: {current_seed}")
            
            # Process text prompt to get embeddings
            try:
                # Try to encode the text prompt
                text_inputs = model.tokenizer_2(
                    [prompt],
                    padding="max_length",
                    max_length=t5_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).to(device)
                
                # Check if text_encoder is a meta tensor and handle accordingly
                is_meta_tensor = False
                try:
                    # Try to access a parameter to check if it's a meta tensor
                    next(model.text_encoder_2.parameters()).device
                except NotImplementedError as e:
                    if "Cannot determine data pointer from a meta tensor" in str(e):
                        is_meta_tensor = True
                        print("Detected meta tensor for text_encoder, using to_empty()")
                
                # Get text embeddings
                if is_meta_tensor:
                    # For meta tensors, we need to use to_empty() first
                    from accelerate import init_empty_weights
                    with init_empty_weights():
                        # Clone the model structure without weights
                        text_encoder_empty = type(model.text_encoder_2)()
                    
                    # Move the empty model to the device
                    text_encoder_empty = text_encoder_empty.to(device)
                    
                    # Load the weights from the original model
                    text_encoder_empty.load_state_dict(model.text_encoder_2.state_dict())
                    
                    # Use the properly loaded model
                    prompt_embeds = text_encoder_empty(
                        text_inputs.input_ids,
                        output_hidden_states=True
                    )[0]
                    
                    # Generate negative prompt embeddings
                    neg_prompt_embeds = text_encoder_empty(
                        model.tokenizer_2(
                            [""],
                            padding="max_length",
                            max_length=t5_max_length,
                            truncation=True,
                            return_tensors="pt"
                        ).input_ids.to(device),
                        output_hidden_states=True
                    )[0]
                    
                    # Clean up to save memory
                    del text_encoder_empty
                else:
                    # For regular tensors, use the normal approach
                    text_encoder = model.text_encoder_2.to(device)
                    prompt_embeds = text_encoder(
                        text_inputs.input_ids,
                        output_hidden_states=True
                    )[0]
                    
                    # Generate negative prompt embeddings
                    neg_prompt_embeds = text_encoder(
                        model.tokenizer_2(
                            [""],
                            padding="max_length",
                            max_length=t5_max_length,
                            truncation=True,
                            return_tensors="pt"
                        ).input_ids.to(device),
                        output_hidden_states=True
                    )[0]
                    
                    # Move text encoder back to CPU to save memory
                    text_encoder.to('cpu')
                
                print(f"Successfully encoded prompt: '{prompt}'")
                print(f"Text embedding shape: {prompt_embeds.shape}")
                
                # Create text attention mask
                text_attention_mask = torch.ones((1, t5_max_length), device=device, dtype=model_dtype)
                
            except Exception as e:
                print(f"Error encoding text prompt: {str(e)}")
                print("Using dummy text embeddings instead")
                # Create dummy text embeddings
                prompt_embeds = torch.zeros((1, t5_max_length, 4096), device=device, dtype=model_dtype)
                neg_prompt_embeds = torch.zeros((1, t5_max_length, 4096), device=device, dtype=model_dtype)
                text_attention_mask = torch.ones((1, t5_max_length), device=device, dtype=model_dtype)
            
            # Create initial random latents at 64x64 resolution
            latents = torch.randn(1, 16, 64, 64, device=device, dtype=model_dtype)
            
            # Prepare for denoising
            timesteps = torch.linspace(1.0, 0.0, steps+1)[:-1].to(device=device, dtype=model_dtype)
            
            # Try to use the full model for denoising if possible
            try_full_model = True
            
            # Denoise the latents
            with torch.no_grad(), isolate_rng():
                # Process each timestep
                for i, t in enumerate(tqdm(timesteps, desc="Denoising")):
                    # Create batch of timesteps
                    timestep = torch.full((1,), t, device=device, dtype=model_dtype)
                    
                    # Prepare inputs for the model
                    # Flatten latents to match what the model expects
                    flat_latents = latents.reshape(1, -1)
                    total_features = flat_latents.shape[1]
                    
                    # Reshape to [batch, 3072, 64] for the img_in layer
                    if total_features < 64*3072:
                        # Pad with zeros
                        padding = torch.zeros((1, 64*3072 - total_features), 
                                            device=flat_latents.device, 
                                            dtype=model_dtype)
                        flat_latents = torch.cat([flat_latents, padding], dim=1)
                    elif total_features > 64*3072:
                        # Truncate
                        flat_latents = flat_latents[:, :64*3072]
                    
                    img = flat_latents.reshape(1, 3072, 64).to(dtype=model_dtype)
                    
                    # Create position IDs with the same sequence length as the second dimension
                    img_ids = torch.zeros((1, img.shape[1], 3), device=device, dtype=model_dtype)
                    for y in range(8):
                        for x in range(8):
                            for idx in range(48):  # 3072 / 64 = 48 tokens per position
                                pos = (y * 8 + x) * 48 + idx
                                if pos < img.shape[1]:
                                    img_ids[0, pos, 0] = x / 8.0
                                    img_ids[0, pos, 1] = y / 8.0
                                    img_ids[0, pos, 2] = idx / 48.0
                    
                    # Create text position IDs
                    txt_ids = torch.zeros((1, t5_max_length, 3), device=device, dtype=model_dtype)
                    for j in range(t5_max_length):
                        txt_ids[0, j, 0] = j / t5_max_length
                    
                    # Create guidance values
                    guidance_value = torch.tensor([guidance], device=device, dtype=model_dtype)
                    
                    # Try to use the full model for better results
                    if try_full_model and i == 0:
                        try:
                            print("Attempting to use full transformer model...")
                            # Apply classifier-free guidance
                            if cfg > 1.0 and i >= first_n_steps_wo_cfg:
                                # Concatenate for classifier-free guidance
                                model_input = torch.cat([img, img], dim=0)
                                text_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)
                                timestep_batch = torch.cat([timestep, timestep], dim=0)
                                guidance_batch = torch.cat([guidance_value, guidance_value], dim=0)
                                img_ids_batch = torch.cat([img_ids, img_ids], dim=0)
                                txt_ids_batch = torch.cat([txt_ids, txt_ids], dim=0)
                                txt_mask_batch = torch.cat([text_attention_mask, text_attention_mask], dim=0)
                                
                                # Forward pass through full model
                                noise_pred = model.transformer(
                                    model_input,
                                    img_ids_batch,
                                    text_embeds,
                                    txt_ids_batch,
                                    txt_mask_batch,
                                    timestep_batch,
                                    guidance_batch
                                )
                                
                                # Split predictions and apply CFG
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                noise_pred = noise_pred_uncond + cfg * (noise_pred_text - noise_pred_uncond)
                            else:
                                # No CFG, just use the prompt
                                noise_pred = model.transformer(
                                    img,
                                    img_ids,
                                    prompt_embeds,
                                    txt_ids,
                                    text_attention_mask,
                                    timestep,
                                    guidance_value
                                )
                            
                            print("Successfully used full transformer model!")
                        except Exception as e:
                            print(f"Error using full transformer model: {str(e)}")
                            print("Falling back to img_in layer only")
                            try_full_model = False
                            
                            # Use just the img_in layer as fallback
                            noise_pred = model.transformer.img_in(img)
                    else:
                        # Use the img_in layer for subsequent steps or if full model failed
                        if not try_full_model:
                            noise_pred = model.transformer.img_in(img)
                        else:
                            # Continue using full model for better results
                            try:
                                # Apply classifier-free guidance
                                if cfg > 1.0 and i >= first_n_steps_wo_cfg:
                                    # Concatenate for classifier-free guidance
                                    model_input = torch.cat([img, img], dim=0)
                                    text_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)
                                    timestep_batch = torch.cat([timestep, timestep], dim=0)
                                    guidance_batch = torch.cat([guidance_value, guidance_value], dim=0)
                                    img_ids_batch = torch.cat([img_ids, img_ids], dim=0)
                                    txt_ids_batch = torch.cat([txt_ids, txt_ids], dim=0)
                                    txt_mask_batch = torch.cat([text_attention_mask, text_attention_mask], dim=0)
                                    
                                    # Forward pass through full model
                                    noise_pred = model.transformer(
                                        model_input,
                                        img_ids_batch,
                                        text_embeds,
                                        txt_ids_batch,
                                        txt_mask_batch,
                                        timestep_batch,
                                        guidance_batch
                                    )
                                    
                                    # Split predictions and apply CFG
                                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                    noise_pred = noise_pred_uncond + cfg * (noise_pred_text - noise_pred_uncond)
                                else:
                                    # No CFG, just use the prompt
                                    noise_pred = model.transformer(
                                        img,
                                        img_ids,
                                        prompt_embeds,
                                        txt_ids,
                                        text_attention_mask,
                                        timestep,
                                        guidance_value
                                    )
                            except Exception as e:
                                print(f"Error using full transformer model: {str(e)}")
                                print("Falling back to img_in layer only")
                                try_full_model = False
                                
                                # Use just the img_in layer as fallback
                                noise_pred = model.transformer.img_in(img)
                    
                    # Reshape noise prediction back to latent shape
                    noise_pred_flat = noise_pred.reshape(1, -1)
                    noise_pred_reshaped = noise_pred_flat[:, :latents.numel()].reshape(latents.shape)
                    
                    # Update latents with improved denoising step
                    # Use a more sophisticated update rule
                    alpha = 0.5 + 0.5 * (1.0 - t)  # Gradually increase strength as we denoise
                    latents = latents - alpha * noise_pred_reshaped
                    
                    # Add a small amount of noise to prevent getting stuck in local minima
                    if i < steps - 5:  # Don't add noise in final steps
                        noise_scale = 0.1 * (1.0 - i/steps)  # Gradually reduce noise
                        latents = latents + noise_scale * torch.randn_like(latents)
                    
                    # Apply normalization to prevent extreme values
                    if i % 5 == 0:  # Every few steps
                        latents = torch.nn.functional.normalize(latents, dim=1) * 4.0
            
            # Try to decode with VAE if available
            try_vae_decode = True
            decoded_image = None
            
            if try_vae_decode:
                try:
                    print("Attempting to decode with VAE...")
                    # Check if VAE is a meta tensor and handle accordingly
                    is_vae_meta = False
                    try:
                        # Try to access a parameter to check if it's a meta tensor
                        next(model.vae.parameters()).device
                    except NotImplementedError as e:
                        if "Cannot determine data pointer from a meta tensor" in str(e):
                            is_vae_meta = True
                            print("Detected meta tensor for VAE, using to_empty()")
                    
                    if is_vae_meta:
                        # For meta tensors, we need to use to_empty() first
                        from accelerate import init_empty_weights
                        with init_empty_weights():
                            # Clone the model structure without weights
                            vae_empty = type(model.vae)()
                        
                        # Move the empty model to the device
                        vae_empty = vae_empty.to(device)
                        
                        # Load the weights from the original model
                        vae_empty.load_state_dict(model.vae.state_dict())
                        
                        # Scale latents according to VAE requirements
                        scaled_latents = 1 / 0.18215 * latents  # Standard scaling factor for SD-based models
                        
                        # Decode the image
                        with torch.cuda.amp.autocast():
                            decoded_image = vae_empty.decode(scaled_latents).sample
                        
                        # Clean up to save memory
                        del vae_empty
                    else:
                        vae = model.vae.to(device)
                        
                        # Scale latents according to VAE requirements
                        scaled_latents = 1 / 0.18215 * latents  # Standard scaling factor for SD-based models
                        
                        # Decode the image
                        with torch.cuda.amp.autocast():
                            decoded_image = vae.decode(scaled_latents).sample
                        
                        # Move VAE back to CPU
                        vae.to('cpu')
                    
                    print("Successfully decoded with VAE!")
                except Exception as e:
                    print(f"Error decoding with VAE: {str(e)}")
                    print("Falling back to direct visualization of latents")
            
            # Process and save the image
            if decoded_image is not None:
                # Process VAE-decoded image
                image = (decoded_image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).float().numpy()
                
                # Convert to PIL Image
                from PIL import Image as PILImage
                image_pil = PILImage.fromarray((image[0] * 255).astype(np.uint8))
                
                # Save individual image
                image_filename = f"sample_{step}_prompt_{prompt_idx}.png"
                image_path = os.path.join(samples_dir, image_filename)
                image_pil.save(image_path)
                print(f"Saved VAE-decoded image to {image_path}")
                
                # Convert back to tensor for grid
                image_tensor = torch.from_numpy(image[0]).permute(2, 0, 1)
                all_images.append(image_tensor)
            else:
                # Fallback: Visualize latents directly with enhanced processing
                # Upscale the latents to 512x512 for better visualization
                upscaled_latents = torch.nn.functional.interpolate(
                    latents, 
                    size=(512, 512), 
                    mode='bilinear', 
                    align_corners=False
                )
                
                # Apply advanced normalization for better contrast
                vis_latents = upscaled_latents[0].cpu()
                
                # Apply histogram equalization-like normalization per channel
                for c in range(vis_latents.shape[0]):
                    channel = vis_latents[c]
                    sorted_channel = torch.sort(channel.flatten())[0]
                    min_val = sorted_channel[int(0.01 * sorted_channel.numel())]  # 1% percentile
                    max_val = sorted_channel[int(0.99 * sorted_channel.numel())]  # 99% percentile
                    vis_latents[c] = torch.clamp((channel - min_val) / (max_val - min_val + 1e-6), 0, 1)
                
                # Create RGB visualization with better color mapping
                rgb_image = torch.zeros(3, 512, 512)
                
                # Use different channels for better visualization
                # Map the first 3 latent channels to RGB
                rgb_image[0] = vis_latents[0]  # R channel
                rgb_image[1] = vis_latents[1]  # G channel
                rgb_image[2] = vis_latents[2]  # B channel
                
                # Apply advanced post-processing to enhance the image
                # Add contrast and saturation
                rgb_image = (rgb_image - 0.5) * 2.0 + 0.5  # Increase contrast
                rgb_image = torch.clamp(rgb_image, 0, 1)
                
                # Apply sharpening filter
                kernel = torch.tensor([
                    [-1, -1, -1],
                    [-1,  9, -1],
                    [-1, -1, -1]
                ], dtype=torch.float32) / 9.0
                kernel = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
                padded_image = F.pad(rgb_image.unsqueeze(0), (1, 1, 1, 1), mode='reflect')
                sharpened = F.conv2d(padded_image, kernel, groups=3)
                rgb_image = torch.clamp(sharpened[0], 0, 1)
                
                # Save the visualization
                image_filename = f"sample_{step}_prompt_{prompt_idx}.png"
                image_path = os.path.join(samples_dir, image_filename)
                save_image(rgb_image, image_path)
                print(f"Saved latent visualization to {image_path}")
                
                # Also save the raw latents for potential future decoding
                latent_path = os.path.join(samples_dir, f"latent_{prompt_idx}_{step}.pt")
                torch.save(latents.cpu(), latent_path)
                print(f"Saved raw latent to {latent_path}")
                
                # Convert back to tensor for grid
                all_images.append(rgb_image)
            
        except Exception as e:
            print(f"Error generating image for prompt '{prompt}': {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Create a grid of all images
    if all_images:
        try:
            grid = make_grid(all_images, nrow=min(4, len(all_images)))
            grid_path = os.path.join(samples_dir, f"grid_step_{step}.png")
            save_image(grid, grid_path)
            print(f"Saved grid image to {grid_path}")
            
            # Log to tensorboard if available
            if tb_writer is not None:
                tb_writer.add_image('samples', grid, step)
        except Exception as e:
            print(f"Error creating image grid: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Restore original model state
    if model_was_in_train:
        if hasattr(model, 'train'):
            model.train()
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'train'):
            model.transformer.train()
    
    # Return to training mode
    model.prepare_block_swap_training()
    
    return samples_dir if len(all_images) > 0 else None




def evaluate(model, model_engine, eval_dataloaders, tb_writer, step, eval_gradient_accumulation_steps, disable_block_swap=False):
    """
    Evaluate the model and generate sample images if configured.
    
    Args:
        model: The model to evaluate
        model_engine: The DeepSpeed engine
        eval_dataloaders: Dictionary of evaluation dataloaders
        tb_writer: TensorBoard writer
        step: Current training step
        eval_gradient_accumulation_steps: Gradient accumulation steps for evaluation
        disable_block_swap: Whether to disable block swapping during evaluation
    """
    if len(eval_dataloaders) == 0:
        print("No evaluation datasets configured, skipping evaluation metrics")
    else:
        # Clear CUDA cache before evaluation
        empty_cuda_cache()
        
        # Prepare model for inference
        model.prepare_block_swap_inference(disable_block_swap=disable_block_swap)
        
        # Run normal evaluation
        with torch.no_grad(), isolate_rng():
            seed = get_rank()
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Run the standard evaluation
            _evaluate(model_engine, eval_dataloaders, tb_writer, step, eval_gradient_accumulation_steps)
        
        # Restore model for training
        model.prepare_block_swap_training()
    
    # Check if we should generate inference samples
    inference_config = model.config.get('inference', {})
    current_epoch = model_engine.train_dataloader.epoch if hasattr(model_engine, 'train_dataloader') else 0
    
    print(f"Checking if inference should run at step {step}")
    print(f"Current epoch: {current_epoch}")
    print(f"Inference config: {inference_config}")
    
    should_generate = False
    
    # Check epoch-based inference
    if inference_config.get('inference_every_n_epochs', 0) > 0:
        epoch_check = current_epoch % inference_config['inference_every_n_epochs'] == 0
        print(f"Should generate based on epoch: {current_epoch} % {inference_config['inference_every_n_epochs']} == {epoch_check}")
        if epoch_check:
            should_generate = True
    
    # Check step-based inference
    if inference_config.get('inference_every_n_steps', 0) > 0:
        step_check = step % inference_config['inference_every_n_steps'] == 0
        print(f"Should generate based on step: {step} % {inference_config['inference_every_n_steps']} == {step_check}")
        if step_check:
            should_generate = True
    
    print(f"Final should_generate: {should_generate}")
    
    # Generate samples if configured
    if should_generate and inference_config.get('prompts'):
        # Clear CUDA cache before inference
        empty_cuda_cache()
        
        # Prepare model for inference
        model.prepare_block_swap_inference(disable_block_swap=disable_block_swap)
        
        with torch.no_grad(), isolate_rng():
            print(f"Attempting to generate samples at step {step} (epoch {current_epoch})...")
            try:
                output_path = generate_sample_images(model, model.config, step=step, tb_writer=tb_writer)
                if output_path:
                    print(f"Sample images saved to {output_path}")
                else:
                    print("No sample images were generated")
            except Exception as e:
                print(f"Error generating sample images: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Clear CUDA cache after inference
        empty_cuda_cache()
        
        # Restore model for training
        model.prepare_block_swap_training()


def distributed_init(args):
    """Initialize distributed training environment."""
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    rank = int(os.getenv('RANK', '0'))
    local_rank = args.local_rank

    # Set environment variables for distributed training
    os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = str(args.master_port)

    return world_size, rank, local_rank

def get_prodigy_d(optimizer):
    d = 0
    for group in optimizer.param_groups:
        d += group['d']
    return d / len(optimizer.param_groups)


if __name__ == '__main__':
    apply_patches()

    # needed for broadcasting Queue in dataset.py
    mp.current_process().authkey = b'afsaskgfdjh4'

    with open(args.config) as f:
        # Inline TOML tables are not pickleable, which messes up the multiprocessing dataset stuff. This is a workaround.
        config = json.loads(json.dumps(toml.load(f)))

    set_config_defaults(config)
    common.AUTOCAST_DTYPE = config['model']['dtype']
    
    # Log optimizer configuration
    if 'optimizer' in config:
        optimizer_config = config['optimizer']
        print("\n" + "="*50)
        print(f"Training with optimizer: {optimizer_config.get('type', 'default')}")
        print(f"Optimizer settings: {', '.join(f'{k}={v}' for k, v in optimizer_config.items() if k != 'type')}")
        print("="*50 + "\n")

    # Initialize distributed environment before deepspeed
    world_size, rank, local_rank = distributed_init(args)

    # Now initialize deepspeed
    deepspeed.init_distributed()

    # needed for broadcasting Queue in dataset.py
    torch.cuda.set_device(0)     # Simplified: rank is always 0

    resume_from_checkpoint = (
        args.resume_from_checkpoint if args.resume_from_checkpoint is not None
        else config.get('resume_from_checkpoint', False)
    )
    regenerate_cache = (
        args.regenerate_cache if args.regenerate_cache is not None
        else config.get('regenerate_cache', False)
    )

    # --- Simplified Model Selection ---
    model_type = config['model']['type']
    assert model_type == 'chroma', f"Configuration error: Expected model type 'chroma', but found '{model_type}' in {args.config}."

    from models import chroma
    model = chroma.ChromaPipeline(config)
    # --- End Simplified Model Selection ---


    # import sys, PIL
    # test_image = sys.argv[1]
    # with torch.no_grad():
    #     vae = model.get_vae().to('cuda')
    #     latents = dataset.encode_pil_to_latents(PIL.Image.open(test_image), vae)
    #     pil_image = dataset.decode_latents_to_pil(latents, vae)
    #     pil_image.save('test.jpg')
    # quit()

    with open(config['dataset']) as f:
        dataset_config = toml.load(f)
    gradient_release = config['optimizer'].get('gradient_release', False)
    ds_config = {
        'train_micro_batch_size_per_gpu': config.get('micro_batch_size_per_gpu', 1),
        'gradient_accumulation_steps': config.get('gradient_accumulation_steps', 1),
        # Can't do gradient clipping with gradient release, since there are no grads at the end of the step anymore.
        'gradient_clipping': 0. if gradient_release else config.get('gradient_clipping', 1.0),
        'steps_per_print': config.get('steps_per_print', 1),
    }
    caching_batch_size = config.get('caching_batch_size', 1)
    dataset_manager = dataset_util.DatasetManager(model, regenerate_cache=regenerate_cache, caching_batch_size=caching_batch_size)

    # Removed skip_dataset_validation=args.i_know_what_i_am_doing
    train_data = dataset_util.Dataset(dataset_config, model)
    dataset_manager.register(train_data)

    eval_data_map = {}
    for i, eval_dataset in enumerate(config['eval_datasets']):
        if type(eval_dataset) == str:
            name = f'eval{i}'
            config_path = eval_dataset
        else:
            name = eval_dataset['name']
            config_path = eval_dataset['config']
        with open(config_path) as f:
            eval_dataset_config = toml.load(f)
        # Removed skip_dataset_validation=args.i_know_what_i_am_doing
        eval_data_map[name] = dataset_util.Dataset(eval_dataset_config, model)
        dataset_manager.register(eval_data_map[name])

    # Video dumping test code commented out in original, remains commented out
    # ... (code remains commented out) ...
    # quit()

    # REMOVED: --dump_dataset functionality block
    # if args.dump_dataset:
    #     ... (entire block removed) ...
    #     # dist.barrier() # Would have been removed anyway
    #     quit()

    dataset_manager.cache()
    if args.cache_only:
        quit()

    model.load_diffusion_model()

    if adapter_config := config.get('adapter', None):
        init_from_existing = adapter_config.get('init_from_existing', None)
        # SDXL special case is removed as model is chroma
        model.configure_adapter(adapter_config) # Configure the LoRA adapter
        is_adapter = True
        if init_from_existing:
            # Assuming chroma doesn't use the diffusers method like SDXL might
            model.load_adapter_weights(init_from_existing)
    else:
        is_adapter = False

    # if this is a new run, create a new dir for it
    # No need for is_main_process check
    if not resume_from_checkpoint:
        run_dir = os.path.join(config['output_dir'], datetime.now(timezone.utc).strftime('%Y%m%d_%H-%M-%S'))
        os.makedirs(run_dir, exist_ok=True)
        shutil.copy(args.config, run_dir)
    # Removed dist.barrier()
    # wait for all processes then get the most recent dir (may have just been created)
    # Barrier was removed as it's a no-op for world size 1

    if resume_from_checkpoint is True:  # No specific folder provided, use most recent
        run_dir = get_most_recent_run_dir(config['output_dir'])
    elif isinstance(resume_from_checkpoint, str):  # Specific folder provided
        run_dir = os.path.join(config['output_dir'], resume_from_checkpoint)
        if not os.path.exists(run_dir):
            raise ValueError(f"Checkpoint directory {run_dir} does not exist")
    else:  # Not resuming, use most recent (newly created) dir
        run_dir = get_most_recent_run_dir(config['output_dir'])

    # Block swapping
    if blocks_to_swap := config.get('blocks_to_swap', 0):
        assert config['pipeline_stages'] == 1, 'Block swapping only works with pipeline_stages=1'
        assert 'adapter' in config, 'Block swapping only works when training LoRA'
        # Don't automatically move to GPU, we'll do that ourselves.
        def to(self, *args, **kwargs):
            pass
        deepspeed.pipe.PipelineModule.to = to
        model.enable_block_swap(blocks_to_swap)

    layers = model.to_layers()
    additional_pipeline_module_kwargs = {}
    activation_checkpointing = config['activation_checkpointing']
    if activation_checkpointing:
        if activation_checkpointing == True:
            # TODO: block swapping doesn't work with Deepspeed non-reentrant checkpoint, but PyTorch native one is fine. Some
            # weights end up on CPU where they shouldn't. Why? Are we giving anything up by not using the Deepspeed implementation?
            #checkpoint_func = deepspeed.checkpointing.non_reentrant_checkpoint
            from functools import partial
            checkpoint_func = partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
        elif activation_checkpointing == 'unsloth':
            checkpoint_func = unsloth_checkpoint
        else:
            raise NotImplementedError(f'activation_checkpointing={activation_checkpointing} is not implemented')
        additional_pipeline_module_kwargs.update({
            'activation_checkpoint_interval': 1,
            'checkpointable_layers': model.checkpointable_layers,
            'activation_checkpoint_func': checkpoint_func,
        })

    num_stages = config.get('pipeline_stages', 1) # Should be 1
    assert num_stages == 1, f"Expected pipeline_stages=1 for single GPU chroma training, found {num_stages}"
    partition_method=config.get('partition_method', 'parameters')
    partition_split = config.get('partition_split',[len(layers) / num_stages])
    pipeline_model = ManualPipelineModule(
        layers=layers,
        num_stages=num_stages,
        partition_method=partition_method,
        manual_partition_split=partition_split,
        loss_fn=model.get_loss_fn(),
        **additional_pipeline_module_kwargs
    )
    parameters_to_train = [p for p in pipeline_model.parameters() if p.requires_grad]

    def get_optimizer(model_parameters):
        optim_config = config['optimizer']
        optim_type = optim_config['type']
        optim_type_lower = optim_type.lower()

        args = []
        kwargs = {k: v for k, v in optim_config.items() if k not in ['type', 'gradient_release']}

        if optim_type_lower == 'adamw':
            # TODO: fix this. I'm getting "fatal error: cuda_runtime.h: No such file or directory"
            # when Deepspeed tries to build the fused Adam extension.
            # klass = deepspeed.ops.adam.FusedAdam
            klass = torch.optim.AdamW
        elif optim_type_lower == 'adamw8bit':
            import bitsandbytes
            klass = bitsandbytes.optim.AdamW8bit
        elif optim_type_lower == 'adamw_optimi':
            import optimi
            klass = optimi.AdamW
        elif optim_type_lower == 'stableadamw':
            import optimi
            klass = optimi.StableAdamW
        elif optim_type_lower == 'sgd':
            klass = torch.optim.SGD
        elif optim_type_lower == 'adamw8bitkahan':
            from optimizers import adamw_8bit
            klass = adamw_8bit.AdamW8bitKahan
        elif optim_type_lower == 'scorn':
            from optimizers.scorn import SCORN
            klass = SCORN
        elif optim_type_lower == 'adafactor':
            from optimizers.adafactor import EnhancedAdafactor
            klass = EnhancedAdafactor
        elif optim_type_lower == 'remaster':
            from optimizers.remaster import REMASTER
            klass = REMASTER
        elif optim_type_lower == 'persona':
            from optimizers.persona import PersonaOptimizer
            klass = PersonaOptimizer
        elif optim_type_lower == 'offload':
            from torchao.prototype.low_bit_optim import CPUOffloadOptimizer
            klass = CPUOffloadOptimizer
            args.append(torch.optim.AdamW)
            kwargs['fused'] = True
        else:
            import pytorch_optimizer
            klass = getattr(pytorch_optimizer, optim_type)

        if optim_config.get('gradient_release', False):
            # Prevent deepspeed from logging every single param group lr
            def _report_progress(self, step):
                lr = self.get_lr()
                mom = self.get_mom()
                deepspeed.utils.logging.log_dist(f"step={step}, skipped={self.skipped_steps}, lr={lr[0]}, mom={mom[0]}", ranks=[0])
            deepspeed.runtime.engine.DeepSpeedEngine._report_progress = _report_progress

            # Deepspeed executes all the code to reduce grads across data parallel ranks even if the DP world size is 1.
            # As part of this, any grads that are None are set to zeros. We're doing gradient release to save memory,
            # so we have to avoid this.
            def _exec_reduce_grads(self):
                return
            deepspeed.runtime.pipe.engine.PipelineEngine._INSTRUCTION_MAP[deepspeed.runtime.pipe.schedule.ReduceGrads] = _exec_reduce_grads

            # When pipelining multiple forward and backward passes, normally updating the parameter in-place causes an error when calling
            # backward() on future micro-batches. But we can modify .data directly so the autograd engine doesn't detect in-place modifications.
            # TODO: this is unbelievably hacky and not mathematically sound, I'm just seeing if it works at all.
            def add_(self, *args, **kwargs):
                self.data.add_(*args, **kwargs)
            for p in model_parameters:
                p.add_ = add_.__get__(p)

            if 'foreach' in inspect.signature(klass).parameters:
                kwargs['foreach'] = False

            # We're doing an optimizer step for each micro-batch. Scale momentum and EMA betas so that the contribution
            # decays at the same rate it would if we were doing one step per batch like normal.
            # Reference: https://alexeytochin.github.io/posts/batch_size_vs_momentum/batch_size_vs_momentum.html
            gas = ds_config['gradient_accumulation_steps']
            if 'betas' in kwargs:
                for i in range(len(kwargs['betas'])):
                    kwargs['betas'][i] = kwargs['betas'][i] ** (1/gas)
            if 'momentum' in kwargs:
                kwargs['momentum'] = kwargs['momentum'] ** (1/gas)

            optimizer_dict = {}
            for pg in model.get_param_groups(model_parameters):
                param_kwargs = kwargs.copy()
                if isinstance(pg, dict):
                    # param group
                    for p in pg['params']:
                        param_kwargs['lr'] = pg['lr']
                        optimizer_dict[p] = klass([p], **param_kwargs)
                else:
                    # param
                    optimizer_dict[pg] = klass([pg], **param_kwargs)

            def optimizer_hook(p):
                optimizer_dict[p].step()
                optimizer_dict[p].zero_grad()

            for p in model_parameters:
                p.register_post_accumulate_grad_hook(optimizer_hook)

            from optimizers import gradient_release
            return gradient_release.GradientReleaseOptimizerWrapper(list(optimizer_dict.values()))
        else:
            model_parameters = model.get_param_groups(model_parameters)
            return klass(model_parameters, *args, **kwargs)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=pipeline_model,
        model_parameters=parameters_to_train,
        optimizer=get_optimizer,
        config=ds_config,
    )
    model.model_engine = model_engine
    
    # Log optimizer information after initialization
    print(f"\n[INFO] Optimizer successfully initialized: {optimizer.__class__.__name__}")
    print(f"[INFO] Parameter groups: {len(optimizer.param_groups)}")
    for i, pg in enumerate(optimizer.param_groups):
        param_count = sum(p.numel() for p in pg['params'])
        print(f"[INFO] Group {i}: {param_count} parameters with learning rate: {pg.get('lr', 'default')}")
    print("")

    # Simplified: model_engine.is_pipe_parallel will be False for num_stages=1
    # The block 'if model_engine.is_pipe_parallel:' is removed.

    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    if config['warmup_steps'] > 0:
        warmup_steps = config['warmup_steps']
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1/warmup_steps, total_iters=warmup_steps)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, lr_scheduler], milestones=[warmup_steps])
    model_engine.lr_scheduler = lr_scheduler

    # Simplified data parallel rank (0) and world size (1)
    train_data.post_init(
        0, # model_engine.grid.get_data_parallel_rank(),
        1, # model_engine.grid.get_data_parallel_world_size(),
        model_engine.train_micro_batch_size_per_gpu(),
        model_engine.gradient_accumulation_steps(),
    )
    for eval_data in eval_data_map.values():
        # Simplified data parallel rank (0) and world size (1)
        eval_data.post_init(
            0, # model_engine.grid.get_data_parallel_rank(),
            1, # model_engine.grid.get_data_parallel_world_size(),
            config.get('eval_micro_batch_size_per_gpu', model_engine.train_micro_batch_size_per_gpu()),
            config['eval_gradient_accumulation_steps'],
        )

    # Might be useful because we set things in fp16 / bf16 without explicitly enabling Deepspeed fp16 mode.
    # Unsure if really needed.
    communication_data_type = config['lora']['dtype'] if 'lora' in config else config['model']['dtype']
    model_engine.communication_data_type = communication_data_type

    train_dataloader = dataset_util.PipelineDataLoader(train_data, model_engine, model_engine.gradient_accumulation_steps(), model)

    step = 1
    # make sure to do this before calling model_engine.set_dataloader(), as that method creates an iterator
    # which starts creating dataloader internal state
    if resume_from_checkpoint:
        load_path, client_state = model_engine.load_checkpoint(
            run_dir,
            load_module_strict=False,
            load_lr_scheduler_states='force_constant_lr' not in config,
        )
        # Removed dist.barrier()
        assert load_path is not None
        train_dataloader.load_state_dict(client_state['custom_loader'])
        step = client_state['step'] + 1
        del client_state
        # Always main process
        print(f'Resuming training from checkpoint. Resuming at epoch: {train_dataloader.epoch}, step: {step}')

    if 'force_constant_lr' in config:
        model_engine.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        for pg in optimizer.param_groups:
            pg['lr'] = config['force_constant_lr']

    steps_per_epoch = len(train_dataloader) // model_engine.gradient_accumulation_steps()
    model_engine.total_steps = steps_per_epoch * config['epochs']

    eval_dataloaders = {
        # Set num_dataloader_workers=0 so dataset iteration is completely deterministic.
        # We want the exact same noise for each image, each time, for a stable validation loss.
        name: dataset_util.PipelineDataLoader(eval_data, model_engine, config['eval_gradient_accumulation_steps'], model, num_dataloader_workers=0)
        for name, eval_data in eval_data_map.items()
    }

    epoch = train_dataloader.epoch
    # Always main process, so tb_writer is always created (or None if SummaryWriter fails)
    tb_writer = SummaryWriter(log_dir=run_dir)
    saver = utils.saver.Saver(args, config, is_adapter, run_dir, model, train_dataloader, model_engine, pipeline_model)

    disable_block_swap_for_eval = config.get('disable_block_swap_for_eval', False)
    if config['eval_before_first_step'] and not resume_from_checkpoint:
        evaluate(model, model_engine, eval_dataloaders, tb_writer, 0, config['eval_gradient_accumulation_steps'], disable_block_swap_for_eval)

    # TODO: this is state we need to save and resume when resuming from checkpoint. It only affects logging.
    epoch_loss = 0
    num_steps = 0
    while True:
        #empty_cuda_cache()
        model_engine.reset_activation_shape()
        iterator = get_data_iterator_for_step(train_dataloader, model_engine)
        loss = model_engine.train_batch(iterator).item()
        epoch_loss += loss
        num_steps += 1
        train_dataloader.sync_epoch()

        new_epoch, checkpointed, saved = saver.process_epoch(epoch, step)
        finished_epoch = True if new_epoch != epoch else False

        # Always main process
        if step % config['logging_steps'] == 0:
            tb_writer.add_scalar(f'train/loss', loss, step)
            if optimizer.__class__.__name__ == 'Prodigy':
                prodigy_d = get_prodigy_d(optimizer)
                tb_writer.add_scalar(f'train/prodigy_d', prodigy_d, step)

        if (config['eval_every_n_steps'] and step % config['eval_every_n_steps'] == 0) or \
           (finished_epoch and config['eval_every_n_epochs'] and epoch % config['eval_every_n_epochs'] == 0):
            evaluate(model, model_engine, eval_dataloaders, tb_writer, step, config['eval_gradient_accumulation_steps'], disable_block_swap_for_eval)

        if finished_epoch:
            tb_writer.add_scalar(f'train/epoch_loss', epoch_loss/num_steps, epoch)
            epoch_loss = 0
            num_steps = 0
            epoch = new_epoch
            if epoch is None: # Check if training finished (saver returns None)
                break

        saver.process_step(step)
        step += 1

    # Save final training state checkpoint and model, unless we just saved them.
    if not checkpointed:
        saver.save_checkpoint(step)
    if not saved:
        saver.save_model(f'epoch{epoch}')

    # Always main process
    print('TRAINING COMPLETE!')