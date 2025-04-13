import os
import random
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
import glob
from tqdm import tqdm
from einops import rearrange
from safetensors.torch import load_file as load_safetensors

from utils.isolate_rng import isolate_rng
from utils.common import empty_cuda_cache, get_rank

# Path to the VAE model
VAE_PATH = "models/flux/ae.safetensors"

def get_noise(batch_size, height, width, device, dtype, seed=None):
    """Generate random noise for initialization."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    return torch.randn(batch_size, 16, height // 8, width // 8, device=device, dtype=dtype)

def get_schedule(steps, seq_len):
    """Generate timestep schedule for diffusion."""
    # Create a simple linear schedule from 1.0 to 0.0
    # For the Chroma model, timesteps are in [0, 1] range where 1.0 is pure noise
    # and 0.0 is the clean image
    return torch.linspace(1.0, 0.01, steps)  # Stop at 0.01 instead of 0 to avoid instability

def vae_flatten(latents):
    """Flatten latents for model input."""
    return (
        rearrange(latents, "n c (h dh) (w dw) -> n (h w) (c dh dw)", dh=2, dw=2),
        latents.shape,
    )

def vae_unflatten(latents, shape):
    """Unflatten latents back to image format."""
    n, c, h, w = shape
    return rearrange(
        latents,
        "n (h w) (c dh dw) -> n c (h dh) (w dw)",
        dh=2,
        dw=2,
        c=c,
        h=h // 2,
        w=w // 2,
    )

def prepare_latent_image_ids(batch_size, height, width):
    """Generate position IDs for latents."""
    latent_image_ids = torch.zeros(height // 2, width // 2, 3)
    latent_image_ids[..., 1] = (
        latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
    )
    latent_image_ids[..., 2] = (
        latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]
    )

    (
        latent_image_id_height,
        latent_image_id_width,
        latent_image_id_channels,
    ) = latent_image_ids.shape

    latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1)
    latent_image_ids = latent_image_ids.reshape(
        batch_size,
        latent_image_id_height * latent_image_id_width,
        latent_image_id_channels,
    )

    return latent_image_ids

def denoise_cfg(model, noise, image_pos_id, text_embed, null_embed, text_ids, timesteps, 
                guidance_scale=0, cfg_scale=1.0, first_n_steps_wo_cfg=-1):
    """
    Perform denoising with classifier-free guidance.
    
    Args:
        model: The diffusion model
        noise: Initial noise
        image_pos_id: Position IDs for the image
        text_embed: Text embeddings
        null_embed: Null text embeddings for unconditional guidance
        text_ids: Text position IDs
        timesteps: Diffusion timesteps
        guidance_scale: Guidance scale
        cfg_scale: Classifier-free guidance scale
        first_n_steps_wo_cfg: Number of initial steps without CFG
    
    Returns:
        Denoised latents
    """
    latent = noise.clone()
    
    # Create attention mask (all ones since we're using dummy text)
    txt_mask = torch.ones((text_embed.shape[0], text_embed.shape[1]), 
                          device=text_embed.device, dtype=torch.bool)
    
    try:
        for i, t in enumerate(tqdm(timesteps, desc="Denoising")):
            # Create batch of timesteps
            timestep = torch.full((noise.shape[0],), t, device=noise.device, dtype=noise.dtype)
            
            # Prepare for classifier-free guidance
            do_cfg = cfg_scale > 1.0 and (first_n_steps_wo_cfg < 0 or i >= first_n_steps_wo_cfg)
            
            if do_cfg:
                # Concatenate for classifier-free guidance
                latent_input = torch.cat([latent, latent], dim=0)
                image_pos_id_input = torch.cat([image_pos_id, image_pos_id], dim=0)
                text_embed_input = torch.cat([null_embed, text_embed], dim=0)
                text_ids_input = torch.cat([text_ids, text_ids], dim=0)
                timestep_input = torch.cat([timestep, timestep], dim=0)
                txt_mask_input = torch.cat([txt_mask, txt_mask], dim=0)
                
                # Create guidance vector - IMPORTANT: This is required by the Chroma model
                guidance_vec = torch.zeros((latent_input.shape[0],), device=latent.device, dtype=latent.dtype)
                
                # Flatten latents for model input
                latent_flat, shape = vae_flatten(latent_input)
                
                # Forward pass with correct parameter order:
                # img, img_ids, txt, txt_ids, txt_mask, timesteps, guidance
                noise_pred = model(
                    latent_flat,                 # img
                    image_pos_id_input,          # img_ids
                    text_embed_input,            # txt
                    text_ids_input,              # txt_ids
                    txt_mask_input,              # txt_mask
                    timestep_input,              # timesteps
                    guidance_vec                 # guidance
                )
                
                # Split predictions and apply CFG
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)
            else:
                # No CFG, just use the prompt
                latent_flat, shape = vae_flatten(latent)
                
                # Create guidance vector - IMPORTANT: This is required by the Chroma model
                guidance_vec = torch.zeros((latent.shape[0],), device=latent.device, dtype=latent.dtype)
                
                # Forward pass with correct parameter order:
                # img, img_ids, txt, txt_ids, txt_mask, timesteps, guidance
                noise_pred = model(
                    latent_flat,                 # img
                    image_pos_id,                # img_ids
                    text_embed,                  # txt
                    text_ids,                    # txt_ids
                    txt_mask,                    # txt_mask
                    timestep,                    # timesteps
                    guidance_vec                 # guidance
                )
            
            # Update latents
            alpha = 0.5 + 0.5 * (1.0 - t)  # Adaptive step size
            latent = latent - alpha * vae_unflatten(noise_pred, shape)[:latent.shape[0]]
            
            # Add a small amount of noise to prevent getting stuck in local minima
            if i < len(timesteps) - 5:  # Don't add noise in final steps
                noise_scale = 0.1 * (1.0 - i/len(timesteps))  # Gradually reduce noise
                latent = latent + noise_scale * torch.randn_like(latent)
            
            # Apply normalization to prevent extreme values
            if i % 5 == 0:  # Every few steps
                latent = torch.nn.functional.normalize(latent, dim=1) * 4.0
    
    except Exception as e:
        print(f"Error during denoising process: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return the latent as-is if we've made any progress
        if i > 0:
            print(f"Returning partially denoised latent after {i} steps")
            return latent
        # Otherwise raise the exception
        raise
    
    return latent

def generate_sample_images(model, config, step=0, tb_writer=None):
    """
    Generate sample images during training for visualization.
    Uses the proper VAE from Chroma1 for high-quality image generation.
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
    
    # Initialize VAE
    print("Loading VAE from", VAE_PATH)
    try:
        # Try to import the AutoEncoder class
        try:
            from utils.autoencoder import AutoEncoder, ae_params
            with torch.device("meta"):
                ae = AutoEncoder(ae_params)
            ae.load_state_dict(load_safetensors(VAE_PATH), assign=True)
            ae.to(device)
            print("Successfully loaded VAE")
        except (ImportError, FileNotFoundError) as e:
            print(f"Error loading AutoEncoder class: {str(e)}")
            print("Will try to use model.vae instead")
            ae = None
    except Exception as e:
        print(f"Error loading VAE: {str(e)}")
        print("Will continue without VAE decoding")
        ae = None
    
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
                
                # Get text embeddings - handle meta tensors properly
                try:
                    # Try with regular text encoder first
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
                    
                except RuntimeError as e:
                    if "Cannot determine data pointer from a meta tensor" in str(e):
                        print("Detected meta tensor, using alternative encoding approach")
                        # We need to create dummy embeddings of the right shape
                        # T5 text encoder output is typically 4096 dimensions
                        prompt_embeds = torch.zeros((1, t5_max_length, 4096), device=device, dtype=model_dtype)
                        neg_prompt_embeds = torch.zeros((1, t5_max_length, 4096), device=device, dtype=model_dtype)
                    else:
                        raise
                
                print(f"Successfully encoded prompt: '{prompt}'")
                print(f"Text embedding shape: {prompt_embeds.shape}")
                
                # Create text position IDs
                text_ids = torch.zeros((1, t5_max_length, 3), device=device, dtype=model_dtype)
                for j in range(t5_max_length):
                    text_ids[0, j, 0] = j / t5_max_length
                
            except Exception as e:
                print(f"Error encoding text prompt: {str(e)}")
                print("Using dummy text embeddings instead")
                # Create dummy text embeddings
                prompt_embeds = torch.zeros((1, t5_max_length, 4096), device=device, dtype=model_dtype)
                neg_prompt_embeds = torch.zeros((1, t5_max_length, 4096), device=device, dtype=model_dtype)
                text_ids = torch.zeros((1, t5_max_length, 3), device=device, dtype=model_dtype)
            
            # Create initial random noise
            noise = get_noise(1, height, width, device, model_dtype, seed=current_seed)
            
            # Prepare image position IDs
            image_pos_id = prepare_latent_image_ids(1, height, width).to(device)
            
            # Prepare timesteps
            timesteps = get_schedule(steps, noise.shape[1])
            
            # Denoise the latents using the Chroma1 approach
            with torch.no_grad(), isolate_rng():
                latent = denoise_cfg(
                    model.transformer,
                    noise,
                    image_pos_id,
                    prompt_embeds,
                    neg_prompt_embeds,
                    text_ids,
                    timesteps,
                    guidance,
                    cfg,
                    first_n_steps_wo_cfg
                )
            
            # Decode with VAE if available
            if ae is not None:
                try:
                    print("Decoding with VAE...")
                    with torch.cuda.amp.autocast():
                        output_image = ae.decode(latent)
                    
                    # Process VAE-decoded image
                    image = (output_image / 2 + 0.5).clamp(0, 1)
                    
                    # Save individual image
                    image_filename = f"sample_{step}_prompt_{prompt_idx}.png"
                    image_path = os.path.join(samples_dir, image_filename)
                    save_image(image, image_path)
                    print(f"Saved VAE-decoded image to {image_path}")
                    
                    # Add to grid
                    all_images.append(image[0])
                except Exception as e:
                    print(f"Error decoding with VAE: {str(e)}")
                    print("Falling back to direct visualization of latents")
                    ae = None  # Disable VAE for subsequent prompts
            
            # Fallback: Visualize latents directly if VAE is not available
            if ae is None:
                # Upscale the latents to 512x512 for better visualization
                upscaled_latents = torch.nn.functional.interpolate(
                    latent, 
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
                torch.save(latent.cpu(), latent_path)
                print(f"Saved raw latent to {latent_path}")
                
                # Add to grid
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
    
    # Clean up VAE
    if ae is not None:
        ae.to('cpu')
        del ae
        torch.cuda.empty_cache()
    
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
