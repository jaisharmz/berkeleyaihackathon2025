import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPProcessor
from PIL import Image
import os
from tqdm.auto import tqdm
import numpy as np

# --- Configuration ---
# Models
MODEL_A_ID = "runwayml/stable-diffusion-v1-5"
MODEL_B_ID = "runwayml/stable-diffusion-inpainting"
CLIP_ID = "openai/clip-vit-large-patch14"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Generation Parameters
PROMPT = "a majestic fantasy castle in a lush valley, high quality, 4k, digital art"
NUM_INFERENCE_STEPS = 50  # Total number of diffusion steps
GUIDANCE_SCALE = 8.0      # How much to adhere to the prompt

# RL / Bandit Parameters
DECISION_INTERVAL = 5 # How many steps to run a model before re-evaluating
EPSILON = 0.3         # Probability of choosing a random model (exploration)
OUTPUT_DIR = "inpainting_output"

def decode_latents_to_pil(vae, latents, dtype):
    """Helper function to decode latents into a PIL image."""
    with torch.no_grad():
        # Move latents to CPU for decoding if they are on CUDA
        latents_for_decode = latents.to(dtype)
        image = vae.decode(1 / 0.18215 * latents_for_decode).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return Image.fromarray((image[0] * 255).round().astype(np.uint8))

def main():
    """
    Main function to run the multi-model generation process and individual model benchmarks.
    """
    print(f"Using device: {DEVICE}")

    # --- 1. Load Models ---
    print("Loading models...")
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_A_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_A_ID, subfolder="text_encoder", torch_dtype=DTYPE).to(DEVICE)
    vae = AutoencoderKL.from_pretrained(MODEL_A_ID, subfolder="vae", torch_dtype=DTYPE).to(DEVICE)
    
    # Model A & B
    unet_A = UNet2DConditionModel.from_pretrained(MODEL_A_ID, subfolder="unet", torch_dtype=DTYPE).to(DEVICE)
    unet_B = UNet2DConditionModel.from_pretrained(MODEL_B_ID, subfolder="unet", torch_dtype=DTYPE).to(DEVICE)

    scheduler = PNDMScheduler.from_pretrained(MODEL_A_ID, subfolder="scheduler")

    # Reward Model
    clip_processor = CLIPProcessor.from_pretrained(CLIP_ID)
    clip_model = CLIPModel.from_pretrained(CLIP_ID, torch_dtype=DTYPE).to(DEVICE)
    print("Models loaded.")

    # --- 2. Helper function for CLIP score ---
    def calculate_clip_score(image, text):
        with torch.no_grad():
            inputs = clip_processor(text=[text], images=[image], return_tensors="pt", padding=True)
            inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k,v in inputs.items()}
            if DTYPE == torch.float16:
                inputs['pixel_values'] = inputs['pixel_values'].to(DTYPE)
            outputs = clip_model(**inputs)
            return outputs.logits_per_image.item()

    # --- 3. Prepare for Generation ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    text_input = tokenizer(PROMPT, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(DEVICE))[0].to(DTYPE)
    uncond_input = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(DEVICE))[0].to(DTYPE)
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    height, width = 512, 512
    generator = torch.manual_seed(42)
    initial_latents = torch.randn(
        (1, unet_A.config.in_channels, height // 8, width // 8),
        generator=generator,
        device='cpu'
    ).to(DEVICE, dtype=DTYPE)

    scheduler.set_timesteps(NUM_INFERENCE_STEPS)

    # --- 4. Mixed-Model Generation (RL Agent with IMPROVED REWARDS) ---
    print("\n--- Starting Mixed-Model Generation (RL Agent with Improved Rewards) ---")
    latents = initial_latents.clone()
    models = {"A": unet_A, "B": unet_B}
    model_names = list(models.keys())
    model_rewards = {name: 0.0 for name in model_names}
    model_counts = {name: 0 for name in model_names}
    choice_history = []
    
    # **MODIFICATION**: Initialize last_clip_score to track score changes
    last_clip_score = 0.0

    for i, t in enumerate(tqdm(scheduler.timesteps, desc="Mixed Model")):
        if i % DECISION_INTERVAL == 0:
            if torch.rand(1).item() < EPSILON or all(c == 0 for c in model_counts.values()):
                chosen_model_name = model_names[torch.randint(0, len(model_names), (1,)).item()]
            else:
                avg_rewards = {name: model_rewards[name] / model_counts[name] for name in model_names if model_counts[name] > 0}
                chosen_model_name = max(avg_rewards, key=avg_rewards.get)
        
        choice_history.append(chosen_model_name)
        unet = models[chosen_model_name]

        if chosen_model_name == 'B':
            mask = torch.zeros_like(latents[:, :1])
            masked_image_latents = torch.zeros_like(latents)
            nine_channel_latents = torch.cat([latents, mask, masked_image_latents], dim=1)
            latent_model_input = torch.cat([nine_channel_latents] * 2)
        else:
            latent_model_input = torch.cat([latents] * 2)

        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

        if (i + 1) % DECISION_INTERVAL == 0 and i < NUM_INFERENCE_STEPS - 1:
            pil_image = decode_latents_to_pil(vae, latents, DTYPE)
            current_score = calculate_clip_score(pil_image, PROMPT)
            
            # **MODIFICATION**: Reward is the IMPROVEMENT in score
            if last_clip_score == 0.0: # For the very first interval
                 reward = current_score
            else:
                 reward = current_score - last_clip_score
            
            last_clip_score = current_score # Update for the next interval

            model_counts[chosen_model_name] += 1
            model_rewards[chosen_model_name] += reward
    
    final_image_mixed = decode_latents_to_pil(vae, latents, DTYPE)
    output_path = os.path.join(OUTPUT_DIR, "final_image_mixed.png")
    final_image_mixed.save(output_path)
    print(f"Mixed model image saved to {output_path}")

    history_path = os.path.join(OUTPUT_DIR, "choice_history.txt")
    with open(history_path, 'w') as f:
        f.write(f"Prompt: {PROMPT}\n\nFinal Rewards (Improvement): {model_rewards}\nCounts: {model_counts}\n\nSequence: {''.join(choice_history)}")
    
    # --- 5. Single-Model Generation (Model A only) ---
    print("\n--- Starting Single-Model Generation (Model A) ---")
    latents_a = initial_latents.clone()
    for t in tqdm(scheduler.timesteps, desc="Model A"):
        latent_model_input = torch.cat([latents_a] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        with torch.no_grad():
            noise_pred = unet_A(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)
        latents_a = scheduler.step(noise_pred, t, latents_a).prev_sample

    final_image_a = decode_latents_to_pil(vae, latents_a, DTYPE)
    output_path_a = os.path.join(OUTPUT_DIR, "final_image_model_A.png")
    final_image_a.save(output_path_a)
    print(f"Model A image saved to {output_path_a}")

    # --- 6. Single-Model Generation (Model B only) ---
    print("\n--- Starting Single-Model Generation (Model B) ---")
    latents_b = initial_latents.clone()
    for t in tqdm(scheduler.timesteps, desc="Model B"):
        mask = torch.zeros_like(latents_b[:, :1])
        masked_image_latents = torch.zeros_like(latents_b)
        nine_channel_latents = torch.cat([latents_b, mask, masked_image_latents], dim=1)
        latent_model_input = torch.cat([nine_channel_latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        with torch.no_grad():
            noise_pred = unet_B(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)
        latents_b = scheduler.step(noise_pred, t, latents_b).prev_sample

    final_image_b = decode_latents_to_pil(vae, latents_b, DTYPE)
    output_path_b = os.path.join(OUTPUT_DIR, "final_image_model_B.png")
    final_image_b.save(output_path_b)
    print(f"Model B image saved to {output_path_b}")

if __name__ == "__main__":
    main()
