# main.py
# This script is a proof-of-concept implementation of the idea to use a 
# Reinforcement Learning agent (a Multi-Armed Bandit) to guide the
# generative process of diffusion models.

# --- 1. Imports and Setup ---
# We need torch for tensor operations, diffusers for the models, PIL for image
# handling, and transformers for our CLIP-based reward model.
import torch
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import os

# --- 2. The Reward Component (CLIP Scorer) ---
# This class is responsible for calculating the reward. It takes a generated
# image and a text prompt and returns a score based on how well they match,
# according to the CLIP model.
class CLIPReward:
    """
    A class to calculate the CLIP similarity score between an image and a text prompt.
    This score will be used as the reward signal for our RL agent.
    """
    def __init__(self, device="cuda"):
        self.device = device
        self.model_id = "openai/clip-vit-large-patch14"
        try:
            self.model = CLIPModel.from_pretrained(self.model_id).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_id)
            print("CLIP Reward model loaded successfully.")
        except Exception as e:
            print(f"Error loading CLIP model: {e}")
            print("CLIPReward will not function. Please check your connection and dependencies.")
            self.model = None
            self.processor = None

    def get_reward(self, image: Image.Image, prompt: str) -> float:
        """
        Calculates the similarity score.
        Args:
            image: The PIL Image to be scored.
            prompt: The text prompt to compare against.
        Returns:
            A floating-point score. We multiply by 100 for a more usable range.
        """
        if self.model is None:
            return 0.0 # Return a neutral reward if the model failed to load

        try:
            # Ensure the image is in RGB format
            if image.mode != "RGB":
                image = image.convert("RGB")

            inputs = self.processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # The logits_per_image represents the cosine similarity
            logits_per_image = outputs.logits_per_image
            score = logits_per_image.item()
            return score * 100.0
        except Exception as e:
            print(f"Error during CLIP reward calculation: {e}")
            return 0.0

# --- 3. The RL Agent (Multi-Armed Bandit) ---
# This is a simple Upper-Confidence-Bound (UCB1) bandit. It's a classic
# algorithm that balances exploring different models ("arms") to learn their
# effectiveness and exploiting the one that seems best so far.
class UCBAgent:
    """
    A simple UCB1 Multi-Armed Bandit to select which model to use.
    It maintains counts and average rewards for each model (arm).
    """
    def __init__(self, num_arms: int, c: float = 2.0):
        self.num_arms = num_arms
        self.c = c  # Exploration parameter
        self.counts = [0] * num_arms
        self.values = [0.0] * num_arms
        self.total_steps = 0

    def select_arm(self) -> int:
        """
        Selects an arm (model) based on the UCB1 formula.
        It prioritizes arms that haven't been tried much or have high average rewards.
        """
        self.total_steps += 1
        
        # First, try each arm once to initialize
        for arm in range(self.num_arms):
            if self.counts[arm] == 0:
                return arm

        # Calculate UCB values for each arm
        ucb_values = [0.0] * self.num_arms
        for arm in range(self.num_arms):
            # Add a small epsilon to the count to avoid division by zero if a model is never chosen after initialization
            count = self.counts[arm] if self.counts[arm] > 0 else 1e-5
            bonus = self.c * np.sqrt(np.log(self.total_steps) / count)
            ucb_values[arm] = self.values[arm] + bonus
        
        # Select the arm with the highest UCB value
        return int(np.argmax(ucb_values))

    def update(self, arm: int, reward: float):
        """
        Updates the statistics for the chosen arm after receiving a reward.
        """
        self.counts[arm] += 1
        n = self.counts[arm]
        old_value = self.values[arm]
        # Update the average reward incrementally
        self.values[arm] = ((n - 1) / n) * old_value + (1 / n) * reward

# --- 4. The Main Orchestration Logic ---
def run_rl_guided_generation(prompt: str, num_inference_steps: int = 50, device: str = "cuda"):
    """
    The main function that orchestrates the generation process.
    """
    print("--- Starting Generation ---")
    
    # --- Step A: Load all necessary models ---
    print("Loading models...")
    
    # Model A: A general-purpose, high-quality model
    model_id_a = "runwayml/stable-diffusion-v1-5"
    # Model B: A model fine-tuned for a specific style (e.g., anime/fantasy)
    model_id_b = "./dreamlike-diffusion-1.0"

    try:
        pipe_a = AutoPipelineForText2Image.from_pretrained(model_id_a, torch_dtype=torch.float16).to(device)
        pipe_b = AutoPipelineForText2Image.from_pretrained(model_id_b, torch_dtype=torch.float16).to(device)
        print("Specialized diffusion models loaded.")
    except Exception as e:
        print(f"Fatal error loading diffusion models: {e}. Aborting.")
        return

    tokenizer = pipe_a.tokenizer
    text_encoder = pipe_a.text_encoder
    
    # Get the scheduler configuration once, to be used for all generation loops.
    scheduler_config = pipe_a.scheduler.config

    models = [
        {"name": "Generalist (SD 1.5)", "unet": pipe_a.unet, "vae": pipe_a.vae},
        {"name": "Stylized (Dreamlike)", "unet": pipe_b.unet, "vae": pipe_b.vae}
    ]

    clip_reward_model = CLIPReward(device=device)
    rl_agent = UCBAgent(num_arms=len(models))
    
    # --- Step B: Prepare for the generation loop ---
    print("\n--- Preparing for Generation ---")
    
    # Encode the prompt for classifier-free guidance
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        prompt_embeds = text_encoder(text_input.input_ids.to(device))[0]
    
    negative_prompt = "" 
    negative_prompt_input = tokenizer(negative_prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        negative_prompt_embeds = text_encoder(negative_prompt_input.input_ids.to(device))[0]

    text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds])

    # Create ONE initial random noise tensor to be used by all generation processes for a fair comparison.
    generator = torch.Generator(device=device).manual_seed(42) # Use a generator for reproducibility
    latents = torch.randn(
        (1, pipe_a.unet.config.in_channels, pipe_a.unet.config.sample_size, pipe_a.unet.config.sample_size),
        generator=generator,
        device=device,
        dtype=torch.float16
    )
    
    # --- Step C: Generate Baseline Images (Model A only and Model B only) ---
    # We clone the initial latents so that each generation process starts from the exact same point.

    # Generate with Model A only
    print("\n--- Generating Baseline Image (Model A Only) ---")
    scheduler_a = DPMSolverMultistepScheduler.from_config(scheduler_config, use_karras_sigmas=True)
    scheduler_a.set_timesteps(num_inference_steps)
    latents_a = latents.clone() * scheduler_a.init_noise_sigma
    for i, t in enumerate(scheduler_a.timesteps):
        latent_model_input = torch.cat([latents_a] * 2)
        latent_model_input = scheduler_a.scale_model_input(latent_model_input, t)
        with torch.no_grad():
            noise_pred = pipe_a.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
        latents_a = scheduler_a.step(noise_pred, t, latents_a).prev_sample
    
    with torch.no_grad():
        image_a = pipe_a.vae.decode(latents_a / pipe_a.vae.config.scaling_factor, return_dict=False)[0]
        image_a = pipe_a.image_processor.postprocess(image_a, output_type="pil")[0]
    
    image_a.save("model_a_output.png")
    print("Model A baseline image saved to 'model_a_output.png'")

    # Generate with Model B only
    print("\n--- Generating Baseline Image (Model B Only) ---")
    scheduler_b = DPMSolverMultistepScheduler.from_config(scheduler_config, use_karras_sigmas=True)
    scheduler_b.set_timesteps(num_inference_steps)
    latents_b = latents.clone() * scheduler_b.init_noise_sigma
    for i, t in enumerate(scheduler_b.timesteps):
        latent_model_input = torch.cat([latents_b] * 2)
        latent_model_input = scheduler_b.scale_model_input(latent_model_input, t)
        with torch.no_grad():
            noise_pred = pipe_b.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
        latents_b = scheduler_b.step(noise_pred, t, latents_b).prev_sample

    with torch.no_grad():
        image_b = pipe_b.vae.decode(latents_b / pipe_b.vae.config.scaling_factor, return_dict=False)[0]
        image_b = pipe_b.image_processor.postprocess(image_b, output_type="pil")[0]

    image_b.save("model_b_output.png")
    print("Model B baseline image saved to 'model_b_output.png'")


    # --- Step D: The RL-Guided Generation Loop ---
    print("\n--- Entering RL-Guided Generation Loop ---")
    scheduler_rl = DPMSolverMultistepScheduler.from_config(scheduler_config, use_karras_sigmas=True)
    scheduler_rl.set_timesteps(num_inference_steps)
    latents_rl = latents.clone() * scheduler_rl.init_noise_sigma # Use a clone for the RL process as well
    for i, t in enumerate(scheduler_rl.timesteps):
        chosen_arm = rl_agent.select_arm()
        chosen_model_info = models[chosen_arm]
        active_unet = chosen_model_info["unet"]
        active_vae = chosen_model_info["vae"]
        
        latent_model_input = torch.cat([latents_rl] * 2) 
        latent_model_input = scheduler_rl.scale_model_input(latent_model_input, t)
        
        with torch.no_grad():
            noise_pred = active_unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
        
        latents_rl = scheduler_rl.step(noise_pred, t, latents_rl).prev_sample

        with torch.no_grad():
            intermediate_image = active_vae.decode(latents_rl / active_vae.config.scaling_factor, return_dict=False)[0]
            intermediate_image = pipe_a.image_processor.postprocess(intermediate_image, output_type="pil")[0]

        reward = clip_reward_model.get_reward(intermediate_image, prompt)
        rl_agent.update(chosen_arm, reward)
        
        print(f"Step {i+1}/{num_inference_steps} | Model Chosen: {chosen_model_info['name']} (Arm {chosen_arm}) | Reward: {reward:.2f}")

    print("\n--- RL Generation Loop Finished ---")
    
    # --- Step E: Final Image Decoding for RL-guided result ---
    final_arm = rl_agent.select_arm() # Select one last time for the final VAE
    final_vae = models[final_arm]["vae"]
    print(f"Decoding final RL-guided image with VAE from: {models[final_arm]['name']}...")
    with torch.no_grad():
        image = final_vae.decode(latents_rl / final_vae.config.scaling_factor, return_dict=False)[0]
        final_image = pipe_a.image_processor.postprocess(image, output_type="pil")[0]

    output_filename = "rl_guided_output.png"
    final_image.save(output_filename)
    print(f"Final RL-guided image saved to '{output_filename}'")
    
    print("\n--- RL Agent Final State ---")
    for i, model_info in enumerate(models):
        print(f"Model: {model_info['name']}")
        print(f"  - Times Chosen: {rl_agent.counts[i]}")
        print(f"  - Avg. Reward: {rl_agent.values[i]:.2f}")


if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PROMPT = "a cute corgi, anime drawing, epic fantasy style"
    NUM_STEPS = 50
    
    model_b_path = "./dreamlike-diffusion-1.0"
    if not os.path.exists(model_b_path):
        print(f"Please download the '{model_b_path}' model first.")
        print("You can do this with: git lfs install && git clone https://huggingface.co/dreamlike-art/dreamlike-diffusion-1.0")
    else:
        run_rl_guided_generation(prompt=PROMPT, num_inference_steps=NUM_STEPS, device=DEVICE)
