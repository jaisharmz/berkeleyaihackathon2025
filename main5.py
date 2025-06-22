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
    print("--- Starting RL-Guided Generation ---")
    
    # --- Step A: Load all necessary models ---
    print("Loading models...")
    
    # Model A: A general-purpose, high-quality model
    model_id_a = "runwayml/stable-diffusion-v1-5"
    # Model B: A model fine-tuned for a specific style (e.g., anime/fantasy)
    # The path now correctly points to the directory created by `git clone`.
    model_id_b = "./dreamlike-diffusion-1.0"

    try:
        # We load the full pipelines to get their components (VAE, UNet, etc.)
        pipe_a = AutoPipelineForText2Image.from_pretrained(model_id_a, torch_dtype=torch.float16).to(device)
        pipe_b = AutoPipelineForText2Image.from_pretrained(model_id_b, torch_dtype=torch.float16).to(device)
        print("Specialized diffusion models loaded.")
    except Exception as e:
        print(f"Fatal error loading diffusion models: {e}. Aborting.")
        print("This may be due to a missing model name, network issues, or memory constraints.")
        return

    # For simplicity and latent space compatibility, we'll use the VAE,
    # tokenizer, and text_encoder from one of the models (Model A) as the "base".
    # The key is swapping out the UNet, which is the core denoising network.
    tokenizer = pipe_a.tokenizer
    text_encoder = pipe_a.text_encoder
    scheduler = DPMSolverMultistepScheduler.from_config(pipe_a.scheduler.config)

    models = [
        {"name": "Generalist (SD 1.5)", "unet": pipe_a.unet, "vae": pipe_a.vae},
        {"name": "Stylized (Dreamlike)", "unet": pipe_b.unet, "vae": pipe_b.vae}
    ]

    # Initialize the reward model and the RL agent
    clip_reward_model = CLIPReward(device=device)
    rl_agent = UCBAgent(num_arms=len(models))
    
    # --- Step B: Prepare for the generation loop ---
    print("\n--- Preparing for Generation Loop ---")
    
    # Encode the prompt
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        prompt_embeds = text_encoder(text_input.input_ids.to(device))[0]
    
    negative_prompt = "" # Or a more descriptive negative prompt
    negative_prompt_input = tokenizer(negative_prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        negative_prompt_embeds = text_encoder(negative_prompt_input.input_ids.to(device))[0]

    text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds])


    # Set the number of steps for the scheduler
    scheduler.set_timesteps(num_inference_steps)
    
    # Create initial random noise (the starting point of diffusion)
    latents = torch.randn(
        (1, pipe_a.unet.config.in_channels, pipe_a.unet.config.sample_size, pipe_a.unet.config.sample_size),
        device=device,
        dtype=torch.float16
    )
    latents = latents * scheduler.init_noise_sigma

    # --- Step C: The Generation Loop ---
    # This is the core of your idea. At each step, we ask the RL agent which
    # model to use, apply it, and then update the agent with a reward.
    print("\n--- Entering Generation Loop ---")
    for i, t in enumerate(scheduler.timesteps):
        # 1. RL Agent selects an action (a model to use)
        chosen_arm = rl_agent.select_arm()
        chosen_model_info = models[chosen_arm]
        active_unet = chosen_model_info["unet"]
        active_vae = chosen_model_info["vae"]
        
        # 2. Perform one step of diffusion with the chosen model's UNet
        latent_model_input = torch.cat([latents] * 2) # for classifier-free guidance
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        
        with torch.no_grad():
            noise_pred = active_unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond) # guidance
        
        latents = scheduler.step(noise_pred, t, latents).prev_sample

        # 3. Get intermediate reward (this is the crucial feedback loop)
        # We decode the current latents to an image to score it.
        # NOTE: This is computationally expensive! In a real system, you might
        # only do this every N steps to find a balance.
        with torch.no_grad():
            # Use the VAE corresponding to the chosen UNet
            intermediate_image = active_vae.decode(latents / active_vae.config.scaling_factor, return_dict=False)[0]
            intermediate_image = (intermediate_image / 2 + 0.5).clamp(0, 1)
            intermediate_image = intermediate_image.cpu().permute(0, 2, 3, 1).numpy()
            # FIX: Call numpy_to_pil from an instantiated pipeline object, not the class.
            intermediate_image = pipe_a.numpy_to_pil(intermediate_image)[0]

        reward = clip_reward_model.get_reward(intermediate_image, prompt)
        
        # 4. Update the RL agent with the reward
        rl_agent.update(chosen_arm, reward)
        
        print(f"Step {i+1}/{num_inference_steps} | Model Chosen: {chosen_model_info['name']} (Arm {chosen_arm}) | Reward: {reward:.2f}")

    print("\n--- Generation Loop Finished ---")
    
    # --- Step D: Final Image Decoding ---
    # After all steps, decode the final latents into an image.
    # We'll use the VAE of the last chosen model for the final decode.
    final_vae = models[rl_agent.select_arm()]["vae"]
    print(f"Decoding final image with VAE from: {models[rl_agent.select_arm()]['name']}...")
    with torch.no_grad():
        image = final_vae.decode(latents / final_vae.config.scaling_factor, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    # FIX: Call numpy_to_pil from an instantiated pipeline object, not the class.
    final_image = pipe_a.numpy_to_pil(image)[0]

    # Save the final image
    output_filename = "rl_guided_output.png"
    final_image.save(output_filename)
    print(f"Final image saved to '{output_filename}'")
    
    # Print the final learned values of the agent
    print("\n--- RL Agent Final State ---")
    for i, model_info in enumerate(models):
        print(f"Model: {model_info['name']}")
        print(f"  - Times Chosen: {rl_agent.counts[i]}")
        print(f"  - Avg. Reward: {rl_agent.values[i]:.2f}")


if __name__ == '__main__':
    # --- Configuration ---
    # Ensure you have a CUDA-enabled GPU. If not, change to "cpu".
    # Note: Running this on CPU will be extremely slow.
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # The prompt that combines elements from both specialized models.
    # For example, "a majestic lion in the style of an epic fantasy painting".
    # Model A (SD 1.5) is good at "majestic lion".
    # Model B (Dreamlike) is good at "epic fantasy painting".
    PROMPT = "a cute corgi, anime drawing, epic fantasy style"
    
    # Number of steps in the diffusion process.
    NUM_STEPS = 50

    # The user clones the repo, which creates a directory named 'dreamlike-diffusion-1.0'
    # This check now looks for the correct directory name.
    model_b_path = "./dreamlike-diffusion-1.0"
    if not os.path.exists(model_b_path):
        print(f"Please download the '{model_b_path}' model first.")
        print("You can do this with: git lfs install && git clone https://huggingface.co/dreamlike-art/dreamlike-diffusion-1.0")
    else:
        run_rl_guided_generation(prompt=PROMPT, num_inference_steps=NUM_STEPS, device=DEVICE)
