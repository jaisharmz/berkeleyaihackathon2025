import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from diffusers import DDIMScheduler, StableDiffusionPipeline
import numpy as np
import os
import math
from tqdm.auto import tqdm

# --- Configuration ---
# --------------------------------------------------------------------------------------
# Model Identifiers from Hugging Face Hub
MODEL_A_ID = "runwayml/stable-diffusion-v1-5"
MODEL_B_ID = "prompthero/openjourney" # A Midjourney-style fine-tune of SD 1.5

# Reinforcement Learning Agent (UCB) Settings
# UCB is used instead of Epsilon-Greedy to encourage better exploration.
# A higher constant encourages more exploration to achieve a more balanced pull distribution.
UCB_EXPLORATION_CONSTANT = 5.0  # Constant 'c' for the UCB1 formula. Higher values encourage more exploration.
PROMPT = "A photorealistic portrait of an astronaut, in the style of a detailed oil painting."
# PROMPT = "A blueprint schematic of a fantasy dragon."
# PROMPT = "A photograph of a New York City street, reimagined by Van Gogh."

# Diffusion Process Settings
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
HEIGHT = 512
WIDTH = 512
SEED = 42

# Output settings
OUTPUT_DIR = "rl_diffusion_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# --------------------------------------------------------------------------------------


class UCB1_Bandit:
    """
    An Upper Confidence Bound (UCB1) Multi-Armed Bandit agent.
    This provides a more robust exploration strategy than Epsilon-Greedy.
    """
    def __init__(self, model_names, exploration_constant):
        self.models = {name: {'pulls': 0, 'value': 0.0} for name in model_names}
        self.exploration_constant = exploration_constant
        self.total_pulls = 0
        self.names = model_names

    def select_action(self):
        """
        Chooses a model (arm) to pull using the UCB1 formula.
        The formula balances exploitation (highest average reward) and
        exploration (uncertainty about under-sampled arms).
        """
        self.total_pulls += 1

        # First, play each arm once to initialize
        for name in self.names:
            if self.models[name]['pulls'] == 0:
                print(f"  >> Agent decision: INITIALIZING -> Chose {name}")
                self.models[name]['pulls'] += 1
                return name
        
        # --- UCB1 Calculation ---
        ucb_scores = {}
        for name in self.names:
            model_stats = self.models[name]
            average_reward = model_stats['value']
            
            # The exploration term encourages trying less-pulled arms
            exploration_bonus = self.exploration_constant * math.sqrt(
                math.log(self.total_pulls) / model_stats['pulls']
            )
            
            ucb_scores[name] = average_reward + exploration_bonus
            print(f"  >> UCB Score for {name}: {average_reward:.3f} (avg) + {exploration_bonus:.3f} (bonus) = {ucb_scores[name]:.3f}")

        # Choose the arm with the highest UCB score
        action = max(ucb_scores, key=ucb_scores.get)
        self.models[action]['pulls'] += 1
        print(f"  >> Agent decision: UCB -> Chose {action}")
        return action

    def update_value(self, model_name, reward):
        """
        Updates the value of a model based on the received reward.
        """
        model_stats = self.models[model_name]
        # Use an incremental average formula to update the value
        # This is equivalent to sum_of_rewards / num_pulls
        model_stats['value'] = ((model_stats['value'] * (model_stats['pulls'] - 1)) + reward) / model_stats['pulls']
        print(f"  >> Agent updated: {model_name} now has value {model_stats['value']:.4f} after {model_stats['pulls']} pulls.")


class RLDrivenDiffusion:
    """
    Manages the diffusion process guided by the RL agent.
    """
    def __init__(self, prompt, model_a_id, model_b_id, seed):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.prompt = prompt
        self.seed = seed

        # --- Load Models ---
        print("Loading models... This may take a while.")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.pipe_a = StableDiffusionPipeline.from_pretrained(model_a_id, torch_dtype=torch.float16).to(self.device)
        self.pipe_b = StableDiffusionPipeline.from_pretrained(model_b_id, torch_dtype=torch.float16).to(self.device)
        self.scheduler = DDIMScheduler.from_config(self.pipe_a.scheduler.config)

        self.tokenizer = self.pipe_a.tokenizer
        self.text_encoder = self.pipe_a.text_encoder
        self.unet_a = self.pipe_a.unet
        self.unet_b = self.pipe_b.unet
        self.vae = self.pipe_a.vae
        
        print("Models loaded successfully.")

    def get_clip_score(self, image):
        with torch.no_grad():
            inputs = self.clip_processor(text=[self.prompt], images=[image], return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.clip_model(**inputs)
            return outputs.logits_per_image.item()

    def _get_text_embeddings(self, prompt):
        text_input = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return text_embeddings

    def generate(self, agent=None, force_model=None, output_filename="final_image.png"):
        """
        The main generation loop.
        - If 'agent' is provided, runs in RL-guided mode.
        - If 'force_model' is 'Model_A' or 'Model_B', uses only that model.
        """
        # --- 1. Setup ---
        uncond_embeddings = self._get_text_embeddings("")
        text_embeddings = self._get_text_embeddings(self.prompt)
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        self.scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=self.device)
        
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        latents = torch.randn((1, self.unet_a.config.in_channels, HEIGHT // 8, WIDTH // 8), generator=generator, device=self.device, dtype=torch.float16)
        latents = latents * self.scheduler.init_noise_sigma
        
        last_score = 0.0
        
        # --- 2. The Diffusion Loop ---
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc=f"Generating with {force_model or 'RL Agent'}")):
            
            if force_model:
                chosen_model_name = force_model
            elif agent:
                chosen_model_name = agent.select_action()
            else:
                raise ValueError("Must provide either an agent or a force_model setting.")

            unet_to_use = self.unet_a if chosen_model_name == "Model_A" else self.unet_b
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            with torch.no_grad():
                noise_pred = unet_to_use(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
            # --- 3. Reward Calculation and Agent Update (periodically) ---
            if (i + 1) % 10 == 0 or i == NUM_INFERENCE_STEPS - 1:
                if agent and not force_model:
                    with torch.no_grad():
                        image = self.pipe_a.decode_latents(latents.to(torch.float16))
                    pil_image = self.pipe_a.numpy_to_pil(image)[0]
                    score = self.get_clip_score(pil_image)
                    reward = score - last_score
                    last_score = score
                    print(f"\n  >> Step {i+1}: Current CLIP Score: {score:.4f} (Reward: {reward:.4f})")
                    agent.update_value(chosen_model_name, reward)
                
        # --- 4. Final Image Generation ---
        print(f"\n--- Diffusion process finished. Decoding final image for {output_filename}. ---")
        with torch.no_grad():
            image = self.pipe_a.decode_latents(latents.to(torch.float16))
        final_image = self.pipe_a.numpy_to_pil(image)[0]
        final_image.save(os.path.join(OUTPUT_DIR, output_filename))
        return final_image

# --- Main execution ---
if __name__ == "__main__":
    rl_diffuser = RLDrivenDiffusion(prompt=PROMPT, model_a_id=MODEL_A_ID, model_b_id=MODEL_B_ID, seed=SEED)

    # --- 1. Run with RL Agent ---
    print("\n" + "="*50)
    print("RUN 1: Generating with RL Agent (UCB1)")
    print("="*50)
    bandit_agent = UCB1_Bandit(model_names=["Model_A", "Model_B"], exploration_constant=UCB_EXPLORATION_CONSTANT)
    rl_image = rl_diffuser.generate(agent=bandit_agent, output_filename="final_image_RL_guided.png")
    print("\n--- Final RL Agent Stats ---")
    for name, stats in bandit_agent.models.items():
        print(f"{name}: Pulls={stats['pulls']}, Final Value={stats['value']:.4f}")

    # --- 2. Run with Model A only ---
    print("\n" + "="*50)
    print("RUN 2: Generating with Model A only (Stable Diffusion 1.5)")
    print("="*50)
    model_a_image = rl_diffuser.generate(force_model="Model_A", output_filename="final_image_Model_A_only.png")

    # --- 3. Run with Model B only ---
    print("\n" + "="*50)
    print("RUN 3: Generating with Model B only (OpenJourney)")
    print("="*50)
    model_b_image = rl_diffuser.generate(force_model="Model_B", output_filename="final_image_Model_B_only.png")

    print(f"\n✅ All runs complete! Check the '{OUTPUT_DIR}' directory for your images.")

    # Display images if in an interactive environment (e.g., Jupyter)
    try:
        from IPython.display import display
        print("\n--- Comparison of Final Images ---")
        print("\n1. RL-Guided Result:")
        display(rl_image)
        print("\n2. Model A (Stable Diffusion 1.5) Only:")
        display(model_a_image)
        print("\n3. Model B (OpenJourney) Only:")
        display(model_b_image)
    except ImportError:
        print("\nTo see images, run this script in a Jupyter Notebook or view the output files.")
