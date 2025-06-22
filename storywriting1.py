import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import math
import os
from tqdm.auto import tqdm
import re

# --- Configuration ---
# --------------------------------------------------------------------------------------
# User Prompt and its breakdown for the reward function
USER_PROMPT = "A futuristic sci-fi noir mystery with witty back-and-forth dialogue."
# These phrases will be embedded to guide the different models
PROMPT_ASPECTS = {
    "Plot": "A futuristic sci-fi noir mystery with suspense and action.",
    "Scene": "A detailed and atmospheric description of a futuristic city.",
    "Dialogue": "Witty, sharp, back-and-forth dialogue between characters."
}

# Reinforcement Learning Agent (UCB) Settings
UCB_EXPLORATION_CONSTANT = 1.5  # Balances exploration and exploitation

# Story Generation Settings
NUM_SENTENCES = 15  # The total number of sentences in the final story
SEED = 42

# Output settings
OUTPUT_DIR = "story_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# --------------------------------------------------------------------------------------

class UCB1_Bandit:
    """The UCB1 Multi-Armed Bandit agent - our 'Head Writer'."""
    def __init__(self, model_names, exploration_constant):
        self.models = {name: {'pulls': 0, 'value': 0.0} for name in model_names}
        self.exploration_constant = exploration_constant
        self.total_pulls = 0
        self.names = model_names

    def select_action(self):
        self.total_pulls += 1
        for name in self.names:
            if self.models[name]['pulls'] == 0:
                print(f"  >> Agent Decision: Initializing -> Chose {name}")
                self.models[name]['pulls'] += 1
                return name
        
        ucb_scores = {}
        for name in self.names:
            model_stats = self.models[name]
            average_reward = model_stats['value']
            exploration_bonus = self.exploration_constant * math.sqrt(
                math.log(self.total_pulls) / model_stats['pulls']
            )
            ucb_scores[name] = average_reward + exploration_bonus
        
        action = max(ucb_scores, key=ucb_scores.get)
        self.models[action]['pulls'] += 1
        print(f"  >> Agent Decision: UCB -> Chose {action}")
        return action

    def update_value(self, model_name, reward):
        model_stats = self.models[model_name]
        model_stats['value'] = ((model_stats['value'] * (model_stats['pulls'] - 1)) + reward) / model_stats['pulls']
        print(f"  >> Agent Update: {model_name} value is now {model_stats['value']:.3f}")

class AI_Writers_Room:
    """Manages the models, generation process, and reward calculation."""
    def __init__(self, prompt_aspects):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # --- Load Models ---
        print("Loading models from Hugging Face Hub...")
        self.reward_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        self.tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-small')
        self.model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-small').to(self.device)
        print("Models loaded successfully.")

        self.prompt_embeddings = {}
        for aspect, text in prompt_aspects.items():
            self.prompt_embeddings[aspect] = self.reward_model.encode(text, convert_to_tensor=True)
            
        self.world_state = {
            "characters": ["Detective Kaito", "a mysterious informant named Anya"],
            "setting": "a rain-slicked street in Neo-Kyoto under a perpetual twilight sky",
            "last_speaker": None,
            "initial_story": "Detective Kaito pulled the collar of his trench coat tighter, the neon signs of Neo-Kyoto reflecting in the puddles at his feet."
        }

    def _generate_with_t5(self, input_text):
        """Helper for T5-based models with improved generation parameters."""
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True).to(self.device)
        outputs = self.model.generate(
            inputs.input_ids, 
            max_new_tokens=60, 
            num_beams=5, 
            no_repeat_ngram_size=2, 
            early_stopping=True,
            temperature=0.9
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    # --- The "Specialized Writers" ---
    def plot_driver(self, story_context):
        # FINAL FIX: Use a simple, unambiguous instruction prefix.
        prompt = f"Write a single sentence to continue the following story: {story_context}"
        return self._generate_with_t5(prompt)

    def scene_setter(self, story_context):
        # FINAL FIX: Use a simple, unambiguous instruction prefix.
        prompt = f"Write a single sentence to describe the setting of the following story in more detail: {story_context}"
        return self._generate_with_t5(prompt)

    def dialogue_writer(self, story_context):
        # FINAL FIX: Use a simple, unambiguous instruction prefix for dialogue.
        if self.world_state['last_speaker'] == "Detective Kaito":
            next_speaker = self.world_state['characters'][1] # Anya
        else:
            next_speaker = "Detective Kaito"
        
        last_line = re.split(r'(?<=[.!?"])\s+', story_context.strip())[-1]
        prompt = (f"The last line of dialogue was: {last_line}. "
                  f"Write the next line of dialogue spoken by {next_speaker}.")
        response = self._generate_with_t5(prompt)
        self.world_state['last_speaker'] = next_speaker
        return f'"{response}"'

    def calculate_reward(self, generated_sentence, model_name):
        """Calculates reward based on semantic similarity to the prompt aspect."""
        if not generated_sentence or not generated_sentence.strip():
            return 0.0
        sentence_embedding = self.reward_model.encode(generated_sentence, convert_to_tensor=True)
        target_embedding = self.prompt_embeddings[model_name]
        
        cosine_score = util.pytorch_cos_sim(sentence_embedding, target_embedding)
        return cosine_score.item()

    def generate_story(self, agent=None, force_model=None, output_filename="story.txt"):
        """Main story generation loop."""
        story = self.world_state["initial_story"]
        models = {
            "Plot": self.plot_driver,
            "Scene": self.scene_setter,
            "Dialogue": self.dialogue_writer
        }
        
        desc = f"Generating with {force_model or 'RL Agent'}"
        for _ in tqdm(range(NUM_SENTENCES), desc=desc):
            if force_model:
                chosen_model_name = force_model
            elif agent:
                chosen_model_name = agent.select_action()
            else:
                raise ValueError("Must provide either agent or force_model")
            
            action_function = models[chosen_model_name]
            new_sentence = action_function(story)
            
            print(f"\n{chosen_model_name} wrote: {new_sentence}")
            
            if new_sentence and new_sentence.strip() and len(new_sentence.split()) > 2:
                story += " " + new_sentence
                if agent and not force_model:
                    reward = self.calculate_reward(new_sentence, chosen_model_name)
                    print(f"  >> Reward: {reward:.3f}")
                    agent.update_value(chosen_model_name, reward)
            else:
                print("  >> Skipping empty/short sentence.")
                if agent and not force_model:
                    agent.update_value(chosen_model_name, -0.5)
                
        with open(os.path.join(OUTPUT_DIR, output_filename), 'w') as f:
            f.write(story.strip())
        print(f"\n✅ Story saved to {os.path.join(OUTPUT_DIR, output_filename)}")
        return story

if __name__ == "__main__":
    torch.manual_seed(SEED)
    
    writers_room = AI_Writers_Room(PROMPT_ASPECTS)
    
    # --- 1. Run with RL Agent ---
    print("\n" + "="*50 + "\nRUN 1: Generating with RL Agent (UCB1)\n" + "="*50)
    bandit_agent = UCB1_Bandit(["Plot", "Scene", "Dialogue"], UCB_EXPLORATION_CONSTANT)
    writers_room.generate_story(agent=bandit_agent, output_filename="story_rl_guided.txt")
    print("\n--- Final RL Agent Stats ---")
    for name, stats in bandit_agent.models.items():
        print(f"  {name}: Pulls={stats['pulls']}, Final Value={stats['value']:.3f}")

    # --- 2. Run with Plot Driver only ---
    print("\n" + "="*50 + "\nRUN 2: Generating with Plot Driver only\n" + "="*50)
    writers_room.generate_story(force_model="Plot", output_filename="story_plot_only.txt")

    # --- 3. Run with Dialogue Writer only ---
    print("\n" + "="*50 + "\nRUN 3: Generating with Dialogue Writer only\n" + "="*50)
    writers_room.generate_story(force_model="Dialogue", output_filename="story_dialogue_only.txt")

