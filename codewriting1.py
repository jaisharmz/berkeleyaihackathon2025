import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import math
import os
import ast
from tqdm.auto import tqdm
import re

# --- Configuration ---
# --------------------------------------------------------------------------------------
USER_PROMPT = "Write a Python function that takes a list of strings and returns a new list with only the strings that are palindromes."

# Reinforcement Learning Agent (UCB) Settings
UCB_EXPLORATION_CONSTANT = 2.5

# Code Generation Settings
NUM_BLOCKS_TO_GENERATE = 15
SEED = 42

# Model and Output settings
# FIX: Upgraded to a larger, more capable model for better results.
CODE_MODEL_ID = "Salesforce/codet5-large"
REWARD_MODEL_ID = "all-MiniLM-L6-v2"
OUTPUT_DIR = "code_gen_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# --------------------------------------------------------------------------------------

class UCB1_Bandit:
    """The UCB1 Multi-Armed Bandit agent - our 'Tech Lead'."""
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
                math.log(self.total_pulls) / self.models[name]['pulls']
            )
            ucb_scores[name] = average_reward + exploration_bonus
        
        action = max(ucb_scores, key=ucb_scores.get)
        self.models[action]['pulls'] += 1
        print(f"  >> Agent Decision: UCB -> Chose {action}")
        return action

    def update_value(self, model_name, reward):
        model_stats = self.models[model_name]
        model_stats['value'] = ((model_stats['value'] * (model_stats['pulls'] - 1)) + reward) / model_stats['pulls']
        print(f"  >> Agent Update: {model_name} value is now {model_stats['value']:.3f} after a reward of {reward:.2f}")

class AI_Pair_Programmer:
    """Manages the models, code generation process, and reward calculation."""
    def __init__(self, user_prompt):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.user_prompt = user_prompt

        print("Loading models...")
        self.reward_model = SentenceTransformer(REWARD_MODEL_ID, device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(CODE_MODEL_ID)
        self.model = T5ForConditionalGeneration.from_pretrained(CODE_MODEL_ID).to(self.device)
        print("Models loaded.")

        self.prompt_embedding = self.reward_model.encode(user_prompt, convert_to_tensor=True)

    def _generate_code(self, prompt, max_tokens=120):
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(self.device)
        outputs = self.model.generate(
            inputs.input_ids, max_new_tokens=max_tokens, num_beams=5, early_stopping=True, no_repeat_ngram_size=2
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    # --- The "Specialized Developers" ---
    def logic_scripter(self, function_signature, docstring, function_body):
        current_code = f"def {function_signature}:\n{docstring}"
        prompt = (f"Based on the following docstring, complete the Python code for the function body.\n\n"
                  f"Function:\n{current_code}\n\n    # Write the code here")
        generated_code = self._generate_code(prompt)
        body_match = re.search(r'def[^{]*?:\s*?"""[^"]*?"""\s*(.*)', generated_code, re.DOTALL)
        return body_match.group(1).strip() if body_match else generated_code


    def docstring_documenter(self, function_signature):
        prompt = f"Write a standard Python docstring for the function `def {function_signature}:` that performs the following task: '{self.user_prompt}'"
        docstring = self._generate_code(prompt, max_tokens=80)
        return f'    """{docstring}"""'

    def example_generator(self, full_function_code):
        prompt = f"Write one or more Python assert statements to test that the following function works correctly:\n\n{full_function_code}"
        return self._generate_code(prompt, max_tokens=60)

    # --- The "Code Reviewer" ---
    def calculate_reward(self, new_chunk, full_code, specialist_name):
        if not new_chunk or not new_chunk.strip(): return -1.0 

        try:
            ast.parse(full_code)
            syntax_reward = 0.5
        except (SyntaxError, IndentationError):
            return -2.0 

        specialist_reward = 0
        if specialist_name == "Docstring":
            clean_chunk = new_chunk.replace('"""', '').strip()
            doc_embedding = self.reward_model.encode(clean_chunk, convert_to_tensor=True)
            specialist_reward = util.pytorch_cos_sim(doc_embedding, self.prompt_embedding).item()
        elif specialist_name == "Example":
            try:
                exec(full_code)
                specialist_reward = 2.0 
            except AssertionError:
                specialist_reward = -2.0
            except Exception:
                specialist_reward = -1.0 
        
        return syntax_reward + specialist_reward

    def generate_function(self, agent=None, fixed_workflow=None, output_filename="code.py"):
        code_parts = {
            "signature": "find_palindromes(strings: list) -> list:",
            "docstring": "",
            "body": "",
            "examples": ""
        }
        
        def assemble_code(parts):
            docstring_str = f"{parts['docstring']}\n" if parts['docstring'] else ""
            body_str = f"    {parts['body'].replace(chr(10), chr(10) + '    ')}\n" if parts['body'] else ""
            examples_str = f"{parts['examples']}\n" if parts['examples'] else ""
            return f"def {parts['signature']}:\n{docstring_str}{body_str}\n{examples_str}"

        specialists = {
            "Logic": self.logic_scripter,
            "Docstring": self.docstring_documenter,
            "Example": self.example_generator,
        }
        
        workflow_type = "RL Agent" if agent else "Fixed Workflow" if fixed_workflow else "Logic Only"
        for i in tqdm(range(NUM_BLOCKS_TO_GENERATE), desc=f"Generating with {workflow_type}"):
            if agent:
                specialist_name = agent.select_action()
            elif fixed_workflow:
                specialist_name = fixed_workflow[i % len(fixed_workflow)]
            else: # This is now the "Logic Only" case
                specialist_name = "Logic"

            new_code_block = ""
            if specialist_name == "Docstring":
                if not code_parts["docstring"]: 
                    new_code_block = specialists[specialist_name](code_parts["signature"])
                    code_parts["docstring"] = new_code_block
            elif specialist_name == "Logic":
                new_code_block = specialists[specialist_name](code_parts["signature"], code_parts["docstring"], code_parts["body"])
                code_parts["body"] = new_code_block
            elif specialist_name == "Example":
                full_code_for_test = assemble_code(code_parts)
                new_code_block = specialists[specialist_name](full_code_for_test)
                code_parts["examples"] = new_code_block
            
            print(f"\n{specialist_name} wrote:\n---\n{new_code_block}\n---")
            
            if agent:
                full_code_snapshot = assemble_code(code_parts)
                reward = self.calculate_reward(new_code_block, full_code_snapshot, specialist_name)
                agent.update_value(specialist_name, reward)

        final_code = assemble_code(code_parts)
        with open(os.path.join(OUTPUT_DIR, output_filename), 'w') as f:
            f.write(final_code)
        print(f"\n✅ Code saved to {os.path.join(OUTPUT_DIR, output_filename)}")

if __name__ == "__main__":
    torch.manual_seed(SEED)
    
    pair_programmer = AI_Pair_Programmer(USER_PROMPT)
    
    # --- Run 1: RL-Guided "Tech Lead" ---
    print("\n" + "="*50 + "\nRUN 1: Generating with RL Agent (UCB1)\n" + "="*50)
    bandit_agent = UCB1_Bandit(["Logic", "Docstring", "Example"], UCB_EXPLORATION_CONSTANT)
    pair_programmer.generate_function(agent=bandit_agent, output_filename="code_rl_guided.py")
    print("\n--- Final RL Agent Stats ---")
    for name, stats in bandit_agent.models.items():
        print(f"  {name}: Pulls={stats['pulls']}, Final Value={stats['value']:.3f}")

    # --- Run 2: "Cowboy Coder" (Logic Only) ---
    print("\n" + "="*50 + "\nRUN 2: Generating with Logic Scripter only\n" + "="*50)
    pair_programmer.generate_function(output_filename="code_logic_only.py")

    # --- Run 3: The "Intern" (Fixed Workflow) ---
    print("\n" + "="*50 + "\nRUN 3: Generating with a Fixed Workflow\n" + "="*50)
    intern_workflow = ["Docstring", "Logic", "Example"]
    pair_programmer.generate_function(fixed_workflow=intern_workflow, output_filename="code_fixed_workflow.py")
