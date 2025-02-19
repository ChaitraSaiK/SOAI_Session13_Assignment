import os
import torch
import argparse
import sys
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
import time
from typing import Dict, List, Optional
import torch.cuda
from torch.amp import autocast
from model import SmolLM2Config, SmolLM2ForCausalLM
# from huggingface_hub import HfApi, create_repo
# from huggingface_hub import login
import json
import logging
import random

# Setup logging#
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler('logs.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set torch precision
torch.set_float32_matmul_precision('high')

# Add argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--resume_from_checkpoint', type=str, help='Path to checkpoint or True to resume from last checkpoint')
parser.add_argument('--additional_steps', type=int, default=5000, help='Number of additional training steps')
parser.add_argument('--push_to_hub', action='store_true', help='Push model to HuggingFace Hub')
parser.add_argument('--hub_model_id', type=str, help='HuggingFace Hub model ID (e.g., "username/model-name")')
parser.add_argument('--hub_token', type=str, help='HuggingFace Hub API token')
args = parser.parse_args()

# Add after logging setup
logger.info("\n" + "="*50)
logger.info(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"Training Configuration:")
logger.info(f"Resume from checkpoint: {args.resume_from_checkpoint}")
logger.info(f"Additional steps: {args.additional_steps}")
logger.info("="*50 + "\n")

# Load config and model
config = SmolLM2Config()
model = SmolLM2ForCausalLM(config)

# Use the SmolLM2 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "HuggingFaceTB/SmolLM2-360M-Instruct",  # Using the SmolLM2 tokenizer
    trust_remote_code=True
)

# Set a different pad token
tokenizer.pad_token = '[PAD]'
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# Resize model embeddings to account for new token
model.resize_token_embeddings(len(tokenizer))

# Update the dataset loading and sample text acquisition
def load_cosmopedia_dataset(batch_size, seq_length):
    print("\nLoading Cosmopedia dataset...")
    raw_dataset = load_dataset(
        "HuggingFaceTB/smollm-corpus",
        name="cosmopedia-v2",
        split="train",
        streaming=True
    )

    # Get sample text before tokenization
    print("\nGetting sample text for generation...")
    sample_text = next(iter(raw_dataset))['text'][:100]
    print("Sample text acquired for generation")

    return raw_dataset, sample_text

# Update the dataset creation section:
# Load and tokenize dataset
seq_length = 512  # Define sequence length
dataset, sample_text = load_cosmopedia_dataset(batch_size=4, seq_length=seq_length)

# Modify the TextDataset class
class TextDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.iterator = iter(dataset)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            item = next(self.iterator)
        except StopIteration:
            # Reset iterator when we reach the end
            self.iterator = iter(self.dataset)
            item = next(self.iterator)
            
        encodings = self.tokenizer(
            item["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Create input_ids and attention_mask
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        
        # Create labels (same as input_ids for causal language modeling)
        labels = input_ids.clone()
        
        # Mask out padding tokens in labels with -100
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# Update these parameters
batch_size = 4
learning_rate = 1.0e-03
max_steps = 5000  # Set fixed number of steps
checkpoint_interval = 500  # Save checkpoint every 500 steps

# Setup training components before checkpoint loading
train_dataset = TextDataset(dataset, tokenizer)  # Remove max_samples parameter
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=0,
    shuffle=False
)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Modify the checkpoint loading section
if args.resume_from_checkpoint:
    # Find the latest checkpoint
    if not os.path.exists('./checkpoints'):
        print("\nNo checkpoints directory found, starting from beginning")
        step_counter = 1
    else:
        checkpoints = [f for f in os.listdir('./checkpoints') if f.startswith('step_')]
        if checkpoints:
            latest_step = max([int(f.split('_')[1].split('.')[0]) for f in checkpoints])
            checkpoint_path = f"./checkpoints/step_{latest_step}.pt"
            logger.info(f"\nResuming Training:")
            logger.info(f"Loading checkpoint from step {latest_step}")
            logger.info(f"Will train for {args.additional_steps} more steps (until step {latest_step + args.additional_steps})")
            
            # Load checkpoint
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Handle compiled model state dict
            state_dict = checkpoint['model_state_dict']
            if all(k.startswith('_orig_mod.') for k in state_dict.keys()):
                state_dict = {k[len('_orig_mod.'):]: v for k, v in state_dict.items()}
            
            # Move model to device first
            model = model.to(device)
            model.load_state_dict(state_dict)
            
            # Load optimizer state and move to correct device
            optimizer_state = checkpoint['optimizer_state_dict']
            for state in optimizer_state['state'].values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            optimizer.load_state_dict(optimizer_state)
            
            step_counter = latest_step + 1  # Start from next step after checkpoint
            total_tokens = checkpoint['total_tokens']
            max_steps = latest_step + args.additional_steps  # Set max steps based on last checkpoint
            
            logger.info(f"Resumed from step {step_counter} with {total_tokens} tokens processed")
            logger.info(f"Will continue training until step {max_steps}")
        else:
            logger.info("\nNo checkpoints found, starting from beginning")
            step_counter = 1
            max_steps = args.additional_steps
else:
    print("\nStarting fresh training for 5000 steps")
    step_counter = 1
    max_steps = 5000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

print("-" * 40)

# Move compilation after checkpoint loading
print("\nCompiling model...")
model = torch.compile(model)
print("Model compilation completed")

# Add before training loop
start_time = time.time()

def generate_sample(model, tokenizer, device, prompt):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)
        
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=256,  # Increased max length
            min_length=32,   # Added min length
            temperature=0.9,  # Slightly increased temperature
            top_p=0.92,      # Adjusted top_p
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            num_beams=1      # Can increase for better quality
        )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    model.train()
    return generated_text

# Update the training loop initialization
print("\nTraining Progress:")
total_tokens = 0 if not 'total_tokens' in locals() else total_tokens
logging_steps = 1
start_time = time.time()

if args.resume_from_checkpoint:
    step_counter = latest_step + 1  # Start from next step after checkpoint
else:
    step_counter = 1  # Start from 1 for fresh training

# Update the sample prompts to be more diverse
sample_prompts = [
    "Write a story about a magical forest: ",
    "Explain the concept of gravity in simple terms: ",
    "Write a recipe for chocolate cake: ",
    "Describe the process of photosynthesis: ",
    "Write a poem about the ocean: ",
    "What would happen if humans could fly? ",
    "Explain how a car engine works: ",
    "Write a dialogue between two friends: ",
    "Describe your perfect vacation: ",
    "What are the benefits of exercise? "
]

random.seed(42)  # For reproducibility

# Training loop
while step_counter <= max_steps:  # Changed back to <= to include final step
    try:
        batch = next(iter(train_loader))
    except StopIteration:
        continue
        
    batch = {k: v.to(device) for k, v in batch.items()}
    
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    
    loss = outputs.loss
    loss.backward()
    
    if step_counter % batch_size == 0:
        optimizer.step()
        optimizer.zero_grad()
        
    total_tokens += batch["input_ids"].ne(tokenizer.pad_token_id).sum().item()
    
    # Print progress every step
    if step_counter % logging_steps == 0:
        elapsed = time.time() - start_time
        tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
        current_time = datetime.now().strftime('%H:%M:%S')
        log_msg = f"[{current_time}] step: {step_counter:4d} | loss: {loss.item():6.2f} | elapsed: {elapsed:6.4f}s | tokens/sec: {tokens_per_sec:8.2f} | total_tokens: {total_tokens}"
        logger.info(log_msg)
    
    # Save checkpoint and generate samples only if additional_steps >= 500
    if step_counter % checkpoint_interval == 0 and args.additional_steps >= 500:
        checkpoint_path = f"checkpoints/step_{step_counter}.pt"
        os.makedirs("checkpoints", exist_ok=True)
        
        # Save checkpoint
        torch.save({
            'step': step_counter,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
            'total_tokens': total_tokens,
        }, checkpoint_path)
        logger.info(f"\n[{datetime.now().strftime('%H:%M:%S')}] Checkpoint saved at step {step_counter}: {checkpoint_path}")
        
        # Generate and print sample text
        logger.info(f"\n=== Model Output Sample at Step {step_counter} ===")
        selected_prompts = random.sample(sample_prompts, 3)
        for prompt in selected_prompts:
            generated = generate_sample(model, tokenizer, device, prompt)
            logger.info(f"\nPrompt: {prompt}\nGenerated: {generated}\n{'-'*50}")
    
    step_counter += 1  # Increment at end of loop

# Add timestamp to final logging
logger.info("\n" + "="*50)
logger.info(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"Final step reached: {step_counter-1}")  # -1 because we increment after the last step
logger.info(f"Total training time: {time.time() - start_time:.2f} seconds")
logger.info("="*50 + "\n")

# Replace the final model saving section with:
if args.push_to_hub and args.hub_model_id and args.hub_token:
    print(f"\nPushing model to HuggingFace Hub: {args.hub_model_id}")
    
    # Login to Hugging Face
    login(token=args.hub_token)
    
    # Create the repo if it doesn't exist
    api = HfApi()
    try:
        create_repo(args.hub_model_id, private=True, exist_ok=True)
    except Exception as e:
        print(f"Repository already exists or error creating: {e}")
    
    # Save model files locally first
    output_dir = "./model_to_push"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model state dict
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    
    # Save config
    config_dict = model.config.__dict__
    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    # Push to Hub
    api.upload_folder(
        folder_path=output_dir,
        repo_id=args.hub_model_id,
        repo_type="model"
    )
    
    print(f"Model successfully pushed to {args.hub_model_id}")
else:
    # Save locally
    os.makedirs("./checkpoints/final", exist_ok=True)
    
    # Save model state dict
    torch.save(model.state_dict(), "./checkpoints/final/pytorch_model.bin")
    
    # Save config
    config_dict = model.config.__dict__
    with open("./checkpoints/final/config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Save tokenizer
    tokenizer.save_pretrained("./checkpoints/final")
    logger.info("\nModel saved locally in ./checkpoints/final")
