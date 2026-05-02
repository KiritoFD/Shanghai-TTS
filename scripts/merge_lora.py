import os
import shutil
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_lora_and_save(base_model_path: str, lora_path: str, output_path: str):
    print(f"Loading base model from {base_model_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cpu"  # Keep it on CPU to save VRAM during merge
    )
    
    print(f"Loading LoRA adapter from {lora_path}...")
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    print("Merging weights...")
    model = model.merge_and_unload()
    
    print(f"Saving merged model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)
    
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)
    
    print("Merge complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="model/Qwen3.5-2B")
    parser.add_argument("--lora", type=str, default="app/model_lora/checkpoint-130")
    parser.add_argument("--output", type=str, default="model/Qwen3.5-2B-split-merged")
    args = parser.parse_args()
    
    merge_lora_and_save(args.base, args.lora, args.output)
