"""
Prune checkpoint to bfloat16 format.

This script:
1. Loads a checkpoint (safetensors or pytorch format)
2. Converts all float tensors to bfloat16
3. Removes unnecessary keys (optimizer states, etc.)
4. Saves the pruned model

Usage:
    python prune_checkpoint.py --input outputs_vi/checkpoint-25000 --output outputs_vi/checkpoint-25000-bf16
"""

import os
import argparse
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def get_model_size(path: str) -> float:
    """Get total size of model files in GB."""
    total = 0
    path = Path(path)
    for f in path.glob("**/*"):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024 ** 3)


def prune_to_bf16(input_dir: str, output_dir: str, keep_fp32_keys: list = None):
    """
    Prune checkpoint to bfloat16.
    
    Args:
        input_dir: Input checkpoint directory
        output_dir: Output directory for pruned model
        keep_fp32_keys: List of key patterns to keep in fp32 (e.g., layer norms)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if keep_fp32_keys is None:
        keep_fp32_keys = []
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find model files
    safetensor_files = list(input_path.glob("*.safetensors"))
    pytorch_files = list(input_path.glob("*.bin")) + list(input_path.glob("pytorch_model*.bin"))
    
    print(f"📂 Input: {input_dir}")
    print(f"📂 Output: {output_dir}")
    print(f"📊 Original size: {get_model_size(input_dir):.2f} GB")
    print()
    
    # Process safetensors files
    if safetensor_files:
        print(f"🔧 Processing {len(safetensor_files)} safetensors file(s)...")
        
        for sf_file in safetensor_files:
            print(f"  - {sf_file.name}")
            
            # Load tensors
            tensors = {}
            with safe_open(sf_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    
                    # Check if should keep fp32
                    keep_fp32 = any(pattern in key for pattern in keep_fp32_keys)
                    
                    # Convert to bf16 if float and not in keep list
                    if tensor.dtype in [torch.float32, torch.float16] and not keep_fp32:
                        tensor = tensor.to(torch.bfloat16)
                    
                    tensors[key] = tensor
            
            # Save pruned tensors
            output_file = output_path / sf_file.name
            save_file(tensors, str(output_file))
            
            # Clear memory
            del tensors
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Process pytorch .bin files
    elif pytorch_files:
        print(f"🔧 Processing {len(pytorch_files)} pytorch file(s)...")
        
        for pt_file in pytorch_files:
            print(f"  - {pt_file.name}")
            
            # Load state dict
            state_dict = torch.load(pt_file, map_location="cpu")
            
            # Convert tensors
            new_state_dict = {}
            for key, tensor in state_dict.items():
                # Skip optimizer states
                if "optimizer" in key.lower():
                    continue
                    
                # Check if should keep fp32
                keep_fp32 = any(pattern in key for pattern in keep_fp32_keys)
                
                # Convert to bf16 if float tensor
                if isinstance(tensor, torch.Tensor):
                    if tensor.dtype in [torch.float32, torch.float16] and not keep_fp32:
                        tensor = tensor.to(torch.bfloat16)
                
                new_state_dict[key] = tensor
            
            # Save pruned state dict
            output_file = output_path / pt_file.name
            torch.save(new_state_dict, output_file)
            
            del state_dict, new_state_dict
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Copy config files
    print("\n📋 Copying config files...")
    config_files = [
        "config.json", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "vocab.json", "merges.txt",
        "generation_config.json", "added_tokens.json", "chat_template.jinja"
    ]
    
    for cfg in config_files:
        src = input_path / cfg
        if src.exists():
            shutil.copy2(src, output_path / cfg)
            print(f"  - {cfg}")
    
    # Print summary
    print()
    print("=" * 50)
    print("✅ Pruning complete!")
    print(f"📊 Original size: {get_model_size(input_dir):.2f} GB")
    print(f"📊 Pruned size: {get_model_size(output_dir):.2f} GB")
    print(f"📉 Reduction: {(1 - get_model_size(output_dir)/get_model_size(input_dir))*100:.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prune checkpoint to bfloat16")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input checkpoint directory")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output directory for pruned model")
    parser.add_argument("--keep-fp32", type=str, nargs="*", default=[],
                        help="Key patterns to keep in fp32 (e.g., 'layernorm' 'embed')")
    
    args = parser.parse_args()
    
    prune_to_bf16(args.input, args.output, args.keep_fp32)

