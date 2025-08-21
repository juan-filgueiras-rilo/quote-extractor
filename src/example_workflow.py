#!/usr/bin/env python
"""
Example workflow showing the complete training and inference pipeline
for HotpotQA using the Quote-based approach with Unsloth.

This demonstrates how the same QuoteHotpotQAProcessor and templates
are reused across training and inference.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and print its status"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Error: Command failed with return code {result.returncode}")
        sys.exit(1)
    return result


def main():
    # Configuration
    TRAIN_DATA = "data/hotpot_train.json"
    DEV_DATA = "data/hotpot_dev.json"
    MODEL_DIR = "models/quote_hotpot_llama3_test"
    RESULTS_DIR = "results"
    
    # Training parameters
    NUM_TRAIN_SAMPLES = 1000  # Use subset for quick test
    NUM_EPOCHS = 1
    PROMPT_STYLE = "detailed"  # Same style for training and inference
    
    # Create directories
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║  HotpotQA Quote-based Training & Inference Pipeline     ║
    ║  Using Unsloth with QuoteHotpotQAProcessor              ║
    ╚══════════════════════════════════════════════════════════╝
    
    This example demonstrates:
    1. Training with Unsloth using the quote-based template
    2. Inference using the trained model with the same processor
    3. Evaluation of results
    """)
    
    # Step 1: Training with Unsloth
    train_cmd = [
        "python", "-m", "src.cli.training",
        "--input", TRAIN_DATA,
        "--output-dir", MODEL_DIR,
        "--num-samples", str(NUM_TRAIN_SAMPLES),
        "--model", "unsloth/llama-3-8b-bnb-4bit",
        "--prompt-style", PROMPT_STYLE,
        "--num-epochs", str(NUM_EPOCHS),
        "--batch-size", "2",
        "--gradient-accumulation-steps", "4",
        "--learning-rate", "2e-4",
        "--lora-r", "16",
        "--lora-alpha", "16",
        "--logging-steps", "10",
        "--save-method", "lora"  # Save as LoRA adapter
    ]
    
    run_command(
        train_cmd,
        f"Training model on {NUM_TRAIN_SAMPLES} samples"
    )
    
    print("\n✓ Training completed successfully!")
    
    # Step 2: Inference with the trained model
    inference_cmd = [
        "python", "-m", "src.cli.inference",
        "--input", DEV_DATA,
        "--output", f"{RESULTS_DIR}/unsloth_quote_results.json",
        "--provider", "unsloth",  # Use Unsloth provider
        "--model", MODEL_DIR,  # Path to trained model
        "--processor", "QuoteHotpotQAProcessor",  # Use Quote processor
        "--prompt-style", PROMPT_STYLE,  # Same style as training
        "--num-samples", "50",  # Test on subset
        "--evaluate",  # Enable evaluation
        "--max-new-tokens", "512",
        "--temperature", "0.1"
    ]
    
    run_command(
        inference_cmd,
        "Running inference with trained model"
    )
    
    print("\n✓ Inference completed successfully!")
    
    # Step 3: Compare with base model (optional)
    print(f"\n{'='*60}")
    print("OPTIONAL: Compare with base model performance")
    print(f"{'='*60}")
    
    base_inference_cmd = [
        "python", "-m", "src.cli.inference",
        "--input", DEV_DATA,
        "--output", f"{RESULTS_DIR}/base_model_results.json",
        "--provider", "ollama-openai",
        "--model", "llama3.1:8b",
        "--processor", "QuoteHotpotQAProcessor",
        "--prompt-style", PROMPT_STYLE,
        "--num-samples", "50",
        "--evaluate"
    ]
    
    print("To compare with base model, run:")
    print(" ".join(base_inference_cmd))
    
    # Step 4: Analyze results
    print(f"\n{'='*60}")
    print("RESULTS ANALYSIS")
    print(f"{'='*60}")
    
    results_file = Path(f"{RESULTS_DIR}/unsloth_quote_results.json")
    if results_file.exists():
        import json
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        successful = [r for r in results if 'error' not in r]
        if successful and 'metrics' in successful[0]:
            exact_matches = sum(1 for r in successful 
                              if r.get('metrics', {}).get('exact_match', False))
            avg_answer_f1 = sum(r.get('metrics', {}).get('answer_f1', 0) 
                               for r in successful) / len(successful)
            avg_sf_f1 = sum(r.get('metrics', {}).get('supporting_facts_f1', 0) 
                           for r in successful) / len(successful)
            
            print(f"Evaluated {len(successful)} examples:")
            print(f"  • Exact Match: {exact_matches}/{len(successful)} ({100*exact_matches/len(successful):.1f}%)")
            print(f"  • Answer F1: {avg_answer_f1:.3f}")
            print(f"  • Supporting Facts F1: {avg_sf_f1:.3f}")
    
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE!")
    print(f"{'='*60}")
    print(f"Trained model saved to: {MODEL_DIR}")
    print(f"Results saved to: {RESULTS_DIR}")
    print("\nKey benefits of this approach:")
    print("  1. Same QuoteHotpotQAProcessor used for training and inference")
    print("  2. Consistent prompt templates across the pipeline")
    print("  3. Structured output with exact quote matching")
    print("  4. Easy to switch between providers (Unsloth, Ollama, OpenAI, etc.)")


if __name__ == "__main__":
    main()