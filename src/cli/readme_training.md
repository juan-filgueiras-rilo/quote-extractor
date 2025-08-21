# HotpotQA Quote-based Training & Inference System

## Overview

This system implements a unified training and inference pipeline for HotpotQA using a quote-based approach. The key innovation is the complete reusability of components between training and inference phases.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Quote Templates                          │
│               (quote_hotpot_qa.py)                          │
│  - Consistent prompts for training & inference              │
│  - Structured JSON schema for outputs                       │
└──────────────────┬──────────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
┌───────────────┐    ┌────────────────┐
│   Training    │    │   Inference    │
│ (training.py) │    │ (inference.py) │
└───────┬───────┘    └────────┬───────┘
        │                     │
        ▼                     ▼
┌───────────────────────────────────┐
│          LLM Providers            │
├───────────────────────────────────┤
│ • UnslothTrainingProvider         │
│ • UnslothInferenceProvider        │
│ • OllamaOpenAIProvider            │
│ • HuggingFaceProvider             │
│ • OpenAIProvider                  │
└───────────────────────────────────┘
```

## Key Components

### 1. **QuoteHotpotQAProcessor** (Reused)
- Used for both training data preparation and inference
- Handles quote extraction and matching using Levenshtein distance
- Ensures consistency between training and inference

### 2. **Quote Templates** (Shared)
- `QuoteHotpotQAPromptTemplate`: Single source of truth for prompts
- Supports multiple styles: `detailed`, `concise`, `chain_of_thought`
- Same template used for training examples and inference prompts

### 3. **Providers** (Modular)
- **UnslothTrainingProvider**: Handles Unsloth model training with LoRA
- **UnslothInferenceProvider**: Loads and runs inference on trained models
- Other providers (Ollama, OpenAI, etc.) for comparison

## Installation

```bash
# Install required packages
pip install unsloth transformers datasets trl
pip install python-Levenshtein openai

# For Unsloth (recommended)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

## Usage

### Training

```bash
# Full training with Unsloth
python -m src.cli.training \
    --input data/hotpot_train.json \
    --output-dir models/quote_hotpot_llama3 \
    --model unsloth/llama-3-8b-bnb-4bit \
    --prompt-style detailed \
    --num-epochs 3 \
    --batch-size 4 \
    --learning-rate 2e-4

# Quick test with small dataset
python -m src.cli.training \
    --input data/hotpot_train.json \
    --output-dir models/test_model \
    --num-samples 100 \
    --num-epochs 1
```

### Inference

```bash
# Inference with trained Unsloth model
python -m src.cli.inference \
    --input data/hotpot_dev.json \
    --output results/unsloth_results.json \
    --provider unsloth \
    --model models/quote_hotpot_llama3 \
    --processor QuoteHotpotQAProcessor \
    --prompt-style detailed \
    --evaluate

# Compare with base model
python -m src.cli.inference \
    --input data/hotpot_dev.json \
    --output results/base_results.json \
    --provider ollama-openai \
    --model llama3.1:8b \
    --processor QuoteHotpotQAProcessor \
    --evaluate
```

## Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lora-r` | 16 | LoRA rank |
| `--lora-alpha` | 16 | LoRA alpha scaling |
| `--lora-dropout` | 0.0 | LoRA dropout |
| `--batch-size` | 4 | Training batch size |
| `--gradient-accumulation-steps` | 4 | Gradient accumulation |
| `--learning-rate` | 2e-4 | Learning rate |
| `--num-epochs` | 1 | Number of training epochs |

## Quote Matching Configuration

The `QuoteHotpotQAProcessor` uses Levenshtein distance for matching quotes to source sentences:

```python
# Default threshold: 0.9 (90% similarity)
processor = QuoteHotpotQAProcessor(
    llm_provider=provider,
    prompt_style="detailed",
    levenshtein_threshold=0.9
)
```

## Output Format

Both training and inference use the same JSON structure:

```json
{
    "answer": "The answer to the question",
    "supporting_quotes": [
        "Exact sentence from context",
        "Another exact supporting sentence"
    ],
    "reasoning": "Brief explanation"
}
```

## Advantages of This Approach

1. **Consistency**: Same templates and processors for training and inference
2. **Modularity**: Easy to swap providers without changing core logic
3. **Verifiability**: Exact quote matching ensures traceable answers
4. **Flexibility**: Supports multiple prompt styles and output formats
5. **Efficiency**: Unsloth optimization for faster training and inference

## Performance Optimization

### Training
- Use 4-bit quantization with Unsloth
- Gradient accumulation for larger effective batch sizes
- LoRA for parameter-efficient fine-tuning

### Inference
- Batch processing support
- Caching of tokenized inputs
- Optimized quote matching with Levenshtein distance

## Evaluation Metrics

The system automatically computes:
- **Exact Match (EM)**: Exact answer matching
- **Answer F1**: Token-level F1 score for answers
- **Supporting Facts F1**: F1 score for supporting fact identification
- **Joint EM**: Combined exact match for answer and supporting facts

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `--batch-size`
   - Increase `--gradient-accumulation-steps`
   - Use `--load-in-4bit`

2. **Quote Matching Failures**
   - Adjust `levenshtein_threshold` (lower = more lenient)
   - Check that training data has exact quotes

3. **JSON Parsing Errors**
   - The system has fallback parsing for malformed JSON
   - Check model temperature (lower = more consistent)

## Advanced Usage

### Custom Prompt Styles

Add new styles to `QuoteHotpotQAPromptTemplate`:

```python
def _render_custom_template(self, question, context, include_structured):
    # Your custom template logic
    pass
```

### Fine-tuning Hyperparameters

```bash
# Aggressive fine-tuning
python -m src.cli.training \
    --lora-r 64 \
    --lora-alpha 128 \
    --learning-rate 5e-4 \
    --num-epochs 5

# Conservative fine-tuning
python -m src.cli.training \
    --lora-r 8 \
    --lora-alpha 8 \
    --learning-rate 5e-5 \
    --num-epochs 1
```

## Model Export

```bash
# Export as LoRA adapter (smallest)
--save-method lora

# Export as merged 16-bit model
--save-method merged_16bit

# Export as GGUF for llama.cpp
--save-method gguf
```

## Contributing

The system is designed for easy extension:
1. Add new providers in `src/providers/`
2. Create new processors in `src/core/processors/`
3. Extend templates in `src/templates/`

All components follow the same interface patterns for consistency.