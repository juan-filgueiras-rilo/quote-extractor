    # HuggingFace
    parser.add_argument('--hf-token', type=str, 
                       help='HuggingFace auth token for gated models')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device for HuggingFace models')
    parser.add_argument('--load-in-4bit', action='store_true',
                       help='Use 4-bit quantization for HuggingFace/Unsloth models')
    parser.add_import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

from src.core.processors import QuoteHotpotQAProcessor
from src.templates.quote_hotpot_qa import QuoteHotpotQAPromptTemplate, QUOTE_HOTPOT_QA_SCHEMA

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UnslothTrainingProvider:
    """Provider for Unsloth fine-tuning, following the same design pattern as inference providers"""
    
    def __init__(self,
                 model_name: str = "unsloth/llama-3-8b-bnb-4bit",
                 max_seq_length: int = 2048,
                 dtype: Optional[torch.dtype] = None,
                 load_in_4bit: bool = True,
                 lora_r: int = 16,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.0,
                 target_modules: Optional[List[str]] = None,
                 use_gradient_checkpointing: str = "unsloth"):
        
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.lora_config = {
            'r': lora_r,
            'lora_alpha': lora_alpha,
            'lora_dropout': lora_dropout,
            'target_modules': target_modules or ["q_proj", "k_proj", "v_proj", "o_proj",
                                                  "gate_proj", "up_proj", "down_proj"],
            'use_gradient_checkpointing': use_gradient_checkpointing
        }
        
        logger.info(f"Initializing Unsloth provider with model: {model_name}")
        
        # Load model and tokenizer
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        
        # Apply LoRA
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            **self.lora_config,
            bias="none",
            use_rslora=False,
            loftq_config=None,
        )
        
        logger.info(f"Model loaded with LoRA config: {self.lora_config}")
    
    def get_model_and_tokenizer(self):
        """Return model and tokenizer for trainer"""
        return self.model, self.tokenizer
    
    def save_model(self, output_dir: str, save_method: str = "lora"):
        """Save the trained model"""
        if save_method == "lora":
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
        elif save_method == "merged_16bit":
            self.model.save_pretrained_merged(output_dir, self.tokenizer, save_method="merged_16bit")
        elif save_method == "gguf":
            self.model.save_pretrained_gguf(output_dir, self.tokenizer)
        else:
            raise ValueError(f"Unknown save method: {save_method}")
        
        logger.info(f"Model saved to {output_dir} using method: {save_method}")


class QuoteHotpotQADataProcessor:
    """Process HotpotQA data for training, reusing the QuoteHotpotQAPromptTemplate"""
    
    def __init__(self, 
                 prompt_style: str = "detailed",
                 max_seq_length: int = 2048,
                 use_structured_output: bool = True):
        
        self.prompt_template = QuoteHotpotQAPromptTemplate()
        self.prompt_style = prompt_style
        self.max_seq_length = max_seq_length
        self.use_structured_output = use_structured_output
        
        logger.info(f"Initialized data processor with style: {prompt_style}")
    
    def prepare_training_example(self, example: Dict[str, Any]) -> Dict[str, str]:
        """Convert a HotpotQA example to training format"""
        
        # Extract question and context
        question = example['question']
        context = example['context']
        answer = example['answer']
        supporting_facts = example.get('supporting_facts', [])
        
        # Generate the input prompt using the same template as inference
        input_prompt = self.prompt_template.render(
            question=question,
            context=context,
            include_structured_instruction=self.use_structured_output,
            template_style=self.prompt_style
        )
        
        # Create the expected output
        if self.use_structured_output:
            # Extract supporting sentences based on supporting_facts
            supporting_quotes = []
            for title, sent_idx in supporting_facts:
                # Find the document with this title
                for doc_title, sentences in context:
                    if doc_title == title and sent_idx < len(sentences):
                        supporting_quotes.append(sentences[sent_idx])
                        break
            
            # Create structured output
            output = json.dumps({
                "answer": answer,
                "supporting_quotes": supporting_quotes,
                "reasoning": f"The answer is derived from the supporting facts in documents about {', '.join(set([sf[0] for sf in supporting_facts]))}"
            }, indent=2)
        else:
            # Create unstructured output
            output = f"Answer: {answer}\n\nSupporting Evidence:\n"
            for title, sent_idx in supporting_facts:
                for doc_title, sentences in context:
                    if doc_title == title and sent_idx < len(sentences):
                        output += f"- {sentences[sent_idx]}\n"
                        break
        
        return {
            "input": input_prompt,
            "output": output
        }
    
    def prepare_dataset(self, data: List[Dict[str, Any]], 
                       tokenizer,
                       num_samples: Optional[int] = None) -> Dataset:
        """Prepare the complete dataset for training"""
        
        if num_samples and num_samples > 0:
            data = data[:num_samples]
        
        logger.info(f"Processing {len(data)} examples...")
        
        # Process all examples
        processed_examples = []
        for i, example in enumerate(data):
            try:
                processed = self.prepare_training_example(example)
                
                # Format for chat template
                messages = [
                    {"role": "user", "content": processed["input"]},
                    {"role": "assistant", "content": processed["output"]}
                ]
                
                # Apply chat template
                formatted_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                
                processed_examples.append({"text": formatted_text})
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(data)} examples")
                    
            except Exception as e:
                logger.error(f"Error processing example {i}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(processed_examples)} examples")
        
        # Convert to Dataset
        dataset = Dataset.from_list(processed_examples)
        return dataset


def create_trainer(args, provider, dataset):
    """Create the SFTTrainer with appropriate configuration"""
    
    model, tokenizer = provider.get_model_and_tokenizer()
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=args.logging_steps,
        save_strategy="steps" if args.save_steps > 0 else "epoch",
        save_steps=args.save_steps if args.save_steps > 0 else None,
        evaluation_strategy="no",  # We'll add validation later if needed
        optim=args.optimizer,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler,
        seed=args.seed,
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=False,  # Can set to True for longer sequences
        args=training_args,
    )
    
    return trainer


def main():
    parser = argparse.ArgumentParser(
        description='HotpotQA Fine-tuning with Unsloth using Quote-based approach',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with default settings
  %(prog)s --input data/hotpot_train.json --output-dir models/quote_hotpot_llama3

  # Training with specific LoRA configuration
  %(prog)s --input data/hotpot_train.json --model unsloth/llama-3-8b-bnb-4bit \\
           --lora-r 32 --lora-alpha 64 --num-epochs 3

  # Quick test with few samples
  %(prog)s --input data/hotpot_train.json --num-samples 100 --num-epochs 1 \\
           --output-dir models/test_run

  # Training with chain-of-thought prompts
  %(prog)s --input data/hotpot_train.json --prompt-style chain_of_thought \\
           --max-seq-length 4096 --batch-size 2
        """
    )
    
    # Data arguments
    parser.add_argument('--input', type=str, required=True,
                       help='Path to HotpotQA training JSON file')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save the trained model')
    parser.add_argument('--num-samples', type=int, default=-1,
                       help='Number of training samples to use (-1 for all)')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='unsloth/llama-3-8b-bnb-4bit',
                       help='Base model to fine-tune')
    parser.add_argument('--max-seq-length', type=int, default=2048,
                       help='Maximum sequence length')
    
    # LoRA arguments
    parser.add_argument('--lora-r', type=int, default=16,
                       help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=16,
                       help='LoRA alpha')
    parser.add_argument('--lora-dropout', type=float, default=0.0,
                       help='LoRA dropout')
    
    # Prompt arguments (same as inference)
    parser.add_argument('--prompt-style', type=str, default='detailed',
                       choices=['detailed', 'concise', 'chain_of_thought'],
                       help='Style of prompts to use for training')
    parser.add_argument('--no-structured', action='store_true',
                       help='Disable structured output format in training')
    
    # Training arguments
    parser.add_argument('--num-epochs', type=int, default=1,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Training batch size per device')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4,
                       help='Gradient accumulation steps')
    parser.add_argument('--learning-rate', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--warmup-steps', type=int, default=5,
                       help='Number of warmup steps')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw_8bit',
                       help='Optimizer to use')
    parser.add_argument('--lr-scheduler', type=str, default='linear',
                       help='Learning rate scheduler')
    
    # Logging and saving
    parser.add_argument('--logging-steps', type=int, default=10,
                       help='Log every N steps')
    parser.add_argument('--save-steps', type=int, default=0,
                       help='Save checkpoint every N steps (0 to save per epoch)')
    parser.add_argument('--save-method', type=str, default='lora',
                       choices=['lora', 'merged_16bit', 'gguf'],
                       help='Method to save the model')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Load dataset
    logger.info(f"Loading dataset from {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} training examples")
    
    # Initialize Unsloth provider
    logger.info("Initializing Unsloth training provider...")
    provider = UnslothTrainingProvider(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    
    # Initialize data processor (reusing the quote template)
    logger.info("Initializing data processor...")
    data_processor = QuoteHotpotQADataProcessor(
        prompt_style=args.prompt_style,
        max_seq_length=args.max_seq_length,
        use_structured_output=not args.no_structured
    )
    
    # Prepare dataset
    logger.info("Preparing training dataset...")
    num_samples = None if args.num_samples == -1 else args.num_samples
    dataset = data_processor.prepare_dataset(
        data=data,
        tokenizer=provider.tokenizer,
        num_samples=num_samples
    )
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = create_trainer(args, provider, dataset)
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save model
    logger.info(f"Saving model to {args.output_dir}...")
    provider.save_model(args.output_dir, save_method=args.save_method)
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Training samples: {len(dataset)}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Prompt style: {args.prompt_style}")
    print(f"Structured output: {not args.no_structured}")
    print(f"Model saved to: {args.output_dir}")
    print(f"Save method: {args.save_method}")
    print("="*60)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()