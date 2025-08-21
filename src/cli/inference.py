import json
import argparse
import logging
from pathlib import Path

from src.core.processors import *
from src.providers.huggingface import HuggingFaceLlamaProvider
from src.providers.ollama import Llama3Provider
from src.providers.ollama_openai import OllamaOpenAIProvider
from src.providers.openai import OpenAIProvider

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_processor(args, llm_provider, use_structured):
    if args.processor == 'HotpotQAProcessor':
        return HotpotQAProcessor(
            llm_provider=llm_provider,
            prompt_style=args.prompt_style,
            use_structured_output=use_structured
        )
    elif args.processor == 'QuoteHotpotQAProcessor':
        return QuoteHotpotQAProcessor(
            llm_provider=llm_provider,
            prompt_style=args.prompt_style,
            use_structured_output=use_structured
        )
    
def create_provider(args):
    if args.provider == 'ollama':
        return Llama3Provider(
            model_name=args.model, 
            api_url=args.api_url
        )
    elif args.provider == 'openai':
        return OpenAIProvider(
            model_name=args.model, 
            api_key=args.api_key
        )
    elif args.provider == 'huggingface':
        return HuggingFaceLlamaProvider(
            model_name=args.model,
            device=args.device,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            max_new_tokens=args.max_new_tokens,
            temperature=getattr(args, 'temperature', 0.1),
            auth_token=args.hf_token
        )
    elif args.provider == 'ollama-openai':
        return OllamaOpenAIProvider(
            model_name=args.model,
            base_url=args.api_url or "http://localhost:11434/v1",
            temperature=getattr(args, 'temperature', 0.1),
            max_tokens=args.max_new_tokens,
            enable_structured_output=not args.no_structured
        )
    else:
        raise ValueError(f"Unknown provider: {args.provider}")


def main():
    parser = argparse.ArgumentParser(
        description='HotpotQA Question Answering System with task-agnostic providers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with Ollama
  %(prog)s --input data/hotpot_dev.json --output results/test.json --num-samples 10

  # HuggingFace with quantization
  %(prog)s --provider huggingface --model meta-llama/Llama-3.2-3B-Instruct --load-in-4bit

  # OpenAI with evaluation
  %(prog)s --provider openai --model gpt-4 --api-key $OPENAI_KEY --evaluate

  # Structured output with Ollama
  %(prog)s --provider ollama-openai --model llama3.1:8b --prompt-style detailed
        """
    )
    
    # Input/Output arguments
    parser.add_argument('--input', type=str, required=True, 
                       help='Path to HotpotQA JSON file')
    parser.add_argument('--output', type=str, 
                       help='Path to save results (auto-generated if not specified)')
    
    # Processor arguments
    parser.add_argument('--processor', type=str, default='HotpotQAProcessor',
                       choices=['HotpotQAProcessor' ,'QuoteHotpotQAProcessor'],
                       help='Processor to use')
    
    # Provider arguments
    parser.add_argument('--provider', type=str, default='ollama-openai',
                       choices=['ollama', 'openai', 'huggingface', 'ollama-openai'],
                       help='LLM provider to use')
    parser.add_argument('--model', type=str, default='llama3.1:8b', 
                       help='Model name/identifier for the provider')
    
    # Task-specific arguments (for processor)
    parser.add_argument('--prompt-style', type=str, default='detailed',
                       choices=['detailed', 'concise', 'chain_of_thought'],
                       help='Style of HotpotQA prompts to use')
    parser.add_argument('--force-structured', action='store_true',
                       help='Force structured output even if provider says it does not support it')
    parser.add_argument('--no-structured', action='store_true',
                       help='Disable structured output even if provider supports it')
    
    # Processing arguments
    parser.add_argument('--num-samples', type=int, default=-1,
                       help='Number of samples to process (use -1 for all)')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate against ground truth')
    
    # Provider-specific arguments
    # OpenAI
    parser.add_argument('--api-key', type=str, 
                       help='API key for OpenAI')
    
    # Ollama/Ollama-OpenAI
    parser.add_argument('--api-url', type=str, 
                       help='API URL for Ollama providers')
    
    # HuggingFace
    parser.add_argument('--hf-token', type=str, 
                       help='HuggingFace auth token for gated models')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device for HuggingFace models')
    parser.add_argument('--load-in-4bit', action='store_true',
                       help='Use 4-bit quantization for HuggingFace models')
    parser.add_argument('--load-in-8bit', action='store_true',
                        help='Use 8-bit quantization for HuggingFace models')
    
    # Generation parameters (apply to all providers)
    parser.add_argument('--max-new-tokens', type=int, default=1024,
                       help='Maximum new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Generation temperature')
    
    args = parser.parse_args()
    
    if args.force_structured and args.no_structured:
        parser.error("Cannot specify both --force-structured and --no-structured")
    
    # Load dataset
    logger.info(f"Loading HotpotQA dataset from {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} questions from dataset")
    
    logger.info(f"Initializing {args.provider} provider...")
    llm_provider = create_provider(args)
    
    logger.info(f"Provider configuration: {llm_provider.get_generation_config()}")
    logger.info(f"Provider supports structured output: {llm_provider.supports_structured_output()}")
    
    use_structured = None 
    if args.force_structured:
        use_structured = True
        logger.info("Forcing structured output (may not work well with all providers)")
    elif args.no_structured:
        use_structured = False
        logger.info("Disabling structured output")
    
    logger.info(f"Initializing {args.processor} processor with prompt style: {args.prompt_style}")
    processor = create_processor(args, llm_provider, use_structured)
    
    results = []
    
    if args.num_samples == -1 or args.num_samples >= len(data):
        num_samples = len(data)
        logger.info(f"Processing ALL {num_samples} questions")
    else:
        num_samples = min(args.num_samples, len(data))
        logger.info(f"Processing {num_samples} out of {len(data)} questions")
    
    for i, question_data in enumerate(data[:num_samples]):
        logger.info(f"Processing question {i+1}/{num_samples}")
        
        try:
            # Process the question using HotpotQA-specific logic
            response = processor.process_question(question_data)
            
            result = {
                '_id': question_data.get('_id'),
                'question': question_data['question'],
                'predicted_answer': response.answer,
                'predicted_supporting_facts': [
                    [sf.title, sf.sentence_idx] for sf in response.supporting_facts
                ],
                'supporting_sentences': [sf.sentence for sf in response.supporting_facts],
                'raw_response': response.raw_response if hasattr(response, 'raw_response') else None
            }
            
            if args.evaluate:
                metrics = processor.evaluate_response(response, question_data)
                result['metrics'] = metrics
                result['ground_truth_answer'] = question_data.get('answer')
                result['ground_truth_supporting_facts'] = question_data.get('supporting_facts', [])
                
                logger.info(f"  Answer: {response.answer[:100]}...")
                logger.info(f"  Exact Match: {metrics.get('exact_match', False)}")
                if 'supporting_facts_f1' in metrics:
                    logger.info(f"  Supporting Facts F1: {metrics['supporting_facts_f1']:.2f}")
                if 'answer_f1' in metrics:
                    logger.info(f"  Answer F1: {metrics['answer_f1']:.2f}")
            else:
                logger.info(f"  Answer: {response.answer[:100]}...")
                logger.info(f"  Supporting Facts: {len(response.supporting_facts)} found")
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing question {i+1}: {e}")
            error_result = {
                '_id': question_data.get('_id'),
                'question': question_data['question'],
                'predicted_answer': f"ERROR: {str(e)}",
                'predicted_supporting_facts': [],
                'supporting_sentences': [],
                'error': str(e)
            }
            results.append(error_result)
            continue
    
    if not args.output:
        input_path = Path(args.input)
        timestamp = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f"results/{input_path.stem}_{args.provider}_{args.model.replace('/', '_').replace(':', '_')}_{timestamp}.json"
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {output_path}")
    
    successful_results = [r for r in results if 'error' not in r]
    
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total Questions: {len(results)}")
    print(f"Successful: {len(successful_results)}")
    print(f"Errors: {len(results) - len(successful_results)}")
    print(f"Provider: {args.provider} ({args.model})")
    print(f"Prompt Style: {args.prompt_style}")
    print(f"Structured Output: {processor.use_structured_output}")
    
    if args.evaluate and successful_results:
        exact_matches = sum(1 for r in successful_results 
                          if r.get('metrics', {}).get('exact_match', False))
        avg_answer_f1 = sum(r.get('metrics', {}).get('answer_f1', 0) 
                           for r in successful_results) / len(successful_results)
        avg_sf_f1 = sum(r.get('metrics', {}).get('supporting_facts_f1', 0) 
                       for r in successful_results) / len(successful_results)
        
        print("\nEVALUATION METRICS")
        print("-" * 30)
        print(f"Exact Match: {exact_matches}/{len(successful_results)} ({100*exact_matches/len(successful_results):.1f}%)")
        print(f"Answer F1: {avg_answer_f1:.3f}")
        print(f"Supporting Facts F1: {avg_sf_f1:.3f}")
        
        joint_matches = sum(1 for r in successful_results 
                          if (r.get('metrics', {}).get('exact_match', False) and 
                              r.get('metrics', {}).get('supporting_facts_f1', 0) > 0.9))
        print(f"Joint EM (Answer + SF): {joint_matches}/{len(successful_results)} ({100*joint_matches/len(successful_results):.1f}%)")
    
    print("="*60)
    logger.info("Processing complete!")


if __name__ == "__main__":
    example_usage = """
    EXAMPLES OF TASK-AGNOSTIC USAGE:
    
    # 1. Basic HotpotQA with Ollama (structured output)
    python -m src.cli.inference \\
        --input data/hotpot_dev_distractor_v1.json \\
        --output results/basic_test.json \\
        --provider ollama-openai \\
        --model llama3.2:latest \\
        --num-samples 10 \\
        --evaluate
    
    # 2. HuggingFace with detailed prompts and quantization
    python -m src.cli.inference \\
        --provider huggingface \\
        --model meta-llama/Llama-3.2-3B-Instruct \\
        --device cuda \\
        --load-in-4bit \\
        --prompt-style detailed \\
        --temperature 0.1 \\
        --max-new-tokens 1024 \\
        --hf-token YOUR_TOKEN
    
    # 3. OpenAI with chain-of-thought prompting
    python -m src.cli.inference \\
        --provider openai \\
        --model gpt-4 \\
        --api-key YOUR_API_KEY \\
        --prompt-style chain_of_thought \\
        --temperature 0.0 \\
        --evaluate
    
    # 4. Concise prompts without structured output
    python -m src.cli.inference \\
        --provider ollama-openai \\
        --model mistral:latest \\
        --prompt-style concise \\
        --no-structured \\
        --temperature 0.2
    
    # 5. Full dataset processing with detailed evaluation
    python -m src.cli.inference \\
        --input data/hotpot_dev_distractor_v1.json \\
        --provider ollama-openai \\
        --model llama3.1:8b \\
        --num-samples -1 \\
        --prompt-style detailed \\
        --evaluate \\
        --output results/full_evaluation.json
    
    # 6. Force structured output with a provider that doesn't support it well
    python -m src.cli.inference \\
        --provider huggingface \\
        --model meta-llama/Llama-3.2-1B-Instruct \\
        --force-structured \\
        --prompt-style concise
    
    TASK-AGNOSTIC DESIGN:
    - Providers handle only generation and parsing
    - HotpotQAProcessor handles task-specific logic
    - Same providers can be reused for other tasks
    - Prompt templates are task-specific and configurable
    """
    
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        print("No arguments provided. Here's how to use this script:")
        print(example_usage)
