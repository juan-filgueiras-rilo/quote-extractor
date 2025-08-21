import json
import argparse
import logging
import gzip
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import random
import torch
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class MSMarcoSegment:
    segment_id: str
    text: str
    relevance: int
    label: str


@dataclass
class MSMarcoTopic:
    query: str
    segments: List[MSMarcoSegment]


class MSMarcoSegmentRetriever:
    
    def __init__(self, segments_dir: str, cache_size: int = 1000):
        self.segments_dir = segments_dir
        self.cache_size = cache_size
        self.text_cache = {}  # LRU-style cache
        self.file_index = {}
        self._build_file_index()
    
    def _build_file_index(self):
        if not os.path.exists(self.segments_dir):
            raise ValueError(f"Segments directory not found: {self.segments_dir}")
        
        for fname in os.listdir(self.segments_dir):
            if fname.endswith('.json.gz'):
                # Extract file: msmarco_v2.1_doc_segmented_XX.json.gz
                parts = fname.replace('.json.gz', '').split('_')
                if len(parts) >= 5:
                    try:
                        # Handle both formats: _XX and _segmented_XX
                        if 'segmented' in fname:
                            idx = parts.index('segmented') + 1
                            if idx < len(parts):
                                file_num = parts[idx].zfill(2)
                                self.file_index[file_num] = os.path.join(self.segments_dir, fname)
                    except (ValueError, IndexError):
                        logger.warning(f"Could not parse file number from: {fname}")
        
        if not self.file_index:
            raise ValueError(f"No valid segment files found in {self.segments_dir}")
        
        logger.info(f"Indexed {len(self.file_index)} segment files")
    
    def _extract_file_number(self, segment_id: str) -> Optional[str]:
        # Format: msmarco_v2.1_doc_XX_XXXXXXXXX#Y_ZZZZZZZZ
        parts = segment_id.split('_')
        if len(parts) >= 4 and parts[0] == 'msmarco':
            file_num = parts[3].zfill(2)
            return file_num
        return None
    
    def _manage_cache(self):
        if len(self.text_cache) > self.cache_size:
            # Remove 20% of oldest items
            items_to_remove = int(self.cache_size * 0.2)
            for _ in range(items_to_remove):
                self.text_cache.pop(next(iter(self.text_cache)))
    
    def get_segment_text(self, segment_id: str) -> Optional[str]:
        # cache first
        if segment_id in self.text_cache:
            text = self.text_cache.pop(segment_id)
            self.text_cache[segment_id] = text
            return text
        
        file_num = self._extract_file_number(segment_id)
        if not file_num or file_num not in self.file_index:
            logger.debug(f"File not found for segment: {segment_id} (file_num: {file_num})")
            return None
        
        fpath = self.file_index[file_num]
        
        try:
            with gzip.open(fpath, 'rt', encoding='utf-8') as f:
                for line in f:
                    try:
                        obj = json.loads(line.strip())
                        if obj.get("id") == segment_id:
                            text = obj.get("text", "")
                            # Add to cache
                            self.text_cache[segment_id] = text
                            self._manage_cache()
                            return text
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Error reading segment {segment_id} from {fpath}: {e}")
        
        return None
    
    def get_segments_batch(self, segment_ids: List[str]) -> Dict[str, str]:
        results = {}
        segments_by_file = {}
        
        # check cache
        remaining_ids = []
        for seg_id in segment_ids:
            if seg_id in self.text_cache:
                results[seg_id] = self.text_cache[seg_id]
            else:
                remaining_ids.append(seg_id)
        
        # Group segments by file
        for seg_id in remaining_ids:
            file_num = self._extract_file_number(seg_id)
            if file_num and file_num in self.file_index:
                if file_num not in segments_by_file:
                    segments_by_file[file_num] = set()
                segments_by_file[file_num].add(seg_id)
        
        # Read file only once
        for file_num, seg_ids_set in segments_by_file.items():
            fpath = self.file_index[file_num]
            try:
                with gzip.open(fpath, 'rt', encoding='utf-8') as f:
                    for line in f:
                        if not seg_ids_set:  # All found
                            break
                        try:
                            obj = json.loads(line.strip())
                            seg_id = obj.get("id")
                            if seg_id in seg_ids_set:
                                text = obj.get("text", "")
                                results[seg_id] = text
                                self.text_cache[seg_id] = text
                                seg_ids_set.remove(seg_id)
                        except json.JSONDecodeError:
                            continue
                
                for missing_id in seg_ids_set:
                    logger.debug(f"Segment not found in file: {missing_id}")
                    
            except Exception as e:
                logger.error(f"Error reading file {fpath}: {e}")
        
        self._manage_cache()
        return results


class MSMarcoTopicAdapter:
    
    def __init__(self, data_file: str, segment_retriever: MSMarcoSegmentRetriever):
        self.segment_retriever = segment_retriever
        self.topics = {}
        self._load_topics(data_file)
    
    def _load_topics(self, data_file: str):
        logger.info(f"Loading topics from {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    query = data['query']
                    
                    if query not in self.topics:
                        self.topics[query] = MSMarcoTopic(query=query, segments=[])
                    
                    segment = MSMarcoSegment(
                        segment_id=data['segment_id'],
                        text="",  # Will be loaded when we needed
                        relevance=data['relevance'],
                        label=data['label']
                    )
                    self.topics[query].segments.append(segment)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing line: {e}")
        
        # Important: sort segments by relevance for each topic
        for topic in self.topics.values():
            topic.segments.sort(key=lambda s: s.relevance, reverse=True)
        
        logger.info(f"Loaded {len(self.topics)} unique topics")
    
    def get_topic(self, query: str) -> Optional[MSMarcoTopic]:
        return self.topics.get(query)
    
    def get_all_topics(self) -> List[MSMarcoTopic]:
        return list(self.topics.values())
    
    def retrieve_topic_segments(self, topic: MSMarcoTopic) -> MSMarcoTopic:
        segment_ids = [s.segment_id for s in topic.segments]
        texts = self.segment_retriever.get_segments_batch(segment_ids)
        
        # Populate texts
        populated_segments = []
        for segment in topic.segments:
            text = texts.get(segment.segment_id, "")
            if text:
                populated_segments.append(MSMarcoSegment(
                    segment_id=segment.segment_id,
                    text=text,
                    relevance=segment.relevance,
                    label=segment.label
                ))
        
        return MSMarcoTopic(query=topic.query, segments=populated_segments)


class MSMarcoTrainingDataProcessor:
    
    def __init__(self, 
                 num_golden: int = 2,
                 max_documents: int = 3,
                 min_relevance_golden: int = 3,
                 max_relevance_distractor: int = 2,
                 max_segments_per_doc: int = 10):
        self.num_golden = num_golden
        self.max_documents = max_documents
        self.min_relevance_golden = min_relevance_golden
        self.max_relevance_distractor = max_relevance_distractor
        self.max_segments_per_doc = max_segments_per_doc
    
    def _extract_document_id(self, segment_id: str) -> str:
        # doc_id: msmarco_v2.1_doc_44_1043805224#0_2182638216 -> msmarco_v2.1_doc_44_1043805224
        return segment_id.split('#')[0] if '#' in segment_id else segment_id
    
    def _extract_segment_number(self, segment_id: str) -> int:
        # seg_id: msmarco_v2.1_doc_44_1043805224#0_2182638216 -> 0
        if '#' in segment_id:
            parts = segment_id.split('#')[1].split('_')
            if parts and parts[0].isdigit():
                return int(parts[0])
        return -1
    
    def select_documents_and_segments(self, topic: MSMarcoTopic) -> Dict[str, List[MSMarcoSegment]]:
        """
        Strategy:
        1. Group all segments by document
        2. Prioritize documents with high-relevance segments
        3. Include complete documents (all available segments) when possible
        4. Ensure we have the target number of golden segments
        """
        segments_by_doc = {}
        for segment in topic.segments:
            doc_id = self._extract_document_id(segment.segment_id)
            if doc_id not in segments_by_doc:
                segments_by_doc[doc_id] = []
            segments_by_doc[doc_id].append(segment)
        
        # Sort by segment number
        for doc_id in segments_by_doc:
            segments_by_doc[doc_id].sort(
                key=lambda s: self._extract_segment_number(s.segment_id)
            )
        
        # Score documents by their best segment relevance and count of good segments
        doc_scores = {}
        golden_segments_by_doc = {}
        
        for doc_id, segments in segments_by_doc.items():
            max_relevance = max(s.relevance for s in segments)
            high_relevance_count = sum(1 for s in segments if s.relevance >= self.min_relevance_golden)
            
            # prioritize docs with golden segments, then add max relevance marker as criteria
            doc_scores[doc_id] = (high_relevance_count * 10) + max_relevance
            
            # track golden segments of docs
            golden_segments_by_doc[doc_id] = [
                s for s in segments if s.relevance >= self.min_relevance_golden
            ]
        
        sorted_docs = sorted(doc_scores.keys(), key=lambda d: doc_scores[d], reverse=True)
        
        selected_docs = {}
        total_golden = 0
        
        for doc_id in sorted_docs:
            if len(selected_docs) >= self.max_documents:
                break
            
            if golden_segments_by_doc[doc_id]: 
                # important: limit segments per document to avoid too long contexts
                doc_segments = segments_by_doc[doc_id][:self.max_segments_per_doc]
                selected_docs[doc_id] = doc_segments
                total_golden += len([s for s in doc_segments if s.relevance >= self.min_relevance_golden])
                
                if total_golden >= self.num_golden:
                    break
        
        # if we have "room", add pure distractor documents
        if len(selected_docs) < self.max_documents:
            for doc_id in sorted_docs:
                if len(selected_docs) >= self.max_documents:
                    break
                
                if doc_id not in selected_docs:
                    doc_segments = segments_by_doc[doc_id][:self.max_segments_per_doc]
                    
                    # only add if it has relevant distractors
                    if any(s.relevance <= self.max_relevance_distractor for s in doc_segments):
                        selected_docs[doc_id] = doc_segments
        
        return selected_docs
    
    def convert_to_hotpot_format(self, 
                                 query: str,
                                 documents: Dict[str, List[MSMarcoSegment]]) -> Dict[str, Any]:
        context = []
        supporting_facts = []
        golden_segments = []
        distractor_segments = []
        
        doc_items = list(documents.items())
        random.shuffle(doc_items)  # avoid position bias
        
        for doc_idx, (doc_id, segments) in enumerate(doc_items):
            # document number is title
            doc_parts = doc_id.split('_')
            doc_num = doc_parts[3] if len(doc_parts) > 3 else str(doc_idx)
            title = f"Document_{doc_num}"
            
            # each segment is a "sentence"
            sentences = []
            segment_positions = {} 
            for seg_idx, segment in enumerate(segments):
                if segment.text:
                    sentences.append(segment.text.strip())
                    segment_positions[segment.segment_id] = seg_idx
                    
                    # track golden vs distractor
                    if segment.relevance >= self.min_relevance_golden:
                        golden_segments.append(segment.segment_id)
                        supporting_facts.append([title, seg_idx])
                    else:
                        distractor_segments.append(segment.segment_id)
            
            if sentences: # safe check
                context.append((title, sentences))
        
        # HotpotQA format: [doc_idx, sentence_idx]
        final_supporting_facts = []
        title_to_idx = {title: idx for idx, (title, _) in enumerate(context)}
        
        for title, seg_idx in supporting_facts:
            if title in title_to_idx:
                final_supporting_facts.append([title_to_idx[title], seg_idx])
        
        return {
            "question": query,
            "context": context,
            "supporting_facts": final_supporting_facts,
            "answer": "",  # to be generated 
            "metadata": {
                "golden_count": len(golden_segments),
                "distractor_count": len(distractor_segments),
                "golden_segments": golden_segments,
                "distractor_segments": distractor_segments,
                "num_documents": len(context),
                "total_segments": len(golden_segments) + len(distractor_segments)
            }
        }
    
    def select_training_segments(self, topic: MSMarcoTopic) -> Tuple[List[MSMarcoSegment], List[MSMarcoSegment]]:
        # Legacy
        golden_segments = []
        distractor_segments = []
        
        for segment in topic.segments:
            if segment.relevance >= self.min_relevance_golden:
                golden_segments.append(segment)
            elif segment.relevance <= self.max_relevance_distractor:
                distractor_segments.append(segment)
        
        golden_segments.sort(key=lambda s: s.relevance, reverse=True)
        distractor_segments.sort(key=lambda s: s.relevance, reverse=True)
        
        return golden_segments[:self.num_golden], distractor_segments


class MSMarcoDataset(Dataset):
    
    def __init__(self,
                 topic_adapter: MSMarcoTopicAdapter,
                 data_processor: MSMarcoTrainingDataProcessor,
                 max_samples: int = -1):
        
        self.topic_adapter = topic_adapter
        self.data_processor = data_processor
        
        self.topics = topic_adapter.get_all_topics()
        
        if max_samples > 0:
            self.topics = self.topics[:max_samples]
        
        logger.info(f"Dataset initialized with {len(self.topics)} topics")
    
    def __len__(self):
        return len(self.topics)
    
    def __getitem__(self, idx):
        topic = self.topics[idx]
        
        topic_with_text = self.topic_adapter.retrieve_topic_segments(topic)
        
        selected_documents = self.data_processor.select_documents_and_segments(topic_with_text)
        
        hotpot_data = self.data_processor.convert_to_hotpot_format(
            topic.query, selected_documents
        )
        
        return hotpot_data


class UnslothMSMarcoTrainer:
    
    def __init__(self,
                 model_name: str,
                 max_seq_length: int = 2048,
                 load_in_4bit: bool = True):
        
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.model = None
        self.tokenizer = None
        
        self._init_model(load_in_4bit)
    
    def _init_model(self, load_in_4bit: bool):
        try:
            from unsloth import FastLanguageModel
            
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=self.max_seq_length,
                dtype=None,
                load_in_4bit=load_in_4bit,
            )
            
            logger.info(f"Loaded model: {self.model_name}")
            
        except ImportError:
            logger.error("Unsloth not installed. Install with: pip install unsloth")
            raise
    
    def prepare_for_training(self,
                             r: int = 16,
                             lora_alpha: int = 16,
                             lora_dropout: float = 0,
                             target_modules: Optional[List[str]] = None):
        from unsloth import FastLanguageModel
        
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                              "gate_proj", "up_proj", "down_proj"]
        
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            use_gradient_checkpointing=True,
            random_state=3407,
        )
        
        logger.info(f"Model prepared with LoRA (r={r}, alpha={lora_alpha})")
    
    def create_prompt(self, sample: Dict[str, Any], include_answer: bool = True) -> str:
        from ..templates.hotpot_qa import HotpotQAPromptTemplate
        
        template = HotpotQAPromptTemplate()
        
        # Create prompt
        prompt = template.render(
            question=sample['question'],
            context=sample['context'],
            include_structured_instruction=True,
            template_style='detailed'
        )
        
        if include_answer:
            # Generate answer based on golden segments
            # TODO: Implement answer generation strategy
            answer = {
                "answer": "Based on the provided context...",
                "supporting_facts": sample['supporting_facts'],
                "reasoning": f"Answer derived from {len(sample['supporting_facts'])} supporting facts."
            }
            
            prompt += "\n" + json.dumps(answer, indent=2)
        
        return prompt
    
    def compute_custom_loss(self, outputs, labels, sample_metadata=None):
        """
        Custom loss function for MSMarco training.
        
        TODO: Implement your custom loss here.
        
        Possible implementations:
        1. Weighted loss based on golden vs distractor presence
        2. Contrastive loss between relevant and non-relevant segments
        3. Multi-task loss combining answer generation and relevance prediction
        """
        # Placeholder - implement your custom loss
        return None
    
    def train(self,
             train_dataset: MSMarcoDataset,
             val_dataset: Optional[MSMarcoDataset] = None,
             output_dir: str = "./checkpoints",
             num_epochs: int = 1,
             batch_size: int = 2,
             learning_rate: float = 2e-4,
             warmup_steps: int = 5,
             logging_steps: int = 10,
             save_steps: int = 100,
             gradient_accumulation_steps: int = 4):
        """Train the model"""
        
        from transformers import TrainingArguments
        from trl import SFTTrainer
        from datasets import Dataset as HFDataset
        
        # Prepare training data
        logger.info("Preparing training data...")
        train_texts = []
        
        for i in range(len(train_dataset)):
            try:
                sample = train_dataset[i]
                prompt = self.create_prompt(sample, include_answer=True)
                train_texts.append(prompt)
            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                continue
        
        if not train_texts:
            raise ValueError("No valid training samples generated")
        
        # Convert to HuggingFace dataset
        hf_train_dataset = HFDataset.from_dict({"text": train_texts})
        
        # Prepare validation data
        hf_val_dataset = None
        if val_dataset:
            logger.info("Preparing validation data...")
            val_texts = []
            for i in range(len(val_dataset)):
                try:
                    sample = val_dataset[i]
                    prompt = self.create_prompt(sample, include_answer=True)
                    val_texts.append(prompt)
                except Exception as e:
                    logger.warning(f"Error processing validation sample {i}: {e}")
                    continue
            
            if val_texts:
                hf_val_dataset = HFDataset.from_dict({"text": val_texts})
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate,
            fp16=torch.cuda.is_available(),
            logging_steps=logging_steps,
            save_steps=save_steps,
            evaluation_strategy="steps" if hf_val_dataset else "no",
            eval_steps=save_steps if hf_val_dataset else None,
            save_strategy="steps",
            load_best_model_at_end=True if hf_val_dataset else False,
            report_to="none",
            remove_unused_columns=False,
            label_names=[],
        )
        
        # Initialize trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=hf_train_dataset,
            eval_dataset=hf_val_dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            args=training_args,
        )
        
        # Start training
        logger.info("Starting training...")
        logger.info(f"Training samples: {len(train_texts)}")
        logger.info(f"Validation samples: {len(val_texts) if val_dataset else 0}")
        
        trainer.train()
        
        # Save final model
        final_path = f"{output_dir}/final"
        logger.info(f"Saving model to {final_path}")
        trainer.save_model(final_path)
        self.tokenizer.save_pretrained(final_path)
        
        logger.info("Training complete!")
        
        return trainer


def main():
    parser = argparse.ArgumentParser(
        description='Train models on MSMarco using HotpotQA-style prompts with Unsloth',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--train-data', type=str, required=True,
                       help='Path to MSMarco training data (JSONL)')
    parser.add_argument('--val-data', type=str,
                       help='Path to validation data (JSONL)')
    parser.add_argument('--segments-dir', type=str, required=True,
                       help='Directory with msmarco_v2.1_doc_segmented_XX.json.gz files')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='unsloth/Llama-3.2-3B-Instruct',
                       help='Base model to fine-tune')
    parser.add_argument('--max-seq-length', type=int, default=2048,
                       help='Maximum sequence length')
    parser.add_argument('--load-in-4bit', action='store_true', default=True,
                       help='Load model in 4-bit')
    
    # Data processing arguments
    parser.add_argument('--num-golden', type=int, default=2,
                       help='Target number of golden truth segments')
    parser.add_argument('--max-documents', type=int, default=3,
                       help='Maximum number of documents to include in context')
    parser.add_argument('--min-relevance-golden', type=int, default=3,
                       help='Minimum relevance for golden segments')
    parser.add_argument('--max-relevance-distractor', type=int, default=2,
                       help='Maximum relevance for distractor segments')
    parser.add_argument('--max-segments-per-doc', type=int, default=10,
                       help='Maximum segments per document to avoid too long contexts')
    
    # LoRA arguments
    parser.add_argument('--lora-r', type=int, default=16,
                       help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=16,
                       help='LoRA alpha')
    parser.add_argument('--lora-dropout', type=float, default=0,
                       help='LoRA dropout')
    
    # Training arguments
    parser.add_argument('--output-dir', type=str, default='./checkpoints/msmarco',
                       help='Output directory')
    parser.add_argument('--num-epochs', type=int, default=1,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='Batch size')
    parser.add_argument('--gradient-accumulation', type=int, default=4,
                       help='Gradient accumulation steps')
    parser.add_argument('--learning-rate', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--max-samples', type=int, default=-1,
                       help='Max training samples (-1 for all)')
    parser.add_argument('--cache-size', type=int, default=1000,
                       help='Cache size for segment retriever')
    
    # Logging arguments
    parser.add_argument('--logging-steps', type=int, default=10,
                       help='Log every N steps')
    parser.add_argument('--save-steps', type=int, default=100,
                       help='Save every N steps')
    parser.add_argument('--warmup-steps', type=int, default=5,
                       help='Warmup steps')
    
    args = parser.parse_args()
    
    # Initialize segment retriever (handles memory-efficient reading)
    logger.info(f"Initializing segment retriever...")
    retriever = MSMarcoSegmentRetriever(
        segments_dir=args.segments_dir,
        cache_size=args.cache_size
    )
    
    # Initialize topic adapter (loads and organizes data by query)
    logger.info("Loading topics from training data...")
    topic_adapter = MSMarcoTopicAdapter(
        data_file=args.train_data,
        segment_retriever=retriever
    )
    
    # Initialize data processor (selects documents with golden/distractor segments)
    data_processor = MSMarcoTrainingDataProcessor(
        num_golden=args.num_golden,
        max_documents=args.max_documents,
        min_relevance_golden=args.min_relevance_golden,
        max_relevance_distractor=args.max_relevance_distractor,
        max_segments_per_doc=args.max_segments_per_doc
    )
    
    # Create training dataset
    train_dataset = MSMarcoDataset(
        topic_adapter=topic_adapter,
        data_processor=data_processor,
        max_samples=args.max_samples
    )
    
    # Create validation dataset if provided
    val_dataset = None
    if args.val_data:
        logger.info("Loading validation data...")
        val_topic_adapter = MSMarcoTopicAdapter(
            data_file=args.val_data,
            segment_retriever=retriever
        )
        val_dataset = MSMarcoDataset(
            topic_adapter=val_topic_adapter,
            data_processor=data_processor,
            max_samples=max(100, args.max_samples // 10) if args.max_samples > 0 else 100
        )
    
    # Initialize trainer
    logger.info(f"Initializing Unsloth trainer with model: {args.model}")
    trainer = UnslothMSMarcoTrainer(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit
    )
    
    # Prepare model for training
    trainer.prepare_for_training(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    
    # Start training
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        gradient_accumulation_steps=args.gradient_accumulation
    )
    
    logger.info("Training pipeline complete!")


if __name__ == "__main__":
    example_usage = """
    USAGE EXAMPLES:
    
    # Basic training
    python train_msmarco.py \\
        --train-data msmarco_quotes_data.jsonl \\
        --segments-dir /path/to/msmarco/segments \\
        --model unsloth/Llama-3.2-3B-Instruct
    
    # Custom golden/distractor configuration with document grouping
    python train_msmarco.py \\
        --train-data train.jsonl \\
        --val-data val.jsonl \\
        --segments-dir ./segments \\
        --num-golden 2 \\
        --max-documents 3 \\
        --min-relevance-golden 3 \\
        --max-relevance-distractor 2 \\
        --max-segments-per-doc 10
    
    # Memory-efficient training with cache control
    python train_msmarco.py \\
        --train-data msmarco_quotes_data.jsonl \\
        --segments-dir ./segments \\
        --cache-size 500 \\
        --batch-size 1 \\
        --gradient-accumulation 8
    
    # Full configuration
    python train_msmarco.py \\
        --train-data train.jsonl \\
        --val-data val.jsonl \\
        --segments-dir /data/msmarco \\
        --model unsloth/Llama-3.2-3B-Instruct \\
        --num-golden 2 \\
        --num-distractors 3 \\
        --lora-r 32 \\
        --lora-alpha 64 \\
        --learning-rate 5e-5 \\
        --num-epochs 3 \\
        --max-samples 1000
    """
    
    import sys
    if len(sys.argv) == 1:
        print(example_usage)
        sys.exit(0)
    
    main()