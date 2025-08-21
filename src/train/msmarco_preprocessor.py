import random
from typing import Dict, List, Tuple
import json
from pathlib import Path

seed = random.Random(42)

def export_to_jsonl(full_dataset: List[Dict], output_path: str):
    with open(output_path, 'w+', encoding='utf8') as f:
        for entry in full_dataset:
            topic_id = entry['topic_id']
            query = entry['query']
            for i, rel in enumerate(entry['relevant_segments']):
                json.dump({
                    "id": f"{topic_id}_{i}",
                    **rel
                }, f)
                f.write('\n')

            if entry.get("distractor_segments"):
                for i, rel in enumerate(entry['distractor_segments']):
                    json.dump({
                        "id": f"{topic_id}_d{i}",
                        **rel
                    }, f)
                    f.write('\n')

    print(f"✅ Archivo guardado en: {output_path}")


def load_topics(path: str) -> Dict[str, str]:
    topics = {}
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            topic_id, query = line.strip().split('\t')
            topics[topic_id] = query
    return topics


def load_qrels(path: str) -> Dict[str, List[Tuple[str, int]]]:
    qrels = {}
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            topic_id, _, segment_id, relevance = line.strip().split()
            if topic_id not in qrels:
                qrels[topic_id] = []
            qrels[topic_id].append((segment_id, segment_id.split('#')[0], int(relevance)))
    return qrels


def load_assessments(path: str) -> Dict[str, List[Tuple[str, int]]]:
    # Example: {"topic_id": "2024-153051", "run_id": "IITD-IRL.zeph_test_rag_rrf_expand_query", "sentences": [{"sentenceID": 0, "text": "Target stores use video surveillance systems to control theft and assist in the recovery of stolen merchandise.", "citations": [{"citationID": 0, "reference": "msmarco_v2.1_doc_19_2021688056#1_3321145933", "support": "2"}]}, {"sentenceID": 1, "text": "Each store has a dedicated staffing budget for asset protection and security employees who identify and apprehend shoplifting suspects.", "citations": [{"citationID": 0, "reference": "msmarco_v2.1_doc_19_2021688056#2_3321147942", "support": "2"}]}, {"sentenceID": 2, "text": "Target works closely with local law enforcement, and all shoplifting suspects will be arrested and prosecuted if proven guilty.", "citations": [{"citationID": 0, "reference": "msmarco_v2.1_doc_19_2021688056#1_3321145933", "support": "2"}]}, {"sentenceID": 3, "text": "Shoplifters are responsible for all criminal fines and legal fees incurred as a result of their arrest.", "citations": [{"citationID": 0, "reference": "msmarco_v2.1_doc_19_2021688056#1_3321145933", "support": "2"}]}]}
    import json
    assessments = {}
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            assessment = json.loads(line)
            topic_id = assessment['topic_id']
            for sentence in assessment['sentences']:
                # Each sentence can be repeated as an assessment of multiple segments (multiple cites). 
                text = sentence['text']
                for citation in sentence['citations']:
                    segment_id = citation['reference']
                    support = citation['support']
                    if (topic_id, segment_id) not in assessments:
                        assessments[topic_id, segment_id] = []
                    assessments[topic_id, segment_id].append({
                        'support': support,
                        'text': text
                    })
    return assessments


def build_topic_segments_dataset(
    topics_path: str,
    qrels_path: str,
    assessments_path_with_pred: str,
    assessments_path_no_pred: str,
    output_path: str,
    min_relevance: int = 3  # >=2 usually as "relevant", using 3 as golden_truth
) -> List[Dict]:
        
    topics = load_topics(topics_path)
    qrels = load_qrels(qrels_path)
    assessments_with_pred = load_assessments(assessments_path_with_pred)
    assessments_no_pred = load_assessments(assessments_path_no_pred)
    
    dataset_accum = []

    for topic_id, query in topics.items():
        if topic_id not in qrels:
            continue
        relevant = [
            {
                'seg_id': seg_id, 
                'doc_id': doc_id, 
                'relevance': rel,
                'assessments': [
                    {
                        'text': assessment['text'],
                        'support': assessment['support'],
                        'type': 'with_prediction'
                    } for assessment in assessments_with_pred.get((topic_id, seg_id), [])
                ] + [
                    {
                        'text': assessment['text'],
                        'support': assessment['support'],
                        'type': 'without_prediction'
                    } for assessment in assessments_no_pred.get((topic_id, seg_id),[])
                ]
            } for seg_id, doc_id, rel in qrels[topic_id]
            if rel >= min_relevance
        ]
        relevant_doc_ids = [rel['doc_id'] for rel in relevant]
        distractors = [
            {
                'seg_id': seg_id, 
                'doc_id': doc_id, 
                'relevance': rel, 
                'distractor_type': "hard_neg" if doc_id in relevant_doc_ids else "soft_neg",
                'assessments': [
                    {
                        'text': assessment['text'],
                        'support': assessment['support'],
                        'type': 'with_prediction'
                    } for assessment in assessments_with_pred.get((topic_id, seg_id),[])
                ] + [
                    {
                        'text': assessment['text'],
                        'support': assessment['support'],
                        'type': 'without_prediction'
                    } for assessment in assessments_no_pred.get((topic_id, seg_id),[])
                ]
            }
            for seg_id, doc_id, rel in qrels[topic_id] if rel < min_relevance
        ]

        dataset_accum.append({
            "topic_id": topic_id,
            "query": query,
            "relevant_segments": relevant,
            "distractor_segments": distractors
        })

    export_to_jsonl(dataset_accum, output_path)
    return dataset_accum


if __name__=='__main__':
    root_dir = str(Path('data').resolve()) + '/'

    dataset = build_topic_segments_dataset(
        topics_path=root_dir + '/TREC-RAG-24/topics.rag24.test.txt',
        qrels_path=root_dir + '/TREC-RAG-24/qrels.rag24.test-umbrela-all.txt',
        assessments_path_with_pred=root_dir + '/TREC-RAG-24/final.citation_judgments_with_prediction.20241025.jsonl',
        assessments_path_no_pred=root_dir + '/TREC-RAG-24/final.citation_judgments_without_prediction.20241025.jsonl',
        output_path=root_dir + '/TREC-RAG-24/msmarco_topic_with_qrels_assessments.jsonl'
    )
