#!/usr/bin/env python3
"""
Convert HotpotQA results from our format to the official evaluation format.

Usage:
    python convert_to_eval_format.py input_results.json output_pred.json
    
    or just:
    python convert_to_eval_format.py results.json
    (will create results_pred.json)
"""

import json
import sys
import os
from pathlib import Path


def convert_to_eval_format(input_file, output_file=None):
    # Load input data
    print(f"Loading results from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print(f"Found {len(results)} predictions")
    
    answer_dict = {}
    sp_dict = {}
    
    for item in results:
        qid = item['_id']
        
        answer_dict[qid] = item['predicted_answer']        
        sp_dict[qid] = item['predicted_supporting_facts']
    
    output_data = {
        'answer': answer_dict,
        'sp': sp_dict
    }
    
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_pred.json"
    
    print(f"Saving evaluation format to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nConversion complete!")
    print(f"  - Total questions: {len(answer_dict)}")
    print(f"  - Average supporting facts per question: {sum(len(sp) for sp in sp_dict.values()) / len(sp_dict):.2f}")
    
    print(f"\nSample output (first question):")
    first_id = list(answer_dict.keys())[0]
    print(f"  ID: {first_id}")
    print(f"  Answer: {answer_dict[first_id]}")
    print(f"  Supporting facts: {sp_dict[first_id]}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_to_eval_format.py input_results.json [output_pred.json]")
        print("\nExample:")
        print("  python convert_to_eval_format.py results.json")
        print("  python convert_to_eval_format.py results.json dev_distractor_pred.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        sys.exit(1)
    
    convert_to_eval_format(input_file, output_file)


if __name__ == "__main__":
    main()