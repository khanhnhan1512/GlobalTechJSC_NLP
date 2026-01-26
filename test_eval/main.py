import json
import time
import os
import pandas as pd
import torch
import gc
from transformers import logging as transformers_logging
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer, CrossEncoder

# --- CONFIGURATION ---
transformers_logging.set_verbosity_error()

EMBED_MODEL = "hiieu/halong_embedding"
RERANKER_MODEL = "itdainb/PhoRanker"

INPUT_FOLDER = 'inputs'
INPUT_TEST_FILE = os.path.join(INPUT_FOLDER, 'gold_standard_dataset_2000.jsonl')
OUTPUT_FOLDER = f'outputs/sensitivity_test'
SUMMARY_FILE = os.path.join(OUTPUT_FOLDER, 'sensitivity_report_2000.csv')

MILVUS_URI = "http://127.0.0.1:19530"
COLLECTION_NAME = "globaltech_nlp_project"

# Test scenarios: Retrieve N candidates -> Rerank -> Take Top 20
CANDIDATE_SCENARIOS = [20, 50, 100, 200]
FINAL_TOP_K = 20

# --- SETUP ---
torch.cuda.empty_cache()
gc.collect()
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print(f"Device: {device.upper()}")
print("Loading Models...")
embed_model = SentenceTransformer(EMBED_MODEL, device=device)
# Keep max_length=256 to avoid CUDA assertion errors with PhoRanker
reranker = CrossEncoder(RERANKER_MODEL, max_length=256, device=device)

print("Connecting to Milvus...")
client = MilvusClient(uri=MILVUS_URI)
client.load_collection(COLLECTION_NAME)

# --- CORE FUNCTION ---
def search_pipeline(query_text, candidate_limit):
    t0 = time.time()
    
    # 1. Retrieval
    query_vector = embed_model.encode([query_text])
    
    search_res = client.search(
        collection_name=COLLECTION_NAME,
        data=query_vector,
        limit=candidate_limit,
        search_params={"metric_type": "COSINE", "params": {"nprobe": 64}},
        output_fields=["title", "text", "original_id", "topic"]
    )
    hits = search_res[0]
    
    if not hits: return [], time.time() - t0

    # 2. Rerank (Only if we have candidates)
    # If candidate_limit is small (e.g. 20), reranking is fast.
    # If large (e.g. 200), reranking takes longer.
    cross_inp = [[query_text, hit['entity']['text']] for hit in hits]
    
    try:
        cross_scores = reranker.predict(cross_inp)
        for idx, hit in enumerate(hits):
            hit['cross_score'] = cross_scores[idx]
            
        # 3. Sort & Slice Top 20
        reranked_hits = sorted(hits, key=lambda x: x['cross_score'], reverse=True)
        final_hits = reranked_hits[:FINAL_TOP_K]
        
    except Exception as e:
        print(f"Rerank error: {e}")
        return [], time.time() - t0
    
    return final_hits, time.time() - t0

# --- BENCHMARK LOOP ---
def run_sensitivity_test():
    try:
        with open(INPUT_TEST_FILE, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
    except Exception as e:
        print(f"Error loading inputs: {e}")
        return

    summary_data = []
    
    print(f"\nSTARTING SENSITIVITY TEST (Scenarios: {CANDIDATE_SCENARIOS})")
    print("="*60)

    for limit in CANDIDATE_SCENARIOS:
        print(f"\n🔄 Testing Candidate Limit: {limit}")
        
        results = []
        total_time = 0
        
        for index, test_case in enumerate(test_cases):
            query = test_case['question']
            case_id = test_case.get('id', str(index))
            ground_truths = [str(gt['doc_id']) for gt in test_case.get('ground_truths', [])]
            
            print(f"   Processing {case_id}...", end='\r')
            
            hits, duration = search_pipeline(query, candidate_limit=limit)
            total_time += duration
            
            # Calculate Hits@20 for this specific query
            is_correct = False
            if hits:
                for hit in hits:
                    if str(hit['entity'].get('original_id', '')) in ground_truths:
                        is_correct = True
                        break
            
            results.append(is_correct)

        # Aggregate metrics for this scenario
        hit_count = sum(results)
        hit_rate = (hit_count / len(test_cases)) * 100
        avg_latency = (total_time / len(test_cases))
        
        print(f"\n   ✅ Finished Limit {limit}. Hit Rate: {hit_rate:.2f}%. Avg Time: {avg_latency:.4f}s")
        
        summary_data.append({
            "Candidate_Limit": limit,
            "Hits@20_Percentage": hit_rate,
            "Avg_Latency_Seconds": round(avg_latency, 4),
            "Avg_Latency_ms": round(avg_latency * 1000, 2)
        })

    # Save Summary Report
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(SUMMARY_FILE, index=False)
    
    print("\n" + "="*60)
    print("📊 FINAL SENSITIVITY REPORT")
    print("="*60)
    print(df_summary.to_string(index=False))
    print(f"\nReport saved to: {SUMMARY_FILE}")

if __name__ == "__main__":
    run_sensitivity_test()