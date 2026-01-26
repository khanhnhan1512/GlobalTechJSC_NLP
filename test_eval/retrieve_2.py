import json
import time
import os
import pandas as pd
import torch
import gc
import logging
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer, CrossEncoder
# Import thư viện logging của transformers để tắt cảnh báo
from transformers import logging as transformers_logging

# --- FIX: TẮT CẢNH BÁO ---
transformers_logging.set_verbosity_error() # Chỉ hiện lỗi nghiêm trọng, ẩn cảnh báo truncation

# ==========================================
# CẤU HÌNH
# ==========================================
EMBED_MODEL = "hiieu/halong_embedding"
RERANKER_MODEL = "itdainb/PhoRanker"

INPUT_FOLDER = 'inputs'
INPUT_TEST_FILE = os.path.join(INPUT_FOLDER, 'gold_standard_dataset.json')
OUTPUT_FOLDER = f'outputs/mode4_rerank_only'
OUTPUT_CSV_FILE = os.path.join(OUTPUT_FOLDER, 'benchmark_results.csv')

MILVUS_URI = "http://127.0.0.1:19530"
COLLECTION_NAME = "globaltech_news_labeled"

TOP_K_EXPORT = 20    
CANDIDATE_LIMIT = 100 

# ==========================================
# MEMORY CLEANUP
# ==========================================
torch.cuda.empty_cache()
gc.collect()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device.upper()}")

# ==========================================
# LOAD MODELS
# ==========================================
print("Loading Embedding Model...")
embed_model = SentenceTransformer(EMBED_MODEL, device=device)

print("Loading Reranker...")
# --- FIX QUAN TRỌNG: QUAY VỀ 256 ĐỂ TRÁNH CRASH ---
# PhoRanker chỉ hỗ trợ tối đa 256 token.
reranker = CrossEncoder(RERANKER_MODEL, max_length=256, device=device)

print("Connecting to Milvus...")
client = MilvusClient(uri=MILVUS_URI)
client.load_collection(COLLECTION_NAME)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================
# SEARCH FUNCTION (NO FILTER -> RERANK)
# ==========================================
def search_rerank_only(query_text, k=20, candidate_limit=100):
    t0 = time.time()
    
    # 1. Embed & Search (Raw Retrieval)
    query_vector = embed_model.encode([query_text])
    
    # Lấy Top 100 bài thô sơ nhất (Candidate Generation)
    search_res = client.search(
        collection_name=COLLECTION_NAME,
        data=query_vector,
        limit=candidate_limit,
        search_params={"metric_type": "COSINE", "params": {"nprobe": 64}},
        output_fields=["title", "text", "original_id", "topic"]
    )
    
    hits = search_res[0]
    if not hits: return [], time.time() - t0

    # 2. Rerank
    # CrossEncoder sẽ tự động cắt (truncate) những bài dài hơn 256 token
    # Nhờ dòng set_verbosity_error() ở trên, nó sẽ cắt âm thầm mà không báo lỗi
    cross_inp = [[query_text, hit['entity']['text']] for hit in hits]
    
    try:
        cross_scores = reranker.predict(cross_inp)
    except Exception as e:
        print(f"\n⚠️ Error during reranking: {e}")
        return [], time.time() - t0
    
    for idx, hit in enumerate(hits):
        hit['cross_score'] = cross_scores[idx]
        
    # 3. Sort & Slice
    reranked_hits = sorted(hits, key=lambda x: x['cross_score'], reverse=True)
    final_hits = reranked_hits[:k]
    
    return final_hits, time.time() - t0

# ==========================================
# BENCHMARK LOOP
# ==========================================
def run_benchmark():
    print(f"🚀 Starting Mode 4: RERANK ONLY (Candidates: {CANDIDATE_LIMIT} -> Top {TOP_K_EXPORT})...")
    
    try:
        with open(INPUT_TEST_FILE, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
    except FileNotFoundError: return

    all_results = []
    
    for index, test_case in enumerate(test_cases):
        query = test_case['question']
        case_id = test_case.get('id', str(index))
        ground_truths = [str(gt['doc_id']) for gt in test_case.get('ground_truths', [])]
        
        print(f"Processing #{case_id}...", end='\r')
        
        hits, duration = search_rerank_only(query, k=TOP_K_EXPORT, candidate_limit=CANDIDATE_LIMIT)
        
        if not hits:
            all_results.append({
                "test_id": case_id, "query": query, "rank": 0, 
                "process_time": duration, "is_correct": False, "total_ground_truths": len(ground_truths)
            })
            continue

        for rank, hit in enumerate(hits):
            retrieved_id = str(hit['entity'].get('original_id', ''))
            is_match = retrieved_id in ground_truths

            all_results.append({
                "test_id": case_id, "query": query,
                "doc_topic": hit['entity'].get('topic', ''),
                "rank": rank + 1, "process_time": duration, 
                "score": float(hit['cross_score']),
                "is_correct": is_match, "total_ground_truths": len(ground_truths),
                "retrieved_id": retrieved_id
            })

    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8-sig')
    
    correct_queries = df.groupby('test_id')['is_correct'].any().sum()
    accuracy = (correct_queries / len(test_cases)) * 100
    
    print("\n" + "="*50)
    print(f"Results saved to: {OUTPUT_CSV_FILE}")
    print(f"Hit Rate @ {TOP_K_EXPORT}: {accuracy:.2f}%")
    print("="*50)

if __name__ == "__main__":
    run_benchmark()