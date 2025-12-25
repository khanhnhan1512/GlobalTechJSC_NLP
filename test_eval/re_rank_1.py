import json
import time
import os
import pandas as pd
import torch
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker 

EMBED_MODEL = "hiieu/halong_embedding"
RERANKER_MODEL = "namdp-ptit/ViRanker"

INPUT_FOLDER = 'inputs'
INPUT_TEST_FILE = os.path.join(INPUT_FOLDER, 'ground_truth_test.json')

OUTPUT_FOLDER = f'outputs/{EMBED_MODEL.replace("/", "_")}_{RERANKER_MODEL.replace("/", "_")}'
OUTPUT_CSV_FILE = os.path.join(OUTPUT_FOLDER, 'benchmark_details_top5.csv')

MILVUS_URI = "http://127.0.0.1:19530"
COLLECTION_NAME = "globaltech_nlp_project"
TOP_K_EXPORT = 5 

print("Loading Models & DB...")
# 1. Load Embedding Model
embed_model = SentenceTransformer(EMBED_MODEL)

# 2. Load Reranker Model (FlagEmbedding)
print(f"Loading FlagReranker: {RERANKER_MODEL}...")
reranker = FlagReranker(RERANKER_MODEL, use_fp16=True) 

# 3. Kết nối Milvus
client = MilvusClient(uri=MILVUS_URI)
client.load_collection(COLLECTION_NAME)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print(f"Ready. Output will be saved to: {OUTPUT_FOLDER}")


def search_and_return_top_k(query_text, k=5, candidate_limit=50):
    t0 = time.time()
    
    query_vector = embed_model.encode([query_text])
    
    search_res = client.search(
        collection_name=COLLECTION_NAME,
        data=query_vector,
        limit=candidate_limit,
        search_params={"metric_type": "COSINE", "params": {"nprobe": 64}},
        output_fields=["title", "text", "original_id"] 
    )
    
    milvus_hits = search_res[0]
    if not milvus_hits:
        return [], (time.time() - t0)


    cross_inp = [[query_text, hit['entity']['text']] for hit in milvus_hits]

    cross_scores = reranker.compute_score(cross_inp, normalize=True) 
    
    for idx, hit in enumerate(milvus_hits):
        hit['cross_score'] = cross_scores[idx]
        
    reranked_hits = sorted(milvus_hits, key=lambda x: x['cross_score'], reverse=True)
    final_hits = reranked_hits[:k]
    
    duration = time.time() - t0
    return final_hits, duration

def run_benchmark():
    print(f"Starting benchmark (Exporting Top {TOP_K_EXPORT})...")
    
    try:
        with open(INPUT_TEST_FILE, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
    except FileNotFoundError:
        print(f"Không tìm thấy file {INPUT_TEST_FILE}")
        return

    all_results = []
    
    for index, test_case in enumerate(test_cases):
        query = test_case['query']
        expected_id = str(test_case['doc_id'])
        case_id = test_case.get('id', index)
        
        print(f"Processing #{case_id}...", end='\r')
        
        # Lấy Top K kết quả
        hits, duration = search_and_return_top_k(query, k=TOP_K_EXPORT)
        
        if not hits:
            all_results.append({
                "test_id": case_id,
                "query": query,
                "rank": 0,
                "process_time": round(duration, 4),
                "retrieved_id": "NOT_FOUND",
                "is_correct": False,
                "score": 0,
                "expected_id": expected_id
            })
            continue

        found_in_top_k = False
        for rank, hit in enumerate(hits):
            retrieved_id = str(hit['entity'].get('original_id', ''))
            
            is_match = (retrieved_id == expected_id)
            if is_match: found_in_top_k = True

            row = {
                "test_id": case_id,
                "query": query,
                "expected_id": expected_id,
                "rank": rank + 1,
                "process_time": round(duration, 4),
                "score": round(float(hit['cross_score']), 6), 
                "is_correct": is_match,
                "retrieved_id": retrieved_id,
                "retrieved_title": hit['entity'].get('title', ''),
                "snippet": hit['entity'].get('text', '')[:200]
            }
            all_results.append(row)

    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8-sig')
    
    if not df.empty and 'is_correct' in df.columns:
        correct_queries = df.groupby('test_id')['is_correct'].any().sum()
        accuracy = (correct_queries / len(test_cases)) * 100
        avg_time = df['process_time'].mean()
    else:
        accuracy = 0
        avg_time = 0
    
    print("\n" + "="*50)
    print(f"✅ Đã hoàn tất benchmark với model: {RERANKER_MODEL}")
    print(f"📂 Kết quả lưu tại: {OUTPUT_CSV_FILE}")
    print(f"🎯 Hit Rate @ {TOP_K_EXPORT}: {accuracy:.2f}%")
    print(f"⏱  Thời gian trung bình: {avg_time:.4f}s")
    print("="*50)

if __name__ == "__main__":
    run_benchmark()