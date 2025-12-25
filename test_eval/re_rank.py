import json
import time
import os
import pandas as pd
import torch
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer, CrossEncoder


EMBED_MODEL = "hiieu/halong_embedding"
RERANKER_MODEL = "itdainb/PhoRanker"
# RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

INPUT_FOLDER = 'inputs'
INPUT_TEST_FILE = os.path.join(INPUT_FOLDER,'ground_truth_test.json')
OUTPUT_FOLDER = f'outputs/{EMBED_MODEL.replace("/", "_")}_{RERANKER_MODEL.replace("/", "_")}'
OUTPUT_CSV_FILE = os.path.join(OUTPUT_FOLDER, 'benchmark_details_top5.csv')

MILVUS_URI = "http://127.0.0.1:19530"
COLLECTION_NAME = "globaltech_nlp_project"
TOP_K_EXPORT = 5


print("Loading Models & DB...")
embed_model = SentenceTransformer(EMBED_MODEL)
device = "cuda" if torch.cuda.is_available() else "cpu"
reranker = CrossEncoder(RERANKER_MODEL, max_length=256, device=device)


client = MilvusClient(uri=MILVUS_URI)
client.load_collection(COLLECTION_NAME)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def search_and_return_top_k(query_text, k=5, candidate_limit=50):
    t0 = time.time()
    
    # 1. Embed & Search Milvus
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

    # 2. Rerank
    cross_inp = [[query_text, hit['entity']['text']] for hit in milvus_hits]
    cross_scores = reranker.predict(cross_inp)
    
    for idx, hit in enumerate(milvus_hits):
        hit['cross_score'] = cross_scores[idx]
        
    # 3. Sort & Slice Top K
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
        
        hits, duration = search_and_return_top_k(query, k=TOP_K_EXPORT)
        
        if not hits:
            all_results.append({
                "test_id": case_id,
                "query": query,
                "rank": 0,
                "process_time": round(duration, 4), 
                "retrieved_id": "NOT_FOUND",
                "is_correct": False,
                "score": 0
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
                
                # CÁC CỘT QUAN TRỌNG
                "rank": rank + 1,                    
                "process_time": round(duration, 4), 
                "score": round(float(hit['cross_score']), 4),
                "is_correct": is_match,              
                
                "retrieved_id": retrieved_id,
                "retrieved_title": hit['entity'].get('title', ''),
                "snippet": hit['entity'].get('text', '')[:200]
            }
            all_results.append(row)

    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8-sig')
    
    correct_queries = df.groupby('test_id')['is_correct'].any().sum()
    accuracy = (correct_queries / len(test_cases)) * 100
    
    print("\n" + "="*50)
    print(f"✅ Đã lưu kết quả chi tiết vào: {OUTPUT_CSV_FILE}")
    print(f"🎯 Hit Rate @ {TOP_K_EXPORT} (Tỉ lệ tìm thấy trong top {TOP_K_EXPORT}): {accuracy:.2f}%")
    print(f"⏱  Thời gian trung bình/query: {df['process_time'].mean():.4f}s")
    print("="*50)

if __name__ == "__main__":
    run_benchmark()