import json
import time
import os
import pandas as pd
import torch
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. CẤU HÌNH
# ==========================================
EMBED_MODEL = "hiieu/halong_embedding"

INPUT_FOLDER = 'inputs'
INPUT_TEST_FILE = os.path.join(INPUT_FOLDER, 'gold_standard_dataset.json')

# Đặt tên folder output khác để phân biệt
OUTPUT_FOLDER = f'outputs/milvus_raw_top20' 
OUTPUT_CSV_FILE = os.path.join(OUTPUT_FOLDER, 'benchmark_raw_results.csv')

MILVUS_URI = "http://127.0.0.1:19530"
COLLECTION_NAME = "globaltech_news_labeled"

# QUAN TRỌNG: Lấy Top 20 để kiểm tra Recall
TOP_K_SEARCH = 20 

# ==========================================
# 2. LOAD MODELS
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Device: {device.upper()}")

print("⏳ Loading Embedding Model...")
embed_model = SentenceTransformer(EMBED_MODEL, device=device)

print("⏳ Connecting to Milvus...")
client = MilvusClient(uri=MILVUS_URI)
client.load_collection(COLLECTION_NAME)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================
# 3. HÀM SEARCH THUẦN (KHÔNG RERANK, KHÔNG FILTER)
# ==========================================
def search_raw(query_text, k=20):
    t0 = time.time()
    
    # 1. Embed Query
    query_vector = embed_model.encode([query_text])
    
    # 2. Search Milvus (Lấy thẳng Top 20)
    search_res = client.search(
        collection_name=COLLECTION_NAME,
        data=query_vector,
        limit=k,
        # Không dùng filter topic nữa để test Raw Search
        search_params={"metric_type": "COSINE", "params": {"nprobe": 64}},
        output_fields=["title", "text", "original_id", "topic"]
    )
    
    hits = search_res[0]
    duration = time.time() - t0
    
    return hits, duration

# ==========================================
# 4. CHẠY BENCHMARK
# ==========================================
def run_benchmark():
    print(f"🚀 Starting RAW benchmark (Top {TOP_K_SEARCH})...")
    
    try:
        with open(INPUT_TEST_FILE, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {INPUT_TEST_FILE}")
        return

    all_results = []
    
    for index, test_case in enumerate(test_cases):
        query = test_case['question']
        case_id = test_case.get('id', str(index))
        
        # Lấy danh sách ID đáp án đúng
        ground_truths = [str(gt['doc_id']) for gt in test_case.get('ground_truths', [])]
        
        print(f"Processing #{case_id}...", end='\r')
        
        hits, duration = search_raw(query, k=TOP_K_SEARCH)
        
        # Trường hợp không tìm thấy gì (hiếm khi xảy ra nếu không filter)
        if not hits:
            all_results.append({
                "test_id": case_id,
                "query": query,
                "rank": 0,
                "process_time": round(duration, 4), 
                "retrieved_id": "NOT_FOUND",
                "is_correct": False,
                "score": 0,
                "total_ground_truths": len(ground_truths)
            })
            continue

        # Ghi lại từng kết quả trong Top 20
        for rank, hit in enumerate(hits):
            retrieved_id = str(hit['entity'].get('original_id', ''))
            
            # Kiểm tra đúng sai
            is_match = retrieved_id in ground_truths

            row = {
                "test_id": case_id,
                "query": query,
                "doc_topic": hit['entity'].get('topic', ''),
                
                "rank": rank + 1,                    
                "process_time": round(duration, 4), 
                # Điểm số lúc này là Cosine Similarity từ Milvus
                "score": round(float(hit['distance']), 4), 
                
                "is_correct": is_match,
                "total_ground_truths": len(ground_truths),
                
                "retrieved_id": retrieved_id,
                "retrieved_title": hit['entity'].get('title', ''),
                "snippet": hit['entity'].get('text', '')[:200]
            }
            all_results.append(row)

    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8-sig')
    
    # Hit Rate sơ bộ @ 20
    correct_queries = df.groupby('test_id')['is_correct'].any().sum()
    accuracy = (correct_queries / len(test_cases)) * 100
    
    print("\n" + "="*50)
    print(f"✅ Đã lưu kết quả tại: {OUTPUT_CSV_FILE}")
    print(f"🎯 Raw Hits@{TOP_K_SEARCH}: {accuracy:.2f}% (Tỷ lệ tìm thấy ít nhất 1 bài đúng trong Top {TOP_K_SEARCH})")
    print(f"⏱  Thời gian TB/query: {df['process_time'].mean():.4f}s")
    print("="*50)

if __name__ == "__main__":
    run_benchmark()