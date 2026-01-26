import json
import time
import os
import pandas as pd
import torch
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 1. CẤU HÌNH
# ==========================================
EMBED_MODEL = "hiieu/halong_embedding"
RERANKER_MODEL = "itdainb/PhoRanker"
LLM_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct" # Model phân loại query

INPUT_FOLDER = 'inputs'
INPUT_TEST_FILE = os.path.join(INPUT_FOLDER, 'ground_truth_test.json')
OUTPUT_FOLDER = f'outputs/{EMBED_MODEL.replace("/", "_")}_{RERANKER_MODEL.replace("/", "_")}_with_topic_filter'
OUTPUT_CSV_FILE = os.path.join(OUTPUT_FOLDER, 'benchmark_details_top5.csv')

MILVUS_URI = "http://127.0.0.1:19530"
COLLECTION_NAME = "globaltech_news_labeled" # Collection mới có chứa trường topic
TOP_K_EXPORT = 5

# Danh sách Topic chuẩn trong DB 
VALID_TOPICS = [
    "thoi-su", "du-lich", "the-gioi", "kinh-doanh", "khoa-hoc", 
    "giai-tri", "the-thao", "phap-luat", "giao-duc", "suc-khoe", "doi-song",
    "Other",
]

# ==========================================
# 2. LOAD MODELS
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Device: {device.upper()}")

print("⏳ Loading Embedding Model...")
embed_model = SentenceTransformer(EMBED_MODEL, device=device)

print("⏳ Loading Reranker...")
reranker = CrossEncoder(RERANKER_MODEL, max_length=256, device=device)

print(f"⏳ Loading LLM Classifier ({LLM_MODEL_ID})...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_ID,
    torch_dtype="auto",
    device_map="auto"
)

print("⏳ Connecting to Milvus...")
client = MilvusClient(uri=MILVUS_URI)
client.load_collection(COLLECTION_NAME)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================
# 3. HÀM PHÂN LOẠI QUERY BẰNG LLM
# ==========================================
def classify_query(query):
    """
    Dùng Qwen để đoán topic của câu query.
    Trả về: Tên topic (str) hoặc None nếu không chắc chắn.
    """
    system_prompt = f"""Bạn là một trợ lý AI chuyên phân loại chủ đề tin tức.
Danh sách chủ đề hợp lệ: {', '.join(VALID_TOPICS)}.
Nhiệm vụ: Chỉ trả về đúng tên chủ đề thuộc danh sách trên mà câu hỏi đang đề cập đến. Không giải thích thêm. Nếu không chắc chắn hoặc không thuộc chủ đề nào, hãy trả về 'Other'."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Câu hỏi: \"{query}\"\nChủ đề:"}
    ]
    
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        generated_ids = llm_model.generate(
            **model_inputs,
            max_new_tokens=20, # Chỉ cần output ngắn
            temperature=0.1,   # Giảm sáng tạo để tăng độ chính xác
            do_sample=False    # Dùng Greedy decoding để ổn định kết quả
        )
        
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    predicted_topic = response.strip()
    
    # Kiểm tra xem output có nằm trong list topic chuẩn không
    if predicted_topic in VALID_TOPICS:
        return predicted_topic
    else:
        # Nếu LLM trả về lung tung hoặc 'Other', ta sẽ không filter
        return None

# ==========================================
# 4. HÀM SEARCH (CÓ FILTER)
# ==========================================
def search_and_return_top_k(query_text, k=5, candidate_limit=50):
    t0 = time.time()
    
    # BƯỚC 1: Phân loại Query
    predicted_topic = classify_query(query_text)
    
    # BƯỚC 2: Tạo Expression Filter
    # Nếu đoán được topic -> Chỉ tìm trong topic đó
    # Nếu không đoán được (None) -> Tìm trên toàn bộ DB
    search_filter = ""
    if predicted_topic:
        search_filter = f"topic == '{predicted_topic}'"
    
    # BƯỚC 3: Embed & Search Milvus
    query_vector = embed_model.encode([query_text])
    
    search_res = client.search(
        collection_name=COLLECTION_NAME,
        data=query_vector,
        limit=candidate_limit,
        filter=search_filter, # <--- ÁP DỤNG FILTER TẠI ĐÂY
        search_params={"metric_type": "COSINE", "params": {"nprobe": 64}},
        output_fields=["title", "text", "original_id", "topic"] # Lấy thêm field topic để debug
    )
    
    milvus_hits = search_res[0]
    
    # Fallback: Nếu filter quá chặt mà không ra kết quả nào, hãy thử tìm lại mà không filter
    if not milvus_hits and predicted_topic:
        # print(f"⚠️ Không tìm thấy trong topic '{predicted_topic}'. Đang tìm kiếm toàn cục...")
        search_res = client.search(
            collection_name=COLLECTION_NAME,
            data=query_vector,
            limit=candidate_limit,
            # Không truyền filter
            search_params={"metric_type": "COSINE", "params": {"nprobe": 64}},
            output_fields=["title", "text", "original_id", "topic"]
        )
        milvus_hits = search_res[0]

    if not milvus_hits:
        return [], (time.time() - t0), predicted_topic

    # BƯỚC 4: Rerank
    cross_inp = [[query_text, hit['entity']['text']] for hit in milvus_hits]
    cross_scores = reranker.predict(cross_inp)
    
    for idx, hit in enumerate(milvus_hits):
        hit['cross_score'] = cross_scores[idx]
        
    reranked_hits = sorted(milvus_hits, key=lambda x: x['cross_score'], reverse=True)
    final_hits = reranked_hits[:k]
    
    duration = time.time() - t0
    return final_hits, duration, predicted_topic

# ==========================================
# 5. RUN BENCHMARK
# ==========================================
def run_benchmark():
    print(f"🚀 Starting benchmark with LLM Topic Filter...")
    
    try:
        with open(INPUT_TEST_FILE, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file {INPUT_TEST_FILE}")
        return

    all_results = []
    
    for index, test_case in enumerate(test_cases):
        query = test_case['query']
        expected_id = str(test_case['doc_id'])
        case_id = test_case.get('id', index)
        
        print(f"Processing #{case_id}...", end='\r')
        
        # Gọi hàm search mới (nhận thêm predicted_topic để log)
        hits, duration, predicted_topic = search_and_return_top_k(query, k=TOP_K_EXPORT)
        
        if not hits:
            all_results.append({
                "test_id": case_id,
                "query": query,
                "predicted_topic": predicted_topic, # Log xem LLM đoán gì
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
                "predicted_topic": predicted_topic, # Log topic dự đoán
                "doc_topic": hit['entity'].get('topic', ''), # Log topic thực tế của bài tìm được
                "expected_id": expected_id,
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
    print(f"✅ Đã lưu kết quả tại: {OUTPUT_CSV_FILE}")
    print(f"🎯 Hit Rate @ {TOP_K_EXPORT}: {accuracy:.2f}%")
    print(f"⏱  Thời gian TB/query: {df['process_time'].mean():.4f}s")
    print("="*50)

if __name__ == "__main__":
    run_benchmark()