import json
import time
import os
import pandas as pd
import torch
import gc
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 1. CẤU HÌNH
# ==========================================
EMBED_MODEL = "hiieu/halong_embedding"
LLM_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

INPUT_FOLDER = 'inputs'
INPUT_TEST_FILE = os.path.join(INPUT_FOLDER, 'gold_standard_dataset.json')
OUTPUT_FOLDER = f'outputs/mode2_filter_only'
OUTPUT_CSV_FILE = os.path.join(OUTPUT_FOLDER, 'benchmark_results.csv')

MILVUS_URI = "http://127.0.0.1:19530"
COLLECTION_NAME = "globaltech_news_labeled"
TOP_K_SEARCH = 20  

VALID_TOPICS = [
    "thoi-su", "du-lich", "the-gioi", "kinh-doanh", "khoa-hoc", 
    "giai-tri", "the-thao", "phap-luat", "giao-duc", "suc-khoe", "doi-song",
    "Other",
]

# ==========================================
# 2. DỌN DẸP BỘ NHỚ TRƯỚC KHI CHẠY
# ==========================================
# Giúp giải phóng VRAM/RAM từ các lần chạy trước
torch.cuda.empty_cache()
gc.collect()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Device: {device.upper()}")

# ==========================================
# 3. LOAD MODELS (TỐI ƯU CHO WINDOWS)
# ==========================================
print("⏳ Loading Embedding Model...")
embed_model = SentenceTransformer(EMBED_MODEL, device=device)

print(f"⏳ Loading LLM Classifier ({LLM_MODEL_ID})...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_ID,
    torch_dtype="auto", 
    device_map="auto",
    low_cpu_mem_usage=True 
)

print("⏳ Connecting to Milvus...")
client = MilvusClient(uri=MILVUS_URI)
client.load_collection(COLLECTION_NAME)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================
def classify_query(query):
    """
    Classify query topic using Qwen.
    """
    system_prompt = f"""Bạn là một trợ lý AI chuyên phân loại chủ đề tin tức.
    Danh sách chủ đề hợp lệ: {', '.join(VALID_TOPICS)}.
    Nhiệm vụ: Chỉ trả về đúng tên chủ đề thuộc danh sách trên mà câu hỏi đang đề cập đến. Không giải thích thêm. Nếu không chắc chắn hoặc không thuộc chủ đề nào, hãy trả về 'Other'."""
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Câu hỏi: \"{query}\"\nChủ đề:"}]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated_ids = llm_model.generate(**model_inputs, max_new_tokens=10, temperature=0.01, do_sample=False)
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # Simple cleaning
    predicted_topic = response.split('\n')[-1].strip().lower().replace('.', '')
    
    if predicted_topic in VALID_TOPICS: return predicted_topic
    return None

def search_filter_only(query_text, k=20):
    t0 = time.time()
    
    # 1. Classify
    predicted_topic = classify_query(query_text)
    
    # 2. Filter
    search_filter = f"topic == '{predicted_topic}'" if predicted_topic else ""
    
    # 3. Retrieval
    query_vector = embed_model.encode([query_text])
    
    search_res = client.search(
        collection_name=COLLECTION_NAME,
        data=query_vector,
        limit=k,
        filter=search_filter, 
        search_params={"metric_type": "COSINE", "params": {"nprobe": 64}},
        output_fields=["title", "text", "original_id", "topic"]
    )
    hits = search_res[0]

    # Fallback: Nếu tìm theo topic không ra gì, tìm lại toàn cục
    if not hits and predicted_topic:
        search_res = client.search(
            collection_name=COLLECTION_NAME, data=query_vector, limit=k,
            search_params={"metric_type": "COSINE", "params": {"nprobe": 64}},
            output_fields=["title", "text", "original_id", "topic"]
        )
        hits = search_res[0]
        
    return hits, time.time() - t0, predicted_topic

# ==========================================
# 5. BENCHMARK EXECUTION
# ==========================================
def run_benchmark():
    print(f"🚀 Starting Mode 2: FILTER ONLY (Top {TOP_K_SEARCH})...")
    
    try:
        with open(INPUT_TEST_FILE, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found {INPUT_TEST_FILE}")
        return

    all_results = []
    
    for index, test_case in enumerate(test_cases):
        query = test_case['question']
        case_id = test_case.get('id', str(index))
        ground_truths = [str(gt['doc_id']) for gt in test_case.get('ground_truths', [])]
        
        print(f"Processing #{case_id}...", end='\r')
        
        hits, duration, predicted_topic = search_filter_only(query, k=TOP_K_SEARCH)
        
        if not hits:
            all_results.append({
                "test_id": case_id, "query": query, "predicted_topic": predicted_topic,
                "rank": 0, "process_time": duration, "is_correct": False, "total_ground_truths": len(ground_truths)
            })
            continue

        for rank, hit in enumerate(hits):
            retrieved_id = str(hit['entity'].get('original_id', ''))
            is_match = retrieved_id in ground_truths

            all_results.append({
                "test_id": case_id, "query": query, "predicted_topic": predicted_topic,
                "doc_topic": hit['entity'].get('topic', ''),
                "rank": rank + 1, "process_time": duration, "score": hit['distance'], # Cosine score
                "is_correct": is_match, "total_ground_truths": len(ground_truths),
                "retrieved_id": retrieved_id
            })

    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8-sig')
    
    # Quick Check
    correct_queries = df.groupby('test_id')['is_correct'].any().sum()
    accuracy = (correct_queries / len(test_cases)) * 100
    
    print("\n" + "="*50)
    print(f"Results saved to: {OUTPUT_CSV_FILE}")
    print(f"Hit Rate @ {TOP_K_SEARCH}: {accuracy:.2f}%")
    print(f"Avg Time/Query: {df['process_time'].mean():.4f}s")
    print("="*50)

if __name__ == "__main__":
    run_benchmark()