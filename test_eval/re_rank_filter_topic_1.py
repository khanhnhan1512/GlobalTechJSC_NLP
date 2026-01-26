# import json
# import time
# import os
# import pandas as pd
# import torch
# from pymilvus import MilvusClient
# from sentence_transformers import SentenceTransformer, CrossEncoder
# from transformers import AutoModelForCausalLM, AutoTokenizer

# # 1. CONFIGURATION
# EMBED_MODEL = "hiieu/halong_embedding"
# RERANKER_MODEL = "itdainb/PhoRanker"
# LLM_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# INPUT_FOLDER = 'inputs'
# INPUT_TEST_FILE = os.path.join(INPUT_FOLDER, 'gold_standard_dataset.json') # Updated filename
# OUTPUT_FOLDER = f'outputs/{EMBED_MODEL.replace("/", "_")}_{RERANKER_MODEL.replace("/", "_")}_with_topic_filter_gold_standard_dataset'
# OUTPUT_CSV_FILE = os.path.join(OUTPUT_FOLDER, 'benchmark_details_top5.csv')

# MILVUS_URI = "http://127.0.0.1:19530"
# COLLECTION_NAME = "globaltech_news_labeled"
# TOP_K_EXPORT = 5

# VALID_TOPICS = [
#     "thoi-su", "du-lich", "the-gioi", "kinh-doanh", "khoa-hoc", 
#     "giai-tri", "the-thao", "phap-luat", "giao-duc", "suc-khoe", "doi-song",
#     "Other",
# ]

# # 2. LOAD MODELS
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Device: {device.upper()}")

# print("Loading Embedding Model...")
# embed_model = SentenceTransformer(EMBED_MODEL, device=device)

# print("Loading Reranker...")
# reranker = CrossEncoder(RERANKER_MODEL, max_length=256, device=device)

# print(f"Loading LLM Classifier ({LLM_MODEL_ID})...")
# tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
# llm_model = AutoModelForCausalLM.from_pretrained(
#     LLM_MODEL_ID,
#     torch_dtype="auto",
#     device_map="auto"
# )

# print("Connecting to Milvus...")
# client = MilvusClient(uri=MILVUS_URI)
# client.load_collection(COLLECTION_NAME)

# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# # 3. QUERY CLASSIFICATION
# def classify_query(query):
#     """
#     Classify query topic using Qwen.
#     Returns: Topic string or None.
#     """
#     system_prompt = f"""Bạn là một trợ lý AI chuyên phân loại chủ đề tin tức.
# Danh sách chủ đề hợp lệ: {', '.join(VALID_TOPICS)}.
# Nhiệm vụ: Chỉ trả về đúng tên chủ đề thuộc danh sách trên mà câu hỏi đang đề cập đến. Không giải thích thêm. Nếu không chắc chắn hoặc không thuộc chủ đề nào, hãy trả về 'Other'."""

#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": f"Câu hỏi: \"{query}\"\nChủ đề:"}
#     ]
    
#     text = tokenizer.apply_chat_template(
#         messages, 
#         tokenize=False, 
#         add_generation_prompt=True
#     )
    
#     model_inputs = tokenizer([text], return_tensors="pt").to(device)
    
#     with torch.no_grad():
#         generated_ids = llm_model.generate(
#             **model_inputs,
#             max_new_tokens=20,
#             temperature=0.1,
#             do_sample=False
#         )
        
#     generated_ids = [
#         output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#     ]
    
#     response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     predicted_topic = response.strip()
    
#     if predicted_topic in VALID_TOPICS:
#         return predicted_topic
#     else:
#         return None

# # 4. SEARCH FUNCTION
# def search_and_return_top_k(query_text, k=5, candidate_limit=50):
#     t0 = time.time()
    
#     # Step 1: Classify
#     predicted_topic = classify_query(query_text)
    
#     # Step 2: Build Filter
#     search_filter = ""
#     if predicted_topic:
#         search_filter = f"topic == '{predicted_topic}'"
    
#     # Step 3: Embed & Search
#     query_vector = embed_model.encode([query_text])
    
#     search_res = client.search(
#         collection_name=COLLECTION_NAME,
#         data=query_vector,
#         limit=candidate_limit,
#         filter=search_filter,
#         search_params={"metric_type": "COSINE", "params": {"nprobe": 64}},
#         output_fields=["title", "text", "original_id", "topic"]
#     )
    
#     milvus_hits = search_res[0]
    
#     # Fallback: Search globally if filter yields no results
#     if not milvus_hits and predicted_topic:
#         search_res = client.search(
#             collection_name=COLLECTION_NAME,
#             data=query_vector,
#             limit=candidate_limit,
#             search_params={"metric_type": "COSINE", "params": {"nprobe": 64}},
#             output_fields=["title", "text", "original_id", "topic"]
#         )
#         milvus_hits = search_res[0]

#     if not milvus_hits:
#         return [], (time.time() - t0), predicted_topic

#     # Step 4: Rerank
#     cross_inp = [[query_text, hit['entity']['text']] for hit in milvus_hits]
#     cross_scores = reranker.predict(cross_inp)
    
#     for idx, hit in enumerate(milvus_hits):
#         hit['cross_score'] = cross_scores[idx]
        
#     reranked_hits = sorted(milvus_hits, key=lambda x: x['cross_score'], reverse=True)
#     final_hits = reranked_hits[:k]
    
#     duration = time.time() - t0
#     return final_hits, duration, predicted_topic

# # 5. BENCHMARK EXECUTION
# def run_benchmark():
#     print(f"Starting benchmark with LLM Topic Filter...")
    
#     try:
#         with open(INPUT_TEST_FILE, 'r', encoding='utf-8') as f:
#             test_cases = json.load(f)
#     except FileNotFoundError:
#         print(f"Error: File not found {INPUT_TEST_FILE}")
#         return

#     all_results = []
    
#     for index, test_case in enumerate(test_cases):
#         query = test_case['question'] # Updated key from 'query' to 'question' based on new JSON
#         case_id = test_case.get('id', str(index))
        
#         # Extract ground truth IDs
#         ground_truths = [str(gt['doc_id']) for gt in test_case.get('ground_truths', [])]
        
#         print(f"Processing #{case_id}...", end='\r')
        
#         hits, duration, predicted_topic = search_and_return_top_k(query, k=TOP_K_EXPORT)
        
#         if not hits:
#             all_results.append({
#                 "test_id": case_id,
#                 "query": query,
#                 "predicted_topic": predicted_topic,
#                 "rank": 0,
#                 "process_time": round(duration, 4), 
#                 "retrieved_id": "NOT_FOUND",
#                 "is_correct": False,
#                 "score": 0,
#                 "total_ground_truths": len(ground_truths)
#             })
#             continue

#         for rank, hit in enumerate(hits):
#             retrieved_id = str(hit['entity'].get('original_id', ''))
            
#             # Check if retrieved ID is in the list of valid ground truths
#             is_match = retrieved_id in ground_truths

#             row = {
#                 "test_id": case_id,
#                 "query": query,
#                 "predicted_topic": predicted_topic,
#                 "doc_topic": hit['entity'].get('topic', ''),
                
#                 "rank": rank + 1,                    
#                 "process_time": round(duration, 4), 
#                 "score": round(float(hit['cross_score']), 4),
                
#                 # Evaluation Metrics Data
#                 "is_correct": is_match,
#                 "total_ground_truths": len(ground_truths),
                
#                 "retrieved_id": retrieved_id,
#                 "retrieved_title": hit['entity'].get('title', ''),
#                 "snippet": hit['entity'].get('text', '')[:200]
#             }
#             all_results.append(row)

#     df = pd.DataFrame(all_results)
#     df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8-sig')
    
#     # Quick Check: Hit Rate (Any correct doc found)
#     correct_queries = df.groupby('test_id')['is_correct'].any().sum()
#     accuracy = (correct_queries / len(test_cases)) * 100
    
#     print("\n" + "="*50)
#     print(f"Results saved to: {OUTPUT_CSV_FILE}")
#     print(f"Hit Rate @ {TOP_K_EXPORT}: {accuracy:.2f}%")
#     print(f"Avg Time/Query: {df['process_time'].mean():.4f}s")
#     print("="*50)

# if __name__ == "__main__":
#     run_benchmark()


import json
import time
import os
import pandas as pd
import torch
import gc
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 1. CONFIGURATION
# ==========================================
EMBED_MODEL = "hiieu/halong_embedding"
RERANKER_MODEL = "itdainb/PhoRanker"
LLM_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

INPUT_FOLDER = 'inputs'
INPUT_TEST_FILE = os.path.join(INPUT_FOLDER, 'gold_standard_dataset.json')
OUTPUT_FOLDER = f'outputs/mode3_full_pipeline_top20' # Changed folder name
OUTPUT_CSV_FILE = os.path.join(OUTPUT_FOLDER, 'benchmark_results.csv')

MILVUS_URI = "http://127.0.0.1:19530"
COLLECTION_NAME = "globaltech_news_labeled"

# CRITICAL CHANGE: Set to 20 to allow Hits@20 calculation
TOP_K_EXPORT = 20  
CANDIDATE_LIMIT = 50 # Retrieval candidate pool size

VALID_TOPICS = [
    "thoi-su", "du-lich", "the-gioi", "kinh-doanh", "khoa-hoc", 
    "giai-tri", "the-thao", "phap-luat", "giao-duc", "suc-khoe", "doi-song",
    "Other",
]

# ==========================================
# 2. MEMORY CLEANUP
# ==========================================
torch.cuda.empty_cache()
gc.collect()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device.upper()}")

# ==========================================
# 3. LOAD MODELS (Optimized)
# ==========================================
print("Loading Embedding Model...")
embed_model = SentenceTransformer(EMBED_MODEL, device=device)

print("Loading Reranker...")
reranker = CrossEncoder(RERANKER_MODEL, max_length=256, device=device)

print(f"Loading LLM Classifier ({LLM_MODEL_ID})...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)

# Optimization: float16 + low_cpu_mem_usage to prevent OSError
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
    low_cpu_mem_usage=True
)

print("Connecting to Milvus...")
client = MilvusClient(uri=MILVUS_URI)
client.load_collection(COLLECTION_NAME)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================
# 4. QUERY CLASSIFICATION
# ==========================================
def classify_query(query):
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
    
    with torch.no_grad():
        generated_ids = llm_model.generate(
            **model_inputs,
            max_new_tokens=10,
            temperature=0.01,
            do_sample=False
        )
        
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    predicted_topic = response.split('\n')[-1].strip().lower().replace('.', '')
    
    if predicted_topic in VALID_TOPICS:
        return predicted_topic
    else:
        return None

# ==========================================
# 5. SEARCH FUNCTION (Full Pipeline)
# ==========================================
def search_and_return_top_k(query_text, k=20, candidate_limit=50):
    t0 = time.time()
    
    # Step 1: Classify
    predicted_topic = classify_query(query_text)
    
    # Step 2: Build Filter
    search_filter = ""
    if predicted_topic:
        search_filter = f"topic == '{predicted_topic}'"
    
    # Step 3: Embed & Search (Get candidates)
    query_vector = embed_model.encode([query_text])
    
    search_res = client.search(
        collection_name=COLLECTION_NAME,
        data=query_vector,
        limit=candidate_limit,
        filter=search_filter,
        search_params={"metric_type": "COSINE", "params": {"nprobe": 64}},
        output_fields=["title", "text", "original_id", "topic"]
    )
    
    milvus_hits = search_res[0]
    
    # Fallback: Search globally if filter yields no results
    if not milvus_hits and predicted_topic:
        search_res = client.search(
            collection_name=COLLECTION_NAME,
            data=query_vector,
            limit=candidate_limit,
            search_params={"metric_type": "COSINE", "params": {"nprobe": 64}},
            output_fields=["title", "text", "original_id", "topic"]
        )
        milvus_hits = search_res[0]

    if not milvus_hits:
        return [], (time.time() - t0), predicted_topic

    # Step 4: Rerank
    cross_inp = [[query_text, hit['entity']['text']] for hit in milvus_hits]
    cross_scores = reranker.predict(cross_inp)
    
    for idx, hit in enumerate(milvus_hits):
        hit['cross_score'] = cross_scores[idx]
        
    # Step 5: Sort and Slice Top K (20)
    reranked_hits = sorted(milvus_hits, key=lambda x: x['cross_score'], reverse=True)
    final_hits = reranked_hits[:k]
    
    duration = time.time() - t0
    return final_hits, duration, predicted_topic

# ==========================================
# 6. BENCHMARK EXECUTION
# ==========================================
def run_benchmark():
    print(f"Starting Mode 3: Full Pipeline (Filter + Rerank) Top {TOP_K_EXPORT}...")
    
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
        
        hits, duration, predicted_topic = search_and_return_top_k(query, k=TOP_K_EXPORT, candidate_limit=CANDIDATE_LIMIT)
        
        if not hits:
            all_results.append({
                "test_id": case_id,
                "query": query,
                "predicted_topic": predicted_topic,
                "rank": 0,
                "process_time": round(duration, 4), 
                "retrieved_id": "NOT_FOUND",
                "is_correct": False,
                "score": 0,
                "total_ground_truths": len(ground_truths)
            })
            continue

        for rank, hit in enumerate(hits):
            retrieved_id = str(hit['entity'].get('original_id', ''))
            is_match = retrieved_id in ground_truths

            row = {
                "test_id": case_id,
                "query": query,
                "predicted_topic": predicted_topic,
                "doc_topic": hit['entity'].get('topic', ''),
                
                "rank": rank + 1,                    
                "process_time": round(duration, 4), 
                "score": round(float(hit['cross_score']), 4),
                
                "is_correct": is_match,
                "total_ground_truths": len(ground_truths),
                
                "retrieved_id": retrieved_id,
                "retrieved_title": hit['entity'].get('title', ''),
                "snippet": hit['entity'].get('text', '')[:200]
            }
            all_results.append(row)

    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8-sig')
    
    # Quick Check: Hit Rate @ 20
    correct_queries = df.groupby('test_id')['is_correct'].any().sum()
    accuracy = (correct_queries / len(test_cases)) * 100
    
    print("\n" + "="*50)
    print(f"Results saved to: {OUTPUT_CSV_FILE}")
    print(f"Hit Rate @ {TOP_K_EXPORT}: {accuracy:.2f}%")
    print(f"Avg Time/Query: {df['process_time'].mean():.4f}s")
    print("="*50)

if __name__ == "__main__":
    run_benchmark()