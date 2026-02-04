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
# Bạn nhớ đổi tên file này thành tên file chứa dữ liệu mới của bạn (ví dụ: test_v2.jsonl)
INPUT_TEST_FILE = os.path.join(INPUT_FOLDER, 'gold_standard_200.jsonl') 

OUTPUT_FOLDER = f'outputs/sensitivity_test_scored_v2' 
SUMMARY_FILE = os.path.join(OUTPUT_FOLDER, 'sensitivity_report_v2.csv')

MILVUS_URI = "http://127.0.0.1:19530"
COLLECTION_NAME = "globaltech_nlp_project"

CANDIDATE_SCENARIOS = [20, 50, 100]
FINAL_TOP_K = 20

# --- SETUP ---
torch.cuda.empty_cache()
gc.collect()
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print(f"Device: {device.upper()}")
print("Loading Models...")
embed_model = SentenceTransformer(EMBED_MODEL, device=device)
reranker = CrossEncoder(RERANKER_MODEL, max_length=256, device=device)

print("Connecting to Milvus...")
client = MilvusClient(uri=MILVUS_URI)
client.load_collection(COLLECTION_NAME)

# --- HELPER: LOAD DATA ---
def load_test_data(file_path):
    data = []
    print(f"📂 Loading data from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Thử đọc toàn bộ file như JSON array
            try:
                content = f.read()
                data = json.loads(content)
            except json.JSONDecodeError:
                # Nếu lỗi, quay lại đầu file và đọc từng dòng (JSONL)
                f.seek(0)
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except:
                            pass 
        print(f"✅ Loaded {len(data)} test cases.")
        return data
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return []

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

    # 2. Rerank
    cross_inp = [[query_text, hit['entity']['text']] for hit in hits]
    
    try:
        cross_scores = reranker.predict(cross_inp)
        for idx, hit in enumerate(hits):
            hit['cross_score'] = float(cross_scores[idx])
            
        # 3. Sort & Slice Top 20
        reranked_hits = sorted(hits, key=lambda x: x['cross_score'], reverse=True)
        final_hits = reranked_hits[:FINAL_TOP_K]
        
    except Exception as e:
        print(f"Rerank error: {e}")
        return [], time.time() - t0
    
    return final_hits, time.time() - t0

# --- BENCHMARK LOOP ---
def run_sensitivity_test():
    test_cases = load_test_data(INPUT_TEST_FILE)
    if not test_cases: return

    summary_data = []
    
    print(f"\nSTARTING SENSITIVITY TEST (Scenarios: {CANDIDATE_SCENARIOS})")
    print("="*60)

    for limit in CANDIDATE_SCENARIOS:
        print(f"\n🔄 Testing Candidate Limit: {limit}")
        
        results = []
        total_time = 0
        valid_cases_count = 0
        
        submission_results = {} 
        
        for index, test_case in enumerate(test_cases):
            case_id = str(test_case.get('id', str(index)))
            
            # Lấy query (hỗ trợ cả key 'question' và 'query')
            query = test_case.get('question') or test_case.get('query')
            
            if not query:
                if limit == CANDIDATE_SCENARIOS[0]:
                    print(f"⚠️ Skipping Case #{case_id}: Missing 'question' field.")
                continue

            valid_cases_count += 1
            
            # --- XỬ LÝ GROUND TRUTHS (Cập nhật cho Test v2) ---
            ground_truths = []
            
            # Ưu tiên lấy key 'ground_truth_ids' (Format mới v2)
            if 'ground_truth_ids' in test_case:
                ground_truths = [str(x) for x in test_case['ground_truth_ids']]
            
            # Nếu không có thì fallback về key cũ 'ground_truths'
            elif 'ground_truths' in test_case:
                raw_gts = test_case['ground_truths']
                for gt in raw_gts:
                    if isinstance(gt, dict) and 'doc_id' in gt:
                        ground_truths.append(str(gt['doc_id']))
                    elif isinstance(gt, (str, int)):
                        ground_truths.append(str(gt))
            
            print(f"   Processing {case_id}...", end='\r')
            
            # --- RUN PIPELINE ---
            hits, duration = search_pipeline(query, candidate_limit=limit)
            total_time += duration
            
            # --- SAVE RESULTS WITH SCORES ---
            top_docs_with_scores = []
            top_doc_ids_only = [] 
            
            for hit in hits:
                doc_id = str(hit['entity'].get('original_id'))
                score = hit.get('cross_score', 0.0)
                
                top_docs_with_scores.append({
                    "doc_id": doc_id,
                    "score": score
                })
                top_doc_ids_only.append(doc_id)

            submission_results[case_id] = top_docs_with_scores
            
            # --- CHECK CORRECTNESS (Hits@20) ---
            is_correct = False
            if hits:
                # Kiểm tra xem có bất kỳ doc_id nào trong top 20 nằm trong ground_truths không
                # Sử dụng set intersection để kiểm tra nhanh hơn
                if set(top_doc_ids_only).intersection(set(ground_truths)):
                    is_correct = True
            
            results.append(is_correct)

        if valid_cases_count == 0:
            print("❌ No valid test cases found!")
            return

        # --- EXPORT JSON FILE ---
        output_json_path = os.path.join(OUTPUT_FOLDER, f'submission_results_limit_{limit}.json')
        with open(output_json_path, 'w', encoding='utf-8') as f_out:
            json.dump(submission_results, f_out, ensure_ascii=False, indent=4)
        print(f"\n   💾 Saved retrieval results to: {output_json_path}")

        # Aggregate metrics
        hit_count = sum(results)
        hit_rate = (hit_count / valid_cases_count) * 100
        avg_latency = (total_time / valid_cases_count)
        
        print(f"   ✅ Finished Limit {limit}. Hit Rate: {hit_rate:.2f}%. Avg Time: {avg_latency:.4f}s")
        
        summary_data.append({
            "Candidate_Limit": limit,
            "Hits@20_Percentage": hit_rate,
            "Avg_Latency_Seconds": round(avg_latency, 4),
            "Output_File": f'submission_results_limit_{limit}.json'
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