import pandas as pd
import os
import numpy as np

# Đường dẫn file output từ script trên
INPUT_CSV_FILE = 'outputs/milvus_raw_top20/benchmark_raw_results.csv'
OUTPUT_METRICS_FILE = 'outputs/milvus_raw_top20/evaluation_metrics_report.csv'

def calculate_metrics(input_path, output_path):
    print(f"Reading data from: {input_path}")
    
    if not os.path.exists(input_path):
        print("Error: Input file not found.")
        return

    df = pd.read_csv(input_path)
    
    if df['is_correct'].dtype == object:
        df['is_correct'] = df['is_correct'].apply(lambda x: str(x).lower() == 'true')

    grouped = df.groupby('test_id')
    total_queries = len(grouped)
    
    reciprocal_ranks = []
    
    # Các biến đếm Hits
    hits_1 = 0
    hits_5 = 0
    hits_10 = 0
    hits_20 = 0
    
    # Các biến list Recall
    recall_1_list = []
    recall_5_list = []
    recall_10_list = []
    recall_20_list = []

    print(f"Calculating metrics for {total_queries} queries...")

    for test_id, group in grouped:
        group = group.sort_values('rank')
        
        total_relevant = group.iloc[0]['total_ground_truths']
        if total_relevant == 0: total_relevant = 1
        
        correct_rows = group[group['is_correct'] == True]
        
        # --- MRR ---
        if not correct_rows.empty:
            reciprocal_ranks.append(1.0 / correct_rows.iloc[0]['rank'])
        else:
            reciprocal_ranks.append(0.0)

        # --- Hits@K ---
        if not correct_rows[correct_rows['rank'] == 1].empty: hits_1 += 1
        if not correct_rows[correct_rows['rank'] <= 5].empty: hits_5 += 1
        if not correct_rows[correct_rows['rank'] <= 10].empty: hits_10 += 1
        if not correct_rows[correct_rows['rank'] <= 20].empty: hits_20 += 1 # Quan tâm nhất cái này

        # --- Recall@K ---
        # Hàm helper tính recall
        def get_recall_at_k(k):
            count = len(correct_rows[correct_rows['rank'] <= k])
            return count / total_relevant

        recall_1_list.append(get_recall_at_k(1))
        recall_5_list.append(get_recall_at_k(5))
        recall_10_list.append(get_recall_at_k(10))
        recall_20_list.append(get_recall_at_k(20))

    # Aggregation
    metrics_data = {
        "Metric": [
            "Total Queries", 
            "MRR", 
            "Hits@1", "Hits@5", "Hits@10", "Hits@20", 
            "Recall@1", "Recall@5", "Recall@10", "Recall@20"
        ],
        "Value": [
            total_queries, 
            round(np.mean(reciprocal_ranks), 4),
            round(hits_1 / total_queries, 4),
            round(hits_5 / total_queries, 4),
            round(hits_10 / total_queries, 4),
            round(hits_20 / total_queries, 4), # Xem cái này cao không?
            round(np.mean(recall_1_list), 4),
            round(np.mean(recall_5_list), 4),
            round(np.mean(recall_10_list), 4),
            round(np.mean(recall_20_list), 4)  # Xem cái này cao không?
        ]
    }
    
    result_df = pd.DataFrame(metrics_data)
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*50)
    print("📊 RAW MILVUS EVALUATION (NO RERANK)")
    print("="*50)
    print(f"Hits@1:     {metrics_data['Value'][2]:.2%}")
    print(f"Hits@5:     {metrics_data['Value'][3]:.2%}")
    print(f"Hits@20:    {metrics_data['Value'][5]:.2%}  <-- Quan trọng nhất")
    print("-" * 30)
    print(f"Recall@5:   {metrics_data['Value'][7]:.4f}")
    print(f"Recall@20:  {metrics_data['Value'][9]:.4f}")
    print("="*50)

if __name__ == "__main__":
    calculate_metrics(INPUT_CSV_FILE, OUTPUT_METRICS_FILE)