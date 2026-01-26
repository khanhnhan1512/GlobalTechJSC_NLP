import pandas as pd
import os
import numpy as np

# Configuration
EMBED_MODEL = "hiieu/halong_embedding"
RERANKER_MODEL = "itdainb/PhoRanker"

INPUT_CSV_FILE = f'outputs/{EMBED_MODEL.replace("/", "_")}_{RERANKER_MODEL.replace("/", "_")}_with_topic_filter_gold_standard_dataset/benchmark_details_top5.csv'
OUTPUT_METRICS_FILE = f'outputs/{EMBED_MODEL.replace("/", "_")}_{RERANKER_MODEL.replace("/", "_")}_with_topic_filter_gold_standard_dataset/evaluation_metrics_report.csv'

def calculate_metrics(input_path, output_path):
    print(f"Reading data from: {input_path}")
    
    if not os.path.exists(input_path):
        print("Error: Input file not found.")
        return

    df = pd.read_csv(input_path)
    
    # Ensure boolean type
    if df['is_correct'].dtype == object:
        df['is_correct'] = df['is_correct'].apply(lambda x: str(x).lower() == 'true')

    grouped = df.groupby('test_id')
    total_queries = len(grouped)
    
    # Metrics containers
    reciprocal_ranks = []
    hits_at_1_count = 0
    hits_at_5_count = 0
    
    recall_at_1_list = []
    recall_at_5_list = []

    print(f"Calculating metrics for {total_queries} queries...")

    for test_id, group in grouped:
        group = group.sort_values('rank')
        
        # Get total relevant docs for this query (stored in every row of the group)
        total_relevant = group.iloc[0]['total_ground_truths']
        if total_relevant == 0: total_relevant = 1 # Prevent division by zero if data is malformed
        
        # Filter correct predictions
        correct_rows = group[group['is_correct'] == True]
        
        # --- MRR Calculation ---
        if not correct_rows.empty:
            first_correct_rank = correct_rows.iloc[0]['rank']
            reciprocal_ranks.append(1.0 / first_correct_rank)
        else:
            reciprocal_ranks.append(0.0)

        # --- Hits@K (Binary: Found at least one?) ---
        # Top 1
        if not correct_rows[correct_rows['rank'] == 1].empty:
            hits_at_1_count += 1
            
        # Top 5
        if not correct_rows[correct_rows['rank'] <= 5].empty:
            hits_at_5_count += 1

        # --- Recall@K (Count of correct / Total relevant) ---
        # Recall @ 1
        correct_at_1 = len(correct_rows[correct_rows['rank'] == 1])
        recall_at_1_list.append(correct_at_1 / total_relevant)

        # Recall @ 5
        correct_at_5 = len(correct_rows[correct_rows['rank'] <= 5])
        recall_at_5_list.append(correct_at_5 / total_relevant)

    # Aggregation
    mrr = np.mean(reciprocal_ranks)
    hit_rate_1 = hits_at_1_count / total_queries
    hit_rate_5 = hits_at_5_count / total_queries
    avg_recall_1 = np.mean(recall_at_1_list)
    avg_recall_5 = np.mean(recall_at_5_list)

    # Save Report
    metrics_data = {
        "Metric": ["Total Queries", "MRR", "Hits@1", "Hits@5", "Recall@1", "Recall@5"],
        "Value": [
            total_queries, 
            round(mrr, 4), 
            round(hit_rate_1, 4), 
            round(hit_rate_5, 4),
            round(avg_recall_1, 4),
            round(avg_recall_5, 4)
        ],
        "Description": [
            "Total number of queries",
            "Mean Reciprocal Rank",
            "Percentage of queries with at least one correct result at Rank 1",
            "Percentage of queries with at least one correct result in Top 5",
            "Average Recall at Rank 1 (Correct Retrieved @ 1 / Total Relevant)",
            "Average Recall at Rank 5 (Correct Retrieved @ 5 / Total Relevant)"
        ]
    }
    
    result_df = pd.DataFrame(metrics_data)
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"MRR:        {mrr:.4f}")
    print(f"Hits@1:     {hit_rate_1:.2%}")
    print(f"Hits@5:     {hit_rate_5:.2%}")
    print(f"Recall@1:   {avg_recall_1:.4f}")
    print(f"Recall@5:   {avg_recall_5:.4f}")
    print(f"Report saved to: {output_path}")
    print("="*50)

if __name__ == "__main__":
    calculate_metrics(INPUT_CSV_FILE, OUTPUT_METRICS_FILE)