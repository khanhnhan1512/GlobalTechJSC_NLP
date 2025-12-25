import pandas as pd
import os
import numpy as np


EMBED_MODEL = "hiieu/halong_embedding"
# RERANKER_MODEL = "itdainb/PhoRanker"
# RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANKER_MODEL = "namdp-ptit/ViRanker"

INPUT_CSV_FILE = f'outputs/{EMBED_MODEL.replace("/", "_")}_{RERANKER_MODEL.replace("/", "_")}/benchmark_details_top5.csv'
OUTPUT_METRICS_FILE = f'outputs/{EMBED_MODEL.replace("/", "_")}_{RERANKER_MODEL.replace("/", "_")}/evaluation_metrics_report.csv'

def calculate_metrics(input_path, output_path):
    print(f"Đang đọc dữ liệu từ: {input_path}")
    
    if not os.path.exists(input_path):
        print("Lỗi: Không tìm thấy file input. Hãy chạy benchmark trước!")
        return

    df = pd.read_csv(input_path)
    
    if df['is_correct'].dtype == object:
        df['is_correct'] = df['is_correct'].apply(lambda x: str(x).lower() == 'true')

    grouped = df.groupby('test_id')
    
    total_queries = len(grouped)
    reciprocal_ranks = []
    
    hits_at_1_count = 0
    hits_at_5_count = 0
    
    print(f"Đang tính toán trên {total_queries} câu hỏi...")

    for test_id, group in grouped:
        group = group.sort_values('rank')
        
        correct_row = group[group['is_correct'] == True]
        
        if not correct_row.empty:
            rank = correct_row.iloc[0]['rank']
            
            reciprocal_ranks.append(1.0 / rank)
            
            if rank == 1:
                hits_at_1_count += 1
            
            if rank <= 5:
                hits_at_5_count += 1
        else:
            reciprocal_ranks.append(0.0)

    mrr = np.mean(reciprocal_ranks)
    hit_rate_1 = hits_at_1_count / total_queries
    hit_rate_5 = hits_at_5_count / total_queries
    

    metrics_data = {
        "Metric": ["Total Queries", "MRR", "Hits@1", "Hits@5",],
        "Value": [total_queries, round(mrr, 4), round(hit_rate_1, 4), round(hit_rate_5, 4)],
        "Description": [
            "Tổng số câu hỏi kiểm thử",
            "Mean Reciprocal Rank (Thứ hạng trung bình nghịch đảo - Càng cao càng tốt, Max 1.0)",
            "Tỉ lệ đáp án đúng nằm ngay vị trí đầu tiên (Top 1)",
            "Tỉ lệ đáp án đúng nằm trong 5 kết quả đầu (Top 5)",
        ]
    }
    
    result_df = pd.DataFrame(metrics_data)
    
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*50)
    print("📊 KẾT QUẢ ĐÁNH GIÁ (EVALUATION)")
    print("="*50)
    print(f"🎯 MRR:       {mrr:.4f}")
    print(f"🥇 Hits@1:    {hit_rate_1:.2%}")
    print(f"🖐️ Hits@5:    {hit_rate_5:.2%}")
    print(f"💾 Đã lưu báo cáo chi tiết vào: {output_path}")
    print("="*50)

if __name__ == "__main__":
    calculate_metrics(INPUT_CSV_FILE, OUTPUT_METRICS_FILE)