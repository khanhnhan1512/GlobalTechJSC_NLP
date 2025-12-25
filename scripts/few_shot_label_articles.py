import os
import json
import ijson
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from wakepy import keep
from tqdm import tqdm

# ==========================================
CURRENT_SCRIPT_PATH = os.path.abspath(__file__) # .../Project/scripts/classify_news_data.py
SCRIPT_DIR = os.path.dirname(CURRENT_SCRIPT_PATH) # .../Project/scripts

PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) # .../Project

SEED_DATA_DIR = os.path.join(PROJECT_ROOT, 'inputs', 'crawl_data', 'results')

INPUT_FILE = os.path.join(PROJECT_ROOT, 'inputs', 'merged_news.json')

OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'inputs', 'merged_news_100k_labeled.json')

MODEL_NAME = "hiieu/halong_embedding"
BATCH_SIZE = 64        
LIMIT_ARTICLES = 10_0000 # Giới hạn số bài xử lý
CONFIDENCE_THRESHOLD = 0.25 

# ==========================================
def compute_topic_centroids(model, seed_dir):
    print(f"Đang đọc dữ liệu seed từ: {seed_dir}")
    
    if not os.path.exists(seed_dir):
        print(f"LỖI: Không tìm thấy thư mục seed tại: {seed_dir}")
        return None, None

    topic_centroids = {}
    topic_names = []

    try:
        subfolders = [f.path for f in os.scandir(seed_dir) if f.is_dir()]
    except Exception as e:
        print(f"Lỗi truy cập thư mục: {e}")
        return None, None

    print(f"Tìm thấy {len(subfolders)} chủ đề.")

    for folder in subfolders:
        topic_name = os.path.basename(folder)
        texts = []
        
        # Đọc file txt trong folder
        files = [f for f in os.listdir(folder) if f.endswith('.txt')]
        for file_name in files:
            file_path = os.path.join(folder, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content: texts.append(content)
            except: pass
        
        if texts:
            embeddings = model.encode(texts, show_progress_bar=False)
            centroid = np.mean(embeddings, axis=0)
            
            # Normalize vector
            norm = np.linalg.norm(centroid)
            if norm > 0: centroid = centroid / norm
            
            topic_centroids[topic_name] = centroid
            topic_names.append(topic_name)

    if not topic_centroids:
        print("Không tạo được centroid nào. Kiểm tra lại file .txt trong seed!")
        return None, None
        
    centroid_matrix = np.array([topic_centroids[t] for t in topic_names])
    print(f"Đã khởi tạo {len(topic_names)} tâm cụm chủ đề.")
    return topic_names, centroid_matrix

# ==========================================
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"LỖI: Không tìm thấy file input tại: {INPUT_FILE}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")
    
    # 1. Load Model
    print("Loading Embedding Model...")
    model = SentenceTransformer(MODEL_NAME, device=device)
    # 2. Compute Topic Centroids
    topic_names, centroid_matrix = compute_topic_centroids(model, SEED_DATA_DIR)
    if centroid_matrix is None: return

    print("="*60)
    print(f"Bắt đầu phân loại và ghi file: {os.path.basename(OUTPUT_FILE)}")
    
    articles_buffer = [] 
    count_processed = 0
    
    with keep.presenting(), open(INPUT_FILE, 'rb') as f_in, open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        
        f_out.write('[')
        first_write = True
    
        objects = ijson.items(f_in, 'item')
        pbar = tqdm(total=LIMIT_ARTICLES, unit="docs")
        
        try:
            for article in objects:
                if count_processed >= LIMIT_ARTICLES:
                    break
                
                # Validate nội dung
                content = article.get('content', '')
                if not content or not isinstance(content, str) or len(content.strip()) < 50:
                    continue
                    
                articles_buffer.append(article)
                
                if len(articles_buffer) >= BATCH_SIZE:
                    process_and_write(articles_buffer, model, topic_names, centroid_matrix, f_out, first_write)
                    
                    if first_write: first_write = False
                    
                    count_processed += len(articles_buffer)
                    pbar.update(len(articles_buffer))
                    articles_buffer = [] # Clear buffer

            if articles_buffer:
                process_and_write(articles_buffer, model, topic_names, centroid_matrix, f_out, first_write)
                count_processed += len(articles_buffer)
                pbar.update(len(articles_buffer))

        except KeyboardInterrupt:
            print("\nDừng thủ công!")
        except Exception as e:
            print(f"\nLỗi runtime: {e}")
        
        f_out.write('\n]')
        pbar.close()

    print("\n" + "="*60)
    print(f"HOÀN TẤT! Đã xử lý {count_processed} bài báo.")
    print(f"Kết quả lưu tại: {OUTPUT_FILE}")

# ==========================================
def process_and_write(articles, model, topic_names, centroid_matrix, f_out, is_first_write):
    # cắt ngắn 1000 ký tự đầu để embed cho nhanh
    texts = [a.get('content', '')[:1000] for a in articles]
    
    # Embed batch
    embeddings = model.encode(texts, batch_size=len(articles), show_progress_bar=False)
    
    # (Batch, 768) x (768, Num_Topics)
    scores = cosine_similarity(embeddings, centroid_matrix)
    
    # max score
    best_indices = np.argmax(scores, axis=1)
    best_scores = np.max(scores, axis=1)
    
    for i, article in enumerate(articles):
        idx = best_indices[i]
        score = best_scores[i]
        
        # Topic label
        topic = topic_names[idx] if score >= CONFIDENCE_THRESHOLD else "Other"
        article['topic'] = topic
        # article['topic_score'] = float(score) 
        
        # Xử lý dấu phẩy giữa các object JSON
        if not is_first_write:
            f_out.write(',\n')
        else:
            is_first_write = False
            
        json.dump(article, f_out, ensure_ascii=False)

if __name__ == "__main__":
    main()