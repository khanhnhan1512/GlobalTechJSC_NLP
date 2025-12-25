import ijson
import os
import time
import torch
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from wakepy import keep

# ==========================================
CURRENT_SCRIPT_PATH = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_SCRIPT_PATH))

FILE_PATH = os.path.join(PROJECT_ROOT, 'inputs', 'merged_news_100k_labeled.json')
CHECKPOINT_FILE = os.path.join(PROJECT_ROOT, 'embeddings', 'checkpoint_index_v2.txt')

MILVUS_URI = "http://127.0.0.1:19530"
COLLECTION_NAME = "globaltech_news_labeled" 

# Batch processing config
ENCODE_BATCH_SIZE = 32
INSERT_BATCH_SIZE = 100

client = MilvusClient(uri=MILVUS_URI)

schema = MilvusClient.create_schema(
    auto_id=True, 
    enable_dynamic_field=True 
)

schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=768)
schema.add_field(field_name="topic", datatype=DataType.VARCHAR, max_length=64)
schema.add_field(field_name="chunk_index", datatype=DataType.INT32)

# Index Params
index_params = MilvusClient.prepare_index_params()

# CẬP NHẬT INDEX VECTOR (IVF_SQ8)
index_params.add_index(
    field_name="vector",
    index_type="IVF_SQ8",   
    metric_type="COSINE", 
    params={"nlist": 2048},   # Chia không gian thành 2048 cụm
    index_name="vector_index_optimized"
)

# Index cho Topic (Scalar Index)
index_params.add_index(
    field_name="topic",
    index_type="Trie", 
    index_name="topic_index"
)

# KHỞI TẠO COLLECTION
if client.has_collection(COLLECTION_NAME):
    print(f"Collection {COLLECTION_NAME} already exists.")
else:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params
    )
    print(f"Created NEW collection: {COLLECTION_NAME} with IVF_SQ8 & Topic schema.")

# LOAD MODEL & PREPARE TEXT SPLITTER
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Processing device: {device.upper()}")
model = SentenceTransformer("hiieu/halong_embedding", device=device)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600, 
    chunk_overlap=100, 
    separators=["\n\n", "\n", ".", " ", ""]
)

# Load Checkpoint
start_index = 0
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, 'r') as f:
        try:
            start_index = int(f.read().strip())
            print(f"Resuming from index {start_index}")
        except: pass

text_buffer = [] 
meta_buffer = [] 
milvus_insert_buffer = []

def process_encode_and_insert(force=False):
    global milvus_insert_buffer, text_buffer, meta_buffer
    
    # Encode
    if text_buffer and (len(text_buffer) >= ENCODE_BATCH_SIZE or force):
        try:
            vectors = model.encode(text_buffer, batch_size=ENCODE_BATCH_SIZE, show_progress_bar=False)
            for i, vec in enumerate(vectors):
                item = meta_buffer[i]
                item['vector'] = vec
                milvus_insert_buffer.append(item)
            text_buffer = []
            meta_buffer = []
        except Exception as e:
            print(f"Embedding Error: {e}")

    # Insert
    if milvus_insert_buffer and (len(milvus_insert_buffer) >= INSERT_BATCH_SIZE or force):
        try:
            client.insert(collection_name=COLLECTION_NAME, data=milvus_insert_buffer)
            print(f"Inserted {len(milvus_insert_buffer)} vectors. (Idx: {current_global_index})", end='\r')
            with open(CHECKPOINT_FILE, 'w') as f:
                f.write(str(current_global_index))
            milvus_insert_buffer = []
        except Exception as e:
            print(f"Milvus Insert Error: {e}")


print(f"Reading input: {FILE_PATH}")
with keep.presenting(): 
    try:
        with open(FILE_PATH, 'rb') as f:
            objects = ijson.items(f, 'item')
            current_global_index = -1
            
            for i, article in enumerate(objects):
                current_global_index = i
                if i < start_index: continue
                
                content = article.get('content', '')
                if not content or len(content.strip()) == 0: continue

                # Lấy Topic từ file JSON
                topic_label = article.get('topic', 'Other') 

                chunks = text_splitter.split_text(content)
                for chunk_idx, chunk_text in enumerate(chunks):
                    text_buffer.append(chunk_text)
                    meta_buffer.append({
                        "text": chunk_text,
                        "original_id": str(article.get('id')),
                        "title": article.get('title', ''),
                        "topic": topic_label,
                        "url": article.get('url', ''),
                        "chunk_index": chunk_idx,
                        "created_time": article.get('created_time', '')
                    })
                
                if len(text_buffer) >= ENCODE_BATCH_SIZE:
                    process_encode_and_insert(force=False)

            process_encode_and_insert(force=True)

        print("DATA LOADING COMPLETE!")
        print("Loading collection to RAM...")
        client.load_collection(COLLECTION_NAME)
        
        # Kiểm tra trạng thái load
        state = client.get_load_state(collection_name=COLLECTION_NAME)
        print(f"READY! Load state: {state}")
        
        if os.path.exists(CHECKPOINT_FILE): os.remove(CHECKPOINT_FILE)

    except KeyboardInterrupt:
        print("\nProgram stopped manually.")