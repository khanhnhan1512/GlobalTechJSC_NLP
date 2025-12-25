import ijson
import os
import time
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from wakepy import keep
import torch

# SYSTEM CONFIGURATION & INDEX SETTINGS
FILE_PATH = './inputs/merged_news.json'
CHECKPOINT_FILE = 'checkpoint_index.txt'
MILVUS_URI = "http://127.0.0.1:19530"
COLLECTION_NAME = "globaltech_nlp_project"

# Batch processing configuration
ENCODE_BATCH_SIZE = 32   # Number of text chunks to embed at once (optimizes GPU usage)
INSERT_BATCH_SIZE = 100 # Number of vectors to insert into DB at once

# HNSW Index Configuration
INDEX_PARAMS = MilvusClient.prepare_index_params()
INDEX_PARAMS.add_index(
    field_name="vector",
    index_type="HNSW",    # High-performance search algorithm
    metric_type="COSINE", # Metric for NLP semantic similarity
    params={
        "M": 16,              # Max number of connections per node
        "efConstruction": 200 # Search depth during index construction (higher = more accurate but slower build)
    },
    index_name="vector_index"
)

# CLIENT & MODEL INITIALIZATION
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Processing device: {device.upper()}")

model = SentenceTransformer("hiieu/halong_embedding", device=device)
# Get embedding dimension from the model (typically 768 for this model)
MODEL_DIM = model.get_sentence_embedding_dimension()

client = MilvusClient(uri=MILVUS_URI)

# Create Collection if it does not exist (Index is built later)
if not client.has_collection(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=MODEL_DIM,
        metric_type="COSINE", # Metric must match the index configuration
        auto_id=True
    )
    print(f"Created new collection: {COLLECTION_NAME}")

# Text splitter configuration using LangChain
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600, 
    chunk_overlap=100, 
    separators=["\n\n", "\n", ".", " ", ""]
)

# Load Checkpoint (Resume capability)
start_index = 0
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, 'r') as f:
        try:
            start_index = int(f.read().strip())
            print(f"Resume: Starting from article index {start_index}")
        except: 
            pass

# PROCESSING LOGIC
text_buffer = [] 
meta_buffer = [] 
milvus_insert_buffer = []

def save_checkpoint(current_idx):
    """Save the current processing index to a file."""
    with open(CHECKPOINT_FILE, 'w') as f:
        f.write(str(current_idx))

def process_encode_and_insert(force=False):
    """
    Encodes text chunks in the buffer and inserts vectors into Milvus.
    Triggered when buffers are full or processing is finished (force=True).
    """
    global milvus_insert_buffer, text_buffer, meta_buffer
    
    # 1. Embedding Step
    if text_buffer and (len(text_buffer) >= ENCODE_BATCH_SIZE or force):
        try:
            # Batch encoding for performance
            vectors = model.encode(text_buffer, batch_size=ENCODE_BATCH_SIZE, show_progress_bar=False)
            
            # Combine vectors with metadata
            for i, vec in enumerate(vectors):
                item = meta_buffer[i]
                item['vector'] = vec
                milvus_insert_buffer.append(item)
            
            # Clear buffers after encoding
            text_buffer = []
            meta_buffer = []
        except Exception as e:
            print(f"Embedding Error: {e}")

    # 2. Insertion Step
    if milvus_insert_buffer and (len(milvus_insert_buffer) >= INSERT_BATCH_SIZE or force):
        try:
            client.insert(collection_name=COLLECTION_NAME, data=milvus_insert_buffer)
            print(f"Inserted {len(milvus_insert_buffer)} vectors. (Checkpoint: {current_global_index})")
            
            # Save checkpoint only after successful insertion
            save_checkpoint(current_global_index)
            
            # Clear insertion buffer
            milvus_insert_buffer = []
        except Exception as e:
            print(f"Milvus Insert Error: {e}")

# MAIN LOOP & AUTO INDEXING
print("Starting process... (System sleep prevented)")

# Use wakepy to prevent the system from sleeping during execution
with keep.presenting(): 
    try:
        with open(FILE_PATH, 'rb') as f:
            # Stream JSON items to avoid loading the entire file into RAM
            objects = ijson.items(f, 'item')
            current_global_index = -1
            
            for i, article in enumerate(objects):
                current_global_index = i
                
                # Logic to skip previously processed articles
                if i < start_index:
                    if i % 1000 == 0: 
                        print(f"Skipping... {i}/{start_index}", end='\r')
                    continue
                
                # Validate content
                content = article.get('content', '')
                if not content or not isinstance(content, str) or len(content.strip()) == 0:
                    continue

                # Chunking
                chunks = text_splitter.split_text(content)
                for chunk_idx, chunk_text in enumerate(chunks):
                    text_buffer.append(chunk_text)
                    meta_buffer.append({
                        "text": chunk_text,
                        "original_id": str(article.get('id')),
                        "title": article.get('title', ''),
                        'author_name': article.get('author_name', ''),
                        'description': article.get('description', ''),
                        "url": article.get('url', ''),
                        'created_time': article.get('created_time', ''),
                        "chunk_index": chunk_idx
                    })
                
                # Trigger processing if buffers are full
                if len(text_buffer) >= ENCODE_BATCH_SIZE or len(milvus_insert_buffer) >= INSERT_BATCH_SIZE:
                    process_encode_and_insert(force=False)

            # Process any remaining data in the buffers
            process_encode_and_insert(force=True)

        print("\nDATA LOADING COMPLETE! STARTING INDEX BUILD...")
        
        # BUILD INDEX (HNSW)
        try:
            client.drop_index(collection_name=COLLECTION_NAME, index_name="vector_index")
        except: 
            pass

        print("⏳ Building HNSW Index (M=16, ef=200)... Please wait...")
        start_build = time.time()
        
        client.create_index(
            collection_name=COLLECTION_NAME,
            index_params=INDEX_PARAMS
        )
        
        # Load collection into RAM for querying
        client.load_collection(COLLECTION_NAME)
        
        end_build = time.time()
        print(f"Index Build Complete in {end_build - start_build:.2f}s")
        print("PROGRAM FINISHED SUCCESSFULLY!")

        # Remove checkpoint file after successful completion
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)

    except KeyboardInterrupt:
        print("\nProgram stopped manually.")
        print("Run again to resume from the last checkpoint.")