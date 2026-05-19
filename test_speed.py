import os, time, shutil

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.core.embeddings import resolve_embed_model

INDEX_DIR = "chroma_data"

# Load embedding model (always needed)
t0 = time.time()
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-zh-v1.5")
t1 = time.time()
print(f"Embed model loaded: {t1 - t0:.2f}s")

# Try loading from cache
cache_exists = os.path.exists(INDEX_DIR) and os.listdir(INDEX_DIR)

if cache_exists:
    print(">>> Cache HIT: loading index from disk")
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index = load_index_from_storage(storage_context)
    t2 = time.time()
    print(f"Index loaded from cache: {t2 - t1:.2f}s")
    print(f"TOTAL: {t2 - t0:.2f}s")
else:
    print(">>> Cache MISS: building index from scratch")
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    t2 = time.time()
    print(f"Index built: {t2 - t1:.2f}s")
    print(f"TOTAL: {t2 - t0:.2f}s")
    # Save for next time
    index.storage_context.persist(persist_dir=INDEX_DIR)
    print(f"Index saved to {INDEX_DIR}/")
