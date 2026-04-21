import time
import numpy as np

from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient

print("Milvus Benchmark (Cached Embeddings)")

sizes = [10000, 50000, 100000]

queries = [
    "What is cloud computing?",
    "What is artificial intelligence?",
    "What is business news?",
    "What is sports competition?",
    "What is world politics?"
]

print("Loading saved embeddings...")
embeddings = np.load("embeddings_100k.npy")
texts = np.load("texts_100k.npy", allow_pickle=True)

print("Embeddings shape:", embeddings.shape)
print("Texts:", len(texts))

print("Loading model for query encoding...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Encoding queries...")
query_embeddings = [
    model.encode([q], convert_to_numpy=True, show_progress_bar=False).astype("float32")[0]
    for q in queries
]

client = MilvusClient("milvus_benchmark.db")

for size in sizes:
    print()
    print("Dataset size:", size)

    data = embeddings[:size].astype("float32")
    memory_usage_mb = data.nbytes / 1024 / 1024

    collection_name = f"agnews_{size}"

    if client.has_collection(collection_name=collection_name):
        client.drop_collection(collection_name=collection_name)

    start = time.time()

    client.create_collection(
        collection_name=collection_name,
        dimension=data.shape[1],
        metric_type="L2"
    )

    insert_data = [
        {"id": int(i), "vector": data[i]}
        for i in range(len(data))
    ]

    client.insert(collection_name=collection_name, data=insert_data)

    build_time = time.time() - start

    latencies = []

    for q in query_embeddings:
        for _ in range(10):
            start = time.time()
            client.search(
                collection_name=collection_name,
                data=[q],
                limit=5
            )
            latencies.append(time.time() - start)

    avg_latency = np.mean(latencies)

    print("build_time:", round(build_time, 6), "seconds")
    print("latency:", round(avg_latency, 6), "seconds")
    print("memory:", round(memory_usage_mb, 2), "MB")