import json
import time
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

print("OpenSearch Benchmark (Cached Embeddings)")

BASE_URL = "http://localhost:9200"
HEADERS_JSON = {"Content-Type": "application/json"}
HEADERS_NDJSON = {"Content-Type": "application/x-ndjson"}

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

def wait_for_ready(index_name, timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        r = requests.get(
            f"{BASE_URL}/_cluster/health/{index_name}?wait_for_status=yellow&timeout=1s"
        )
        if r.ok:
            return True
        time.sleep(1)
    return False

for size in sizes:
    print()
    print("Dataset size:", size)

    index_name = f"agnews_{size}"
    data = embeddings[:size].astype("float32")
    memory_usage_mb = data.nbytes / 1024 / 1024

    requests.delete(f"{BASE_URL}/{index_name}")

    index_body = {
        "settings": {
            "index": {
                "knn": True
            }
        },
        "mappings": {
            "properties": {
                "doc_id": {"type": "integer"},
                "text": {"type": "text"},
                "vector": {
                    "type": "knn_vector",
                    "dimension": 384,
                    "space_type": "l2"
                }
            }
        }
    }

    start = time.time()

    r = requests.put(
        f"{BASE_URL}/{index_name}",
        headers=HEADERS_JSON,
        data=json.dumps(index_body)
    )
    r.raise_for_status()

    bulk_batch_size = 1000

    for i in range(0, len(data), bulk_batch_size):
        lines = []
        end = min(i + bulk_batch_size, len(data))

        for j in range(i, end):
            lines.append(json.dumps({"index": {"_index": index_name, "_id": int(j)}}))
            lines.append(json.dumps({
                "doc_id": int(j),
                "text": str(texts[j]),
                "vector": data[j].tolist()
            }))

        payload = "\n".join(lines) + "\n"

        r = requests.post(
            f"{BASE_URL}/_bulk",
            headers=HEADERS_NDJSON,
            data=payload
        )
        r.raise_for_status()

        resp = r.json()
        if resp.get("errors"):
            raise RuntimeError(f"Bulk indexing failed for {index_name}")

    requests.post(f"{BASE_URL}/{index_name}/_refresh")
    wait_for_ready(index_name)

    build_time = time.time() - start

    latencies = []

    for q in query_embeddings:
        query_body = {
            "size": 5,
            "query": {
                "knn": {
                    "vector": {
                        "vector": q.tolist(),
                        "k": 5
                    }
                }
            }
        }

        for _ in range(10):
            start = time.time()
            r = requests.post(
                f"{BASE_URL}/{index_name}/_search",
                headers=HEADERS_JSON,
                data=json.dumps(query_body)
            )
            r.raise_for_status()
            latencies.append(time.time() - start)

    avg_latency = np.mean(latencies)

    print("build_time:", round(build_time, 6), "seconds")
    print("latency:", round(avg_latency, 6), "seconds")
    print("memory:", round(memory_usage_mb, 2), "MB")