# Benchmarking-Vector-Retrieval-Engines-for-Cloud-based-RAG-Systems

## Project Overview
This project benchmarks three vector retrieval systems for cloud-based RAG:
- FAISS
- Milvus
- OpenSearch

## Environment
- AWS EC2 t3.large
- Ubuntu 22.04
- Python 3.x

## Dataset
AG News

## Embedding Model
sentence-transformers/all-MiniLM-L6-v2

## How to Run
### FAISS
python faiss_benchmark.py

### Milvus
python milvus_benchmark.py

### OpenSearch
python opensearch_benchmark.py

## Expected Output
Build time, query latency, and memory usage for 10k, 50k, and 100k datasets.
