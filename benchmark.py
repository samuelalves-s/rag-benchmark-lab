import os
import json
import glob
import pickle
from datetime import datetime
import faiss
from sentence_transformers import SentenceTransformer
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # mude aqui quando trocar de modelo
CHUNK_SIZE = 500   # ajuste se mudar o chunking no build_index.py
OVERLAP = 50
TOP_K = 10         # quantos resultados a busca no FAISS considera

# ---------------------------------------------------------
# 1. Carregar o modelo (o mesmo usado para construir o índice)
# ---------------------------------------------------------
print(f"Carregando modelo: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

# ---------------------------------------------------------
# 2. Carregar o índice FAISS e os metadados já salvos em disco
#    (não recalcula embeddings, só lê o que já foi gerado)
# ---------------------------------------------------------
print("Carregando índice FAISS salvo...")
index = faiss.read_index("vector_store/scifact.index")

with open("vector_store/chunk_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

chunk_texts = metadata["chunk_texts"]
chunk_doc_ids = metadata["chunk_doc_ids"]

print(f"Índice carregado com {index.ntotal} vetores")

# ---------------------------------------------------------
# 3. Carregar queries e qrels (o gabarito) do dataset SciFact
#    (precisamos deles para saber o que perguntar e o que é "certo")
# ---------------------------------------------------------
data_path = "datasets/scifact"
_, queries, qrels = GenericDataLoader(data_path).load(split="test")

print(f"Total de queries no benchmark: {len(queries)}")

# ---------------------------------------------------------
# 4. Rodar a busca (aqui o modelo "responde" cada pergunta)
# ---------------------------------------------------------
print("Rodando buscas no índice para todas as queries...")
results = {}

for query_id, query_text in queries.items():
    query_embedding = model.encode([query_text], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k=TOP_K)

    doc_scores = {}
    for rank, idx in enumerate(indices[0]):
        doc_id = chunk_doc_ids[idx]
        score = 1.0 / (distances[0][rank] + 1e-6)

        if doc_id not in doc_scores or score > doc_scores[doc_id]:
            doc_scores[doc_id] = float(score)

    results[query_id] = doc_scores

print(f"Buscas concluídas para {len(results)} queries")

# ---------------------------------------------------------
# 5. Avaliar com métricas estatísticas (nDCG, Recall, MAP, Precision)
# ---------------------------------------------------------
evaluator = EvaluateRetrieval()
ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, evaluator.k_values)

print("\n================ RESULTADOS ================")
print("\nnDCG:")
for k, v in ndcg.items():
    print(f"  {k}: {v:.4f}")

print("\nRecall:")
for k, v in recall.items():
    print(f"  {k}: {v:.4f}")

print("\nMAP:")
for k, v in _map.items():
    print(f"  {k}: {v:.4f}")

print("\nPrecision:")
for k, v in precision.items():
    print(f"  {k}: {v:.4f}")

# ---------------------------------------------------------
# 6. Salvar os resultados em um arquivo versionado (v1, v2, v3...)
# ---------------------------------------------------------
os.makedirs("results", exist_ok=True)

# Detecta a próxima versão disponível olhando os arquivos já existentes
existing_versions = glob.glob("results/v*_results.json")
version_numbers = [
    int(os.path.basename(f).split("_")[0].replace("v", ""))
    for f in existing_versions
]
next_version = max(version_numbers, default=0) + 1
output_path = f"results/v{next_version}_results.json"

run_summary = {
    "version": f"v{next_version}",
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "config": {
        "model_name": MODEL_NAME,
        "chunk_size": CHUNK_SIZE,
        "overlap": OVERLAP,
        "top_k": TOP_K,
    },
    "metrics": {
        "ndcg": ndcg,
        "map": _map,
        "recall": recall,
        "precision": precision,
    },
}

with open(output_path, "w") as f:
    json.dump(run_summary, f, indent=2)

print(f"\nResultados salvos em: {output_path}")