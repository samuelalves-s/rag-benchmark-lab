import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from beir.retrieval.evaluation import EvaluateRetrieval
from dotenv import load_dotenv

from langfuse import Langfuse, Evaluation

load_dotenv()
langfuse = Langfuse()

DATASET_NAME = "scifact-queries"

# ============================================
# PARÂMETROS DA VERSÃO ATUAL — mude aqui entre testes
# ============================================
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 2     # precisa bater com o que foi usado no rag.py pra gerar o índice
OVERLAP = 2
TOP_K = 10

RUN_NAME = f"chunk{CHUNK_SIZE}-overlap{OVERLAP}-topk{TOP_K}"

# ---------------------------------------------------------
# Carrega modelo e índice (uma vez só, fora do loop de itens)
# ---------------------------------------------------------
print(f"Carregando modelo: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

print("Carregando índice FAISS salvo...")
index = faiss.read_index("vector_store/scifact.index")

with open("vector_store/chunk_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

chunk_texts = metadata["chunk_texts"]
chunk_doc_ids = metadata["chunk_doc_ids"]
print(f"Índice carregado com {index.ntotal} vetores")


# ---------------------------------------------------------
# TASK: roda para CADA item do dataset (cada query)
# Recebe o item (input = texto da query) e devolve os
# documentos recuperados com seus scores.
# ---------------------------------------------------------
def retrieve_task(*, item, **kwargs):
    query_text = item.input
    query_embedding = model.encode([query_text], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k=TOP_K)

    doc_scores = {}
    for rank, idx in enumerate(indices[0]):
        doc_id = chunk_doc_ids[idx]
        score = 1.0 / (distances[0][rank] + 1e-6)
        if doc_id not in doc_scores or score > doc_scores[doc_id]:
            doc_scores[doc_id] = float(score)

    return doc_scores


# ---------------------------------------------------------
# RUN EVALUATOR: roda UMA VEZ no final, com os resultados
# de TODAS as queries juntos. É aqui que recalculamos as
# métricas oficiais do BEIR (nDCG, Recall, MAP, Precision),
# que só fazem sentido calculadas sobre o conjunto inteiro.
# ---------------------------------------------------------
def beir_metrics_evaluator(*, item_results, **kwargs):
    # Reconstrói o formato que o EvaluateRetrieval do BEIR espera:
    # results = {query_id: {doc_id: score, ...}, ...}
    # qrels   = {query_id: {doc_id: relevance, ...}, ...}
    results = {}
    qrels = {}

    for item_result in item_results:
        query_id = item_result.item.metadata["query_id"]
        results[query_id] = item_result.output
        qrels[query_id] = item_result.item.expected_output or {}

    evaluator = EvaluateRetrieval()
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, evaluator.k_values)

    metric_key = f"@{TOP_K}"
    evaluations = []
    for name, metric_dict in [("ndcg", ndcg), ("map", _map), ("recall", recall), ("precision", precision)]:
        for k, v in metric_dict.items():
            if k.endswith(metric_key):
                evaluations.append(Evaluation(name=f"{name}{metric_key}", value=float(v)))

    return evaluations


# ---------------------------------------------------------
# Roda o experimento
# ---------------------------------------------------------
if __name__ == "__main__":
    dataset = langfuse.get_dataset(DATASET_NAME)

    result = dataset.run_experiment(
        name="rag-benchmark",
        run_name=RUN_NAME,
        description=f"chunk_size={CHUNK_SIZE}, overlap={OVERLAP}, top_k={TOP_K}, model={MODEL_NAME}",
        task=retrieve_task,
        run_evaluators=[beir_metrics_evaluator],
        metadata={
            "chunk_size": str(CHUNK_SIZE),
            "overlap": str(OVERLAP),
            "top_k": str(TOP_K),
            "model_name": MODEL_NAME,
        },
    )

    print(result.format())

    langfuse.flush()