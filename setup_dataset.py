"""
Roda isso UMA VEZ SÓ pra criar o dataset no Langfuse com as queries do SciFact.
Depois disso, use benchmark_experiment.py pra rodar experimentos contra ele.
"""
import os
from collections import defaultdict
from beir.datasets.data_loader import GenericDataLoader
from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()
langfuse = Langfuse()

DATASET_NAME = "scifact-queries"

data_path = "datasets/scifact"
_, queries, qrels = GenericDataLoader(data_path).load(split="test")

print(f"Total de queries: {len(queries)}")

# Cria o dataset (se já existir com esse nome, o Langfuse só reaproveita)
langfuse.create_dataset(
    name=DATASET_NAME,
    description="Queries de teste do SciFact (BEIR) para avaliação do RAG",
    metadata={"source": "BEIR/scifact", "split": "test"},
)

# Sobe cada query como um item do dataset
# input = a pergunta em si
# expected_output = os documentos relevantes (gabarito / qrels), com sua nota de relevância
# metadata = query_id, pra conseguirmos recompor os resultados depois
count = 0
for query_id, query_text in queries.items():
    relevant_docs = qrels.get(query_id, {})  # {doc_id: relevance_score}

    langfuse.create_dataset_item(
        dataset_name=DATASET_NAME,
        input=query_text,
        expected_output=relevant_docs,
        metadata={"query_id": query_id},
    )
    count += 1

print(f"{count} itens enviados para o dataset '{DATASET_NAME}' no Langfuse.")
print("Confira em: Datasets > scifact-queries no dashboard.")