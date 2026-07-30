import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from beir import util
from beir.datasets.data_loader import GenericDataLoader

# Modelo leve
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Dataset
dataset = "scifact"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
data_path = util.download_and_unzip(url, "datasets")

corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

print(f"Documentos no corpus: {len(corpus)}")


# Chunking
def chunk_text(text, chunk_size=500, overlap=50):
    """Divide um texto em pedaços menores, com sobreposição."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


chunk_texts = []     # lista com o texto de cada chunk
chunk_doc_ids = []   # de qual documento original cada chunk veio

for doc_id, doc in corpus.items():
    full_text = doc.get("title", "") + " " + doc.get("text", "")
    chunks = chunk_text(full_text)
    for chunk in chunks:
        chunk_texts.append(chunk)
        chunk_doc_ids.append(doc_id)

print(f"Total de chunks gerados: {len(chunk_texts)}")


# Gerando embeddings dos chunks
print("Gerando embeddings dos chunks...")
chunk_embeddings = model.encode(
    chunk_texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

print(f"Shape dos embeddings: {chunk_embeddings.shape}")

# Criando índice FAISS
dimension = chunk_embeddings.shape[1]  # 384 no caso do MiniLM

index = faiss.IndexFlatL2(dimension)
index.add(chunk_embeddings)

print(f"Total de vetores no índice: {index.ntotal}")

# Salvando índice e metadados em disco (com checagem de persistência)
index_path = "vector_store/scifact.index"
metadata_path = "vector_store/chunk_metadata.pkl"

os.makedirs("vector_store", exist_ok=True)

if os.path.exists(index_path) and os.path.exists(metadata_path):
    resposta = input(
        "Já existe um índice salvo. Deseja mantê-lo (usar o que já está em disco) "
        "ou refazer (sobrescrever com os dados atuais)? [manter/refazer]: "
    ).strip().lower()

    if resposta == "manter":
        print("Mantendo o índice existente. Recarregando do disco...")
        index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        chunk_texts = metadata["chunk_texts"]
        chunk_doc_ids = metadata["chunk_doc_ids"]
    elif resposta == "refazer":
        faiss.write_index(index, index_path)
        with open(metadata_path, "wb") as f:
            pickle.dump({"chunk_texts": chunk_texts, "chunk_doc_ids": chunk_doc_ids}, f)
        print("Índice refeito e salvo em disco!")
    else:
        print("Resposta inválida. Mantendo os dados apenas em memória (nada salvo).")
else:
    faiss.write_index(index, index_path)
    with open(metadata_path, "wb") as f:
        pickle.dump({"chunk_texts": chunk_texts, "chunk_doc_ids": chunk_doc_ids}, f)
    print("Índice criado e salvo em disco pela primeira vez!")


# Realizando uma busca de teste
query = "What is the effect of stress on immune function?"
query_embedding = model.encode([query], convert_to_numpy=True)

k = 5  # quantos resultados mais próximos queremos
distances, indices = index.search(query_embedding, k)

print(f"\nQuery: {query}\n")
for rank, idx in enumerate(indices[0]):
    print(f"Rank {rank+1} (distância={distances[0][rank]:.4f}):")
    print(f"  Documento original: {chunk_doc_ids[idx]}")
    print(f"  Trecho: {chunk_texts[idx][:200]}...\n")