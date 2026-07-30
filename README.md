# rag-benchmark-lab
 
Laboratório de testes para experimentar diferentes modelos de embedding, estratégias de chunking e configurações de retrieval em pipelines de RAG (Retrieval-Augmented Generation), com avaliação através de métricas estatísticas padrão de IR (Information Retrieval).
 
## Objetivo
 
Este repositório serve para:
- Testar diferentes modelos de embedding (ex: `all-MiniLM-L6-v2`, `BAAI/bge-m3`, `BAAI/bge-small-en-v1.5`)
- Testar diferentes estratégias de chunking (tamanho, overlap)
- Avaliar os resultados com métricas estatísticas (nDCG, MAP, Recall, Precision) usando o benchmark **BEIR**
- Versionar cada experimento para comparação posterior
