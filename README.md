# Assistente de Saúde da Mulher
## Arquitetura RAG + Fine-Tuning (QLoRA) + LangGraph

### Visão Geral
Assistente médico especializado em saúde da mulher. Combina duas abordagens:
- **RAG** (Retrieval-Augmented Generation) orquestrado via LangGraph;
- **Fine-tuning** de um LLM open-source com dados médicos, usando QLoRA.

### Stack Tecnológica
| Componente         | Tecnologia                          |
|--------------------|-------------------------------------|
| LLM (RAG)          | LLaMA 3.1 via OpenRouter (free)     |
| LLM (fine-tuning)  | google/gemma-2-2b-it + QLoRA        |
| Fine-tuning        | PEFT (LoRA 4-bit) + TRL SFTTrainer  |
| Orquestração       | LangGraph                           |
| RAG / Embeddings   | LangChain + ChromaDB                |
| Embeddings model   | all-MiniLM-L6-v2                    |
| Anonimização       | Regex + spaCy NER (pt_core_news_sm) |
| Ambiente           | Google Colab (GPU T4)               |

### Dataset
| Fonte                               | Registros | Idioma |
|-------------------------------------|-----------|--------|
| Sintético (protocolos SBG/FEBRASGO) | 35        | pt-BR  |
| MedQuAD - NIH                       | ~281      | en     |
| Menstrual Health Awareness Dataset  | ~516      | en     |
| **Total**                           | **832**   |        |

### Preparação dos Dados (Etapa 4)
1. **Preprocessing** — normalização Unicode, remoção de HTML, limpeza de espaços.
2. **Anonimização** — remoção de PII via regex (CPF, telefone, e-mail, datas,
   prontuário, cartão SUS) e NER spaCy (nomes e locais). Gera o arquivo
   `dataset_anonimizado.json`.
3. **Curadoria** — deduplicação por pergunta, filtro de qualidade (respostas
   curtas), truncamento de respostas longas. Split 85/15 treino/validação.

### Pipeline de Fine-Tuning (QLoRA)
- Modelo base `google/gemma-2-2b-it` carregado em 4-bit (NF4).
- Adaptadores LoRA (r=16, alpha=32) treinados com TRL `SFTTrainer`.
- 3 épocas, otimizador `paged_adamw_8bit`, lr 2e-4, ~15-20 min em GPU T4.
- Avaliação base vs. fine-tuned com métricas ROUGE e comparação qualitativa.

### Pipeline RAG
```
Pergunta → [Triagem] → [Busca ChromaDB] → [LLM] → [Resposta formatada]
```

### Nós LangGraph
- **triagem**: classifica urgência (alta/média/baixa) por palavras-chave
- **busca_rag**: recupera top-3 documentos por similaridade semântica
- **gerar_resposta**: envia contexto + pergunta ao LLM com prompt médico
- **formatar_saida**: exibe resposta com emoji de urgência

A Célula 28 reconstrói esse fluxo usando o modelo fine-tuned (encapsulado em
`HuggingFacePipeline`) no lugar do LLM do OpenRouter.

### Como executar
1. Abra o notebook no Google Colab e selecione o runtime **T4 GPU**.
2. Configure nos *Secrets* do Colab:
   - `OPENROUTER_API_KEY` — chave do OpenRouter (etapa RAG);
   - `HF_TOKEN` — token HuggingFace, com a licença do Gemma aceita.
3. Execute as células em ordem:
   - Células 1–16: dataset, RAG e LangGraph;
   - Células 17–28: anonimização, curadoria, fine-tuning e avaliação.

### Segurança e Validação
- O assistente nunca prescreve medicamentos diretamente — sempre orienta a
  validação por um profissional de saúde.
- Cada resposta do RAG indica a fonte do conhecimento utilizado (explainability).
- A triagem de urgência emite alerta para casos graves.
