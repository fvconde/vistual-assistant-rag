# Assistente de Saude da Mulher
## Arquitetura RAG + Fine-Tuning + LangGraph

### Visao Geral
Assistente medico especializado em saude da mulher, construido com
arquitetura **RAG** (Retrieval-Augmented Generation) e **Fine-Tuning** (QLoRA),
orquestrado via **LangGraph**.

O projeto implementa dois pipelines complementares:
1. **RAG + OpenRouter**: usa LLaMA 3.1 via API para respostas de alta qualidade
2. **RAG + Fine-Tuning**: usa Gemma-2-2B-it treinado com dados medicos para inferencia local

### Stack Tecnologica
| Componente          | Tecnologia                          |
|---------------------|-------------------------------------|
| LLM (API)           | LLaMA 3.1 via OpenRouter (free)     |
| LLM (Fine-Tuned)    | Gemma-2-2B-it + QLoRA               |
| Orquestracao        | LangGraph                           |
| RAG / Embeddings    | LangChain + ChromaDB                |
| Embeddings model    | all-MiniLM-L6-v2                    |
| Fine-Tuning stack   | transformers + peft + trl + bitsandbytes |
| Anonimizacao        | SpaCy (pt_core_news_sm) + regex     |
| Avaliacao           | ROUGE (evaluate)                    |
| Ambiente            | Google Colab (T4 GPU)               |

### Dataset
| Fonte                               | Registros | Idioma |
|-------------------------------------|-----------|--------|
| Sintetico (protocolos SBG/FEBRASGO) | 35        | pt-BR  |
| MedQuAD - NIH                       | ~281      | en     |
| Menstrual Health Awareness Dataset  | ~516      | en     |
| **Total**                           | **832**   |        |

### Pipeline RAG
```
Pergunta -> [Triagem] -> [Busca ChromaDB] -> [LLM] -> [Resposta formatada]
```

### Nos LangGraph
- **triagem**: classifica urgencia (alta/media/baixa) por palavras-chave
- **busca_rag**: recupera top-3 documentos por similaridade semantica
- **gerar_resposta**: envia contexto + pergunta ao LLM com prompt medico
- **formatar_saida**: exibe resposta com nivel de urgencia

### Pipeline de Fine-Tuning (Etapa 4)
```
Dataset -> [Anonimizacao] -> [Curadoria] -> [Formato Instrucao] -> [QLoRA Training] -> [Avaliacao]
```

**Preprocessing e Anonimizacao:**
- Regex para PII brasileira (CPF, telefone, email, datas, prontuarios)
- SpaCy NER (pt_core_news_sm) para nomes e locais
- Filtragem de qualidade: remocao de respostas curtas, deduplicacao, truncamento

**Fine-Tuning QLoRA:**
- Modelo: Gemma-2-2B-it (quantizado 4-bit NF4)
- Tecnica: LoRA (r=16, alpha=32, dropout=0.1)
- Modulos alvo: attention (q/k/v/o_proj) + MLP (gate/up/down_proj)
- Treinamento: SFTTrainer, 3 epochs, batch efetivo 16, lr 2e-4

**Avaliacao:**
- Metricas ROUGE (ROUGE-1, ROUGE-2, ROUGE-L)
- Comparacao qualitativa com perguntas de teste
- Curva de loss (training + validation)

### Como executar
1. Abra o notebook `virtual-assistant.ipynb` no Google Colab
2. Configure nos Secrets do Colab:
   - `OPENROUTER_API_KEY` (chave do OpenRouter)
   - `HF_TOKEN` (token do HuggingFace com licenca do Gemma aceita)
3. Execute as celulas em ordem:
   - **Celulas 1-8**: Carregamento e preprocessamento do dataset
   - **Celulas 9-11**: RAG com ChromaDB + LangChain
   - **Celulas 12-14**: Orquestracao com LangGraph + testes
   - **Celulas 15-16**: README e backup no Drive
   - **Celulas 17-28**: Fine-tuning completo (Etapa 4)
     - 17-18: Setup e login HuggingFace
     - 19: Anonimizacao
     - 20: Curadoria
     - 21: Formato de instrucao + split treino/validacao
     - 22-23: Carregamento do modelo + LoRA
     - 24-25: Treinamento + salvamento
     - 26-27: Avaliacao + curva de loss
     - 28: Integracao com LangGraph
