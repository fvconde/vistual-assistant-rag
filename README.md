# Assistente de Saúde da Mulher
## Arquitetura RAG + LangGraph

### Visão Geral
Assistente médico especializado em saúde da mulher, construído com
arquitetura RAG (Retrieval-Augmented Generation) orquestrada via LangGraph.

### Stack Tecnológica
| Componente        | Tecnologia                        |
|-------------------|-----------------------------------|
| LLM               | LLaMA 3.1 via OpenRouter (free)   |
| Orquestração      | LangGraph                         |
| RAG / Embeddings  | LangChain + ChromaDB              |
| Embeddings model  | all-MiniLM-L6-v2                  |
| Ambiente          | Google Colab                      |

### Dataset
| Fonte                               | Registros | Idioma |
|-------------------------------------|-----------|--------|
| Sintético (protocolos SBG/FEBRASGO) | 35        | pt-BR  |
| MedQuAD - NIH                       | ~281      | en     |
| Menstrual Health Awareness Dataset  | ~516      | en     |
| **Total**                           | **832**   |        |

### Pipeline RAG
\`\`\`
Pergunta → [Triagem] → [Busca ChromaDB] → [LLM] → [Resposta formatada]
\`\`\`

### Nós LangGraph
- **triagem**: classifica urgência (alta/média/baixa) por palavras-chave
- **busca_rag**: recupera top-3 documentos por similaridade semântica
- **gerar_resposta**: envia contexto + pergunta ao LLM com prompt médico
- **formatar_saida**: exibe resposta com emoji de urgência

### Como executar
1. Abra o notebook no Google Colab
2. Configure OPENROUTER_API_KEY nos Secrets do Colab
3. Execute as células em ordem (1 a 16)

### Observação sobre Fine-Tuning
A abordagem RAG como alternativa ao fine-tuning tradicional. 
RAG foi escolhido por ser a abordagem
padrão do mercado para domínios especializados, não exigir GPU dedicada
e permitir atualização do conhecimento sem retreinamento.
