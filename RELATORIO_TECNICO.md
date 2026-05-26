# Relatório Técnico — Tech Challenge Fase 3
## Assistente Médico de Saúde da Mulher: Fine-Tuning QLoRA + RAG + LangGraph

**Data:** Maio de 2026  
**Modelo base:** `google/gemma-2-2b-it`  
**Método de ajuste fino:** QLoRA (Quantized Low-Rank Adaptation)  
**Arquitetura de orquestração:** LangGraph + ChromaDB (RAG)

---

## 1. Descrição do Assistente Médico Criado

O assistente foi concebido como uma ferramenta de apoio a profissionais de saúde, especializado em **saúde da mulher**. Seu objetivo é responder perguntas clínicas em português brasileiro, recuperar contexto de uma base de conhecimento médico e classificar a urgência de cada consulta.

### Capacidades principais

| Capacidade | Descrição |
|---|---|
| Triagem de urgência | Classifica automaticamente cada pergunta em baixa, média ou alta urgência por meio de palavras-chave clínicas |
| Busca semântica (RAG) | Recupera os 3 trechos mais relevantes do ChromaDB usando embeddings `all-MiniLM-L6-v2` |
| Geração contextualizada | Produz respostas fundamentadas no contexto recuperado, usando o modelo Gemma-2-2B ajustado |
| Domínio restrito | Orientado a ginecologia, obstetrícia, endocrinologia feminina e saúde reprodutiva |

### System prompt

```
Você é um assistente médico especializado em saúde da mulher.
Responda sempre em português brasileiro, de forma clara, empática e
baseada em evidências. Nunca prescreva medicamentos diretamente:
sempre oriente a validação por um profissional de saúde.
```

O prompt de sistema é embutido no turno do usuário porque o chat template do Gemma-2 não aceita a role `system` separada.

---

## 2. Processo de Fine-Tuning

### 2.1 Composição do Dataset

Três fontes foram combinadas para formar o corpus de treinamento:

| Fonte | Idioma | Descrição |
|---|---|---|
| **MedQuAD** (NIH) | Inglês | Pares Q&A médicos reais de agências federais americanas |
| **Menstrual Health** (HuggingFace) | Inglês | Dataset especializado em saúde menstrual |
| **Dataset sintético pt-BR** | Português | Pares Q&A criados internamente sobre saúde da mulher |

### 2.2 Pré-processamento e Anonimização

#### Limpeza textual

- Remoção de tags HTML
- Normalização Unicode (NFKC)
- Colapso de espaços múltiplos e quebras de linha excessivas

#### Anonimização de PII (dados pessoais)

A anonimização foi aplicada em duas etapas:

**Etapa 1 — Regex para identificadores estruturados:**

| Tag substituída | Padrão detectado |
|---|---|
| `[CPF]` | XXX.XXX.XXX-XX |
| `[TELEFONE]` | Formatos nacionais com/sem DDI |
| `[EMAIL]` | Endereços de e-mail |
| `[DATA]` | Datas no formato DD/MM/AAAA |
| `[PRONTUARIO]` | Referências a prontuário/matrícula + número |
| `[CNS]` | Cartão Nacional de Saúde (formato 15 dígitos) |

**Etapa 2 — NER com spaCy (`pt_core_news_sm`):**

- Entidades `PER` (pessoas) → substituídas por `[NOME]`
- Entidades `LOC`/`GPE` (locais) → substituídas por `[LOCAL]`

#### Curadoria de qualidade

| Filtro | Critério |
|---|---|
| Pergunta muito curta | < 10 caracteres → removido |
| Resposta muito curta | < 80 caracteres → removido |
| Deduplicação | Perguntas normalizadas idênticas → mantém apenas o primeiro |
| Truncamento | Respostas > 1500 caracteres são cortadas no limite de sentença |

### 2.3 Formato de Instrução

O dataset curado foi convertido para o formato de mensagens (chat template do Gemma-2):

```
[Usuário]
<system_prompt>

[Categoria: <categoria> | Urgência: <nível>]
<pergunta>

[Assistente]
<resposta>
```

**Divisão treino/validação:** 85% treino / 15% validação (`random_state=42`)

### 2.4 Configuração do Modelo Base e Quantização (QLoRA)

**Modelo base:** `google/gemma-2-2b-it` (2B parâmetros, instruction-tuned)

**Quantização 4-bit com BitsAndBytes:**

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # Normal Float 4
    bnb_4bit_compute_dtype=float16,     # fp16 em GPU T4; bf16 em A100/L4
    bnb_4bit_use_double_quant=True,     # quantização aninhada para economizar VRAM
)
```

A double quantization reduz o consumo de VRAM em ~0,4 bits por parâmetro adicionalmente.

### 2.5 Adaptadores LoRA

Os adaptadores LoRA foram inseridos em todas as projeções de atenção e nas camadas do bloco MLP:

```python
LoraConfig(
    r=16,                    # rank dos adaptadores
    lora_alpha=32,           # escala efetiva = alpha / r = 2
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",   # atenção
        "gate_proj", "up_proj", "down_proj"         # MLP (SwiGLU)
    ],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
```

Com rank 16 e os 7 módulos alvo, apenas uma fração pequena dos parâmetros originais foi treinada, mantendo o modelo base congelado.

### 2.6 Hiperparâmetros de Treinamento

| Parâmetro | Valor | Observação |
|---|---|---|
| Épocas | 3 | |
| Batch por device | 2 | Reduzido de 4 para economizar VRAM |
| Gradient accumulation | 8 | Batch efetivo = 2 × 8 = **16** |
| Learning rate | 2e-4 | |
| LR scheduler | cosine | |
| Warmup ratio | 0.1 | |
| Weight decay | 0.01 | |
| Optimizer | paged_adamw_8bit | Otimizador paginado para menor uso de VRAM |
| Max sequence length | 512 | Reduzido de 1024 para economizar VRAM |
| Gradient checkpointing | True | `use_reentrant=False` |
| Avaliação | A cada 50 steps | |
| Checkpoint salvo | A cada 50 steps (máx. 2) | Carrega o melhor ao final |
| Métrica de seleção | `eval_loss` | |
| Trainer | `SFTTrainer` (TRL) | Supervised Fine-Tuning |

---

## 3. Diagrama do Fluxo LangGraph

O assistente é orquestrado por um grafo linear com 4 nós. O fluxo é o mesmo na versão com OpenRouter (Etapa 3) e na versão com o modelo fine-tuned local (Etapa 4 — Célula 28).

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │                    EstadoConsulta (TypedDict)                        │
 │   pergunta | urgencia | contexto | resposta | encaminhamento         │
 └─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────┐
              │        no_triagem         │
              │  Classifica urgência por  │
              │  palavras-chave clínicas  │
              │                           │
              │  ALTA  → hemorragia,      │
              │          choque, sepse... │
              │  MÉDIA → febre, dor,      │
              │          infecção...      │
              │  BAIXA → demais           │
              └────────────┬──────────────┘
                           │
                           ▼
              ┌───────────────────────────┐
              │       no_busca_rag        │
              │  Embedding da pergunta    │
              │  (all-MiniLM-L6-v2)       │
              │         │                 │
              │  Busca nos vetores do     │
              │  ChromaDB (top-3)         │
              │         │                 │
              │  Retorna trechos com      │
              │  [Fonte | Categoria]      │
              └────────────┬──────────────┘
                           │
                           ▼
              ┌───────────────────────────┐
              │    no_gerar_resposta      │
              │                           │
              │  [versão OpenRouter]      │
              │  LangChain ChatOpenAI     │
              │  via ChatPromptTemplate   │
              │                           │
              │  [versão fine-tuned]      │
              │  Gemma-2-2B-it + LoRA     │
              │  HuggingFacePipeline      │
              │  max_new_tokens=400       │
              │  repetition_penalty=1.3   │
              │                           │
              │  Se urgência=ALTA:        │
              │  injeta alerta no prompt  │
              └────────────┬──────────────┘
                           │
                           ▼
              ┌───────────────────────────┐
              │     no_formatar_saida     │
              │  Exibe nível de urgência  │
              │  com emoji colorido       │
              │  🟢 BAIXA / 🟡 MÉDIA      │
              │  🔴 ALTA                  │
              └────────────┬──────────────┘
                           │
                           ▼
                         END
```

### Componentes de infraestrutura

```
Pergunta do usuário
        │
        ▼
  ┌─────────────────────────────────────────────────────┐
  │                  LangGraph Runtime                   │
  │                                                     │
  │  ┌──────────┐   ┌──────────┐   ┌─────────────────┐ │
  │  │ Triagem  │──▶│  RAG     │──▶│ Geração (LLM)   │ │
  │  │ (regras) │   │ChromaDB  │   │ Gemma-2-2B-it   │ │
  │  └──────────┘   │MiniLM-L6 │   │ + LoRA adapters │ │
  │                 └──────────┘   └─────────────────┘ │
  └─────────────────────────────────────────────────────┘
        │
        ▼
  Resposta formatada com urgência e encaminhamento
```

---

## 4. Avaliação do Modelo e Análise dos Resultados

### 4.1 Curva de Loss

![Curva de Loss — Fine-Tuning QLoRA (Gemma-2-2B)](curva_loss.png)

A curva de loss apresentou comportamento esperado para um ajuste fino supervisionado:

| Fase | Steps | Loss de treino | Loss de validação |
|---|---|---|---|
| Descida abrupta | 0–20 | 3,1 → 1,2 | — |
| Convergência rápida | 20–50 | 1,2 → 0,65 | 0,63 (primeiro checkpoint) |
| Platô estável | 50–100 | 0,65 → 0,45 | 0,60 |

**Observações:**
- A ausência de overfitting é evidenciada pela proximidade entre loss de treino e de validação ao longo de todo o treinamento.
- O platô a partir do step 50 indica que o modelo assimilou os padrões do domínio sem memorizar os exemplos individuais.
- A validação estabiliza levemente acima do treino, comportamento típico e saudável.

### 4.2 Avaliação Qualitativa

#### Pergunta 1 — Sintomas da SOP (dentro do domínio, pt-BR)

| Aspecto | Modelo Base | Fine-Tuned |
|---|---|---|
| Cobertura de sintomas | Parcial (5-6 itens) | Abrangente (>10 itens) |
| Terminologia clínica | Genérica | Específica (oligomenorreia, amenorreia, hirsutismo) |
| Tom | Genérico com emojis | Mais clínico e direto |
| Qualidade geral | Adequada para leigo | Mais adequada para profissional |

#### Pergunta 2 — Hemorragia pós-parto (alta urgência, pt-BR)

| Aspecto | Modelo Base | Fine-Tuned |
|---|---|---|
| Reconhecimento da urgência | Parcial ("procure ajuda") | Presente, com menção ao SAMU/192 |
| Conduta clínica imediata | Ausente | Abordada superficialmente |
| Problema identificado | Resposta muito genérica | Placeholders `[LOCAL]`, `[NOME]` na saída; divagação |

#### Pergunta 3 — Contraceptivos para hipertensas (pt-BR)

| Aspecto | Modelo Base | Fine-Tuned |
|---|---|---|
| Recomendação de consulta | Sim | Sim |
| Métodos listados | Não | Sim (injetáveis, DIU, implante) |
| Coerência | Alta | Média (mistura com placeholders) |

#### Pergunta 4 — Sintomas de cólica menstrual (inglês)

| Aspecto | Modelo Base | Fine-Tuned |
|---|---|---|
| Idioma de resposta | Português (ignora inglês) | Inglês (respeita o idioma da pergunta) |
| Abrangência | Moderada | Alta (lista extensa de sintomas associados) |
| Problema | — | Tendência à hiperlista (acumula itens sem filtro de relevância) |

#### Pergunta 5 — Investimento na bolsa (fora do domínio)

| Aspecto | Modelo Base | Fine-Tuned |
|---|---|---|
| Recusa de domínio | Sim (educadamente) | Parcial: responde sobre investimentos e depois retorna ao médico |
| Alinhamento ao domínio | Mantido | Comprometido — não recusa com clareza |

**Avaliação geral qualitativa:** O modelo fine-tuned demonstra domínio clínico superior ao modelo base em perguntas dentro do escopo da saúde da mulher. Os principais problemas identificados são: (1) aparecimento de placeholders de anonimização nas respostas geradas, (2) dificuldade em recusar perguntas fora do domínio, e (3) tendência à verbosidade e à alucinação de detalhes em respostas longas.

### 4.3 Avaliação Quantitativa — Métricas ROUGE

Avaliação realizada sobre **15 amostras do conjunto de validação**:

| Métrica | Modelo Base | Fine-Tuned | Ganho relativo |
|---|---|---|---|
| **ROUGE-1** | 0.0722 | **0.2976** | +312% |
| **ROUGE-2** | 0.0091 | **0.0963** | +958% |
| **ROUGE-L** | 0.0430 | **0.2128** | +395% |
| **ROUGE-Lsum** | 0.0606 | **0.2199** | +263% |

#### Interpretação

- **ROUGE-1 (unigramas):** O modelo fine-tuned usa vocabulário muito mais próximo ao das respostas de referência, confirmando a assimilação do léxico médico do domínio.
- **ROUGE-2 (bigramas):** O salto de +958% é o indicador mais expressivo. Bigramas são sensíveis à coerência local da linguagem — o fine-tuned reproduz sequências de termos médicos corretos, enquanto o modelo base raramente alinha dois tokens seguidos com a referência.
- **ROUGE-L (subsequência mais longa):** Confirma que as respostas do fine-tuned têm estrutura narrativa mais próxima às referências, não apenas vocabulário similar.

> **Limitação:** O conjunto de avaliação é pequeno (n=15). Uma avaliação robusta exigiria ao menos 100–200 amostras e métricas complementares (BERTScore, G-Eval, avaliação humana cega).

### 4.4 Avaliação no Fluxo LangGraph Integrado (Célula 28)

Os testes com o modelo fine-tuned integrado ao grafo mostraram comportamento diferenciado por nível de urgência:

| Pergunta | Urgência detectada | Contexto RAG | Comportamento do LLM |
|---|---|---|---|
| Vacinas para mulheres adultas | BAIXA | 2472 chars recuperados | Resposta longa, coerente, com placeholders de anonimização |
| Critérios de vaginose bacteriana | MÉDIA | 2208 chars recuperados | Critérios diagnósticos corretos listados; mistura com informações não relacionadas no final |
| Hemorragia pós-parto | ALTA | 2349 chars recuperados | Resposta fragmentada; não segue conduta de urgência esperada |

**Diagnóstico:** As respostas de urgência alta são as mais afetadas pela verbosidade e pela confusão de contexto, provavelmente por dois fatores: (1) o alerta de urgência no prompt desestabiliza o modelo, e (2) o RAG recupera trechos genéricos que diluem o sinal clínico específico.

---

## 5. Limitações Identificadas e Recomendações

### Limitações

| Problema | Causa provável | Impacto |
|---|---|---|
| Placeholders `[NOME]`, `[LOCAL]` nas respostas | Modelo aprendeu a reproduzir os tokens de anonimização do treino | Médio — degrada apresentação |
| Respostas fora de domínio não recusadas | Dataset de treino sem exemplos negativos (out-of-scope) | Alto — risco de informação incorreta |
| Verbosidade excessiva | Respostas de treino longas + `max_new_tokens=400` | Baixo — afeta usabilidade |
| Avaliação ROUGE em n=15 | Amostra pequena por limitação de tempo de inferência em GPU T4 | Médio — limita generalização dos resultados |

### Recomendações para iterações futuras

1. **Filtrar placeholders do dataset de treino:** pós-processar respostas para substituir `[NOME]`/`[LOCAL]` por variações neutras antes de alimentar o SFTTrainer.
2. **Adicionar exemplos de recusa:** incluir pares de pergunta fora de domínio com resposta padrão de recusa educada.
3. **Reduzir `max_new_tokens`:** limitar a 200–250 tokens para respostas mais concisas.
4. **Avaliação expandida:** executar ROUGE sobre o conjunto de validação completo e adicionar BERTScore multilíngue.
5. **Aumentar o rank LoRA para r=32:** potencialmente melhora o alinhamento em perguntas de alta complexidade clínica, ao custo de +25–30% de VRAM.

---

## 6. Resumo do Stack Técnico

| Componente | Tecnologia |
|---|---|
| Modelo base | `google/gemma-2-2b-it` |
| Fine-tuning | QLoRA (PEFT + BitsAndBytes + TRL SFTTrainer) |
| Quantização | NF4 4-bit com double quantization |
| Orquestração | LangGraph (StateGraph linear) |
| Recuperação de contexto | ChromaDB + `all-MiniLM-L6-v2` |
| LLM original (Etapa 3) | OpenRouter via LangChain ChatOpenAI |
| LLM fine-tuned (Etapa 4) | HuggingFacePipeline embutido no LangGraph |
| Anonimização | Regex + spaCy `pt_core_news_sm` |
| Avaliação quantitativa | ROUGE (biblioteca `evaluate` da HuggingFace) |
| Ambiente de treinamento | Google Colab (GPU T4 / 15GB VRAM) |
| Armazenamento dos pesos | Google Drive + Hugging Face Hub |
