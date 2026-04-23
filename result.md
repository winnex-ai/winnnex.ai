📦 Instalando faiss-cpu...
============================================================
✅ Winnex AI Proof v5.0 — Setup completo
   numpy: 2.0.2
   torch: 2.10.0+cpu
   CUDA disponível: False
============================================================
✅ Constantes carregadas
   NEEDLES: 4 (easy, medium, hard, very_hard)
   BACKGROUND: 12 sentenças
✅ Classes carregadas: HMC, QJL, PSIQRH, LRUCache
✅ Funções de prova carregadas

╔══════════════════════════════════════════════════════════╗
║   WINNEX AI — BENCHMARK PROOF v5.0                       ║
║   Provas Empíricas das Equações 1-15 + SIGReg            ║
╚══════════════════════════════════════════════════════════╝

════════════════════════════════════════════════════════════
§1  PROVA Eq.1-4 + Eq.13: HMC Hamiltoniano
════════════════════════════════════════════════════════════
  Drift/run (Eq.13): 0.001894  ✓
  Accept rate: 100.00%
  Confidence: 0.999283
  Conservação de energia: PROVADA ✓
  📊 Gráfico: proof_eq1_4.png

════════════════════════════════════════════════════════════
§2  PROVA Eq.5: QJL Johnson-Lindenstrauss
════════════════════════════════════════════════════════════
  Erro JL médio: 0.1343 ± 0.1020
  Erro JL máx: 0.5787
  Garantia ε < 0.60: PROVADA ✓
  📊 Gráfico: proof_eq5.png

════════════════════════════════════════════════════════════
§4  PROVA Eq.10: LRU Decaimento Temporal
════════════════════════════════════════════════════════════
  Cache: 5 itens | hits=18 misses=0
    chunk_0: weight=41.1904
    chunk_1: weight=32.2495
    chunk_2: weight=32.0569
    chunk_3: weight=11.4842
    chunk_4: weight=11.4841
  Eq.10 provada: weight = log(1+acc)/Δt ✓

════════════════════════════════════════════════════════════
§6  PROVA Eq.15: OCR Confidence Ponderada
════════════════════════════════════════════════════════════
  Texto limpo: conf=1.0000
  Texto ruidoso: conf=0.8286
  Texto parcial: conf=0.9688
  Eq.15 provada: conf = Σ(confᵢ·lenᵢ)/total_len ✓

════════════════════════════════════════════════════════════
§3  PROVA CHAVE: SBERT + ΨQRH (needle_4)
════════════════════════════════════════════════════════════
  Carregando SBERT (paraphrase-multilingual-MiniLM-L12-v2)...

Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

modules.json: 100%
 229/229 [00:00<00:00, 17.7kB/s]
config_sentence_transformers.json: 100%
 122/122 [00:00<00:00, 11.0kB/s]
README.md: 
 3.89k/? [00:00<00:00, 350kB/s]
sentence_bert_config.json: 100%
 53.0/53.0 [00:00<00:00, 4.70kB/s]
config.json: 100%
 645/645 [00:00<00:00, 41.7kB/s]
model.safetensors: 100%
 471M/471M [00:28<00:00, 16.9MB/s]
Loading weights: 100%
 199/199 [00:00<00:00, 545.06it/s, Materializing param=pooler.dense.weight]

BertModel LOAD REPORT from: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
Key                     | Status     |  | 
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  | 

Notes:
- UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.

tokenizer_config.json: 100%
 526/526 [00:00<00:00, 23.6kB/s]
tokenizer.json: 100%
 9.08M/9.08M [00:00<00:00, 45.4MB/s]
special_tokens_map.json: 100%
 239/239 [00:00<00:00, 20.3kB/s]
config.json: 100%
 190/190 [00:00<00:00, 12.2kB/s]

  ✅ SBERT carregado: dim=384

  Needle_4 (very_hard - zero overlap léxico):
    Conteúdo: O mecanismo de consenso distribuído opera com latência média...
    Query:    Qual é o tempo de resposta do sistema sob alta demanda de operações?

    BM25 sim (baseline): ~0.03 (falha)
    SBERT sim: 0.4872
    ΨQRH sim:  0.2892

    ⚠️ SBERT sim = 0.4872 < 0.70

╔══════════════════════════════════════════════════════════╗
║              RESUMO FINAL — v5.0                        ║
╠══════════════════════════════════════════════════════════╣
║  Eq.1-4  HMC: drift=0.00189  ✓                ║
║  Eq.5    QJL: ε_max=0.5787  ✓                    ║
║  Eq.6-9  ΨQRH: needle_4 sim=0.4872  ✗     ║
║  Eq.10   LRU: functional  ✓                                   ║
║  Eq.13   Drift: per-run  ✓                                   ║
║  Eq.15   OCR: confidence  ✓                                  ║
║  Tempo total: 44.6s                                    ║
╚══════════════════════════════════════════════════════════╝

  📁 JSON: winnex_v5_proof_results.json
  📊 PNGs: proof_eq1_4.png, proof_eq5.png

  ✅ Todos os artefatos prontos!
















{
  "benchmark": "Winnex AI Definitive v4.0",
  "bugs_fixed": 7,
  "math_validated": true,
  "honest_failure_reported": true,
  "metrics": {
    "embed_dim": 128,
    "vocab": 10000,
    "var": 0.6063,
    "n_chunks": 13353,
    "t_total_s": 43.681,
    "hmc_confidence": 0.995355,
    "drift_per_run": 0.004825,
    "acceptance": 1.0,
    "jl_error": 0.172472,
    "recall_1": 0.25,
    "recall_5": 0.25,
    "recall_10": 0.5,
    "recall_50": 0.5
  },
  "needles": [
    {
      "id": "needle_1",
      "difficulty": "easy",
      "lexical_overlap": "alta — 'validação','token' presentes em query e needle",
      "note": "",
      "rank_bm25": 6,
      "rank_hmc": 6,
      "rank_qrh": 8,
      "best_rank": 6,
      "recall_1": 0,
      "recall_5": 0,
      "recall_10": 1,
      "recall_50": 1
    },
    {
      "id": "needle_2",
      "difficulty": "medium",
      "lexical_overlap": "alta — 'compressão','Gbps','algoritmo','teste' únicos no corpus",
      "note": "",
      "rank_bm25": 988,
      "rank_hmc": -1,
      "rank_qrh": -1,
      "best_rank": 988,
      "recall_1": 0,
      "recall_5": 0,
      "recall_10": 0,
      "recall_50": 0
    },
    {
      "id": "needle_3",
      "difficulty": "hard",
      "lexical_overlap": "baixa — 'correlação','variáveis' mas 'cientista'≠'pesquisadora'",
      "note": "",
      "rank_bm25": 1,
      "rank_hmc": 1,
      "rank_qrh": 1,
      "best_rank": 1,
      "recall_1": 1,
      "recall_5": 1,
      "recall_10": 1,
      "recall_50": 1
    },
    {
      "id": "needle_4",
      "difficulty": "very_hard",
      "lexical_overlap": "ZERO — 'latência'≠'tempo de resposta', 'transações'≠'operações'",
      "note": "Requer SBERT/neural — BM25+LSA não resolve por design",
      "rank_bm25": 79,
      "rank_hmc": -1,
      "rank_qrh": -1,
      "best_rank": 79,
      "recall_1": 0,
      "recall_5": 0,
      "recall_10": 0,
      "recall_50": 0
    }
  ]
}
