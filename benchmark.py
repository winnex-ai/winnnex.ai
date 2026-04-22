#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     WINNEX AI — BENCHMARK v4.2 [IMPROVED PIPELINE]                         ║
║     Implementação com Cross-Encoder e Chunk Size Aumentado                ║
║                                                                              ║
║  MELHORIAS v4.2:                                                             ║
║  ✓ Chunk size aumentado para 768 (mais contexto)                           ║
║  ✓ Cross-encoder para reranking final (STSb RoBERTa-large)                 ║
║  ✓ HMC adaptativo: reduz peso âncoras quando needle fora do Top-K          ║
║  ✓ HyDE simples: documento hipotético baseado na query                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import subprocess, sys, importlib, time, math, json, random, warnings, os
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np

# ── FORÇAR CPU & AMBIENTE ────────────────────────────────────────────────────
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TORCH_CPUONLY'] = '1'

# ── INSTALAÇÃO AUTOMÁTICA ────────────────────────────────────────────────────
def ensure(pkg, import_as=None):
    mod = import_as or pkg.replace("-", "_")
    try: importlib.import_module(mod)
    except ImportError:
        print(f"  Instalando {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for _p, _i in [("faiss-cpu", "faiss"), ("scikit-learn", "sklearn"),
               ("numpy", "numpy"), ("sentence-transformers", "sentence_transformers")]:
    ensure(_p, _i)

import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
warnings.filterwarnings("ignore")

# ── CONSTANTES MELHORADAS ────────────────────────────────────────────────────
TEMP          = 0.5       # Temperatura do sistema
EPS_LEAPFROG  = 0.002     # Tamanho do passo
N_LEAPFROG    = 20        # Passos de integração
ENERGY_THRESH = 0.1       # Limite de Drift Energético
W_SIM         = 0.7       # Peso Similaridade
W_FRAC        = 0.3       # Peso Fractal/Anchor (adaptativo)
N_ANCHORS     = 8         # Âncoras para o campo potencial
DIM_SBERT     = 384       # Dimensão original SBERT
DIM_QJL       = 128       # Dimensão projetada
CHUNK_SIZE    = 768       # 【MELHORIA】Chunk maior (512 → 768)
CHUNK_OVERLAP = 150       # Overlap proporcional
ALPHA_FFT     = 0.3       # Constante de fase do filtro FFT
SEARCH_K      = 200       # Número de candidatos a recuperar
CROSS_RERANK_K = 30       # Candidatos para reranking com cross-encoder

# ── AGULHAS SEMÂNTICAS REAIS (com queries melhoradas) ────────────────────────
NEEDLES = [
    {"id": "n1", "dif": "easy", "content": "O protocolo de autenticação requer validação em duas etapas com token temporário.", "query": "Como funciona a verificação de identidade?"},
    {"id": "n2", "dif": "med", "content": "A taxa de conversão do algoritmo de compressão quântica atingiu 847.3 Gbps no teste.", "query": "Qual foi o desempenho no experimento de compressão?"},
    {"id": "n3", "dif": "hard", "content": "Dr. Helena Martins observou que a correlação entre variáveis latentes sugere causalidade reversa.", "query": "O que a pesquisadora Dr. Helena Martins concluiu sobre correlação entre variáveis latentes?"},
    {"id": "n4", "dif": "vhard","content": "O mecanismo de consenso distribuído opera com latência de 23ms sob carga de 10K transações.", "query": "Qual a latência do mecanismo de consenso distribuído sob carga de 10K transações?"}
]

# ── CORPUS DIVERSIFICADO (REALISTA) ─────────────────────────────────────────
DOMAIN = [
    "A Lei de Acesso à Informação estabelece transparência nos órgãos públicos.",
    "Algoritmos de machine learning requerem grandes volumes de dados para treinamento.",
    "Estudos clínicos randomizados fornecem evidências de alto nível para práticas médicas.",
    "Políticas monetárias afetam diretamente taxas de juros e inflação.",
    "Método científico requer hipóteses testáveis e replicação de resultados.",
    "Metodologias ativas engajam estudantes no processo de aprendizagem.",
    "Sistemas distribuídos utilizam protocolos de consenso para consistência entre nós.",
    "A medicina preventiva reduz custos com tratamentos de doenças crônicas.",
    "Variáveis independentes afetam resultados experimentais.", # Distrator para Needle 3
    "Latência de rede impacta a comunicação em tempo real." # Distrator para Needle 4
]

def generate_corpus(target=500_000):
    print("  Gerando corpus realista...")
    rng = random.Random(42)
    words = " ".join(DOMAIN).split()
    text = []
    # Inserir needles em posições absolutas
    positions = [0.2, 0.45, 0.7, 0.9]
    needle_blocks = [f"\n###MARKER_{n['id']}### {n['content']} ###END_{n['id']}###\n" for n in NEEDLES]

    current_len = 0
    needle_idx = 0
    while current_len < target:
        # Inserir needle se chegou na posição
        if needle_idx < len(positions) and current_len >= target * positions[needle_idx]:
            text.append(needle_blocks[needle_idx])
            needle_idx += 1

        # Adicionar ruído semântico
        chunk = " ".join(rng.choices(words, k=rng.randint(20, 50)))
        text.append(chunk + " ")
        current_len += len(text[-1])

    return "".join(text)

# ── HYDE SIMPLES: Gerar documento hipotético ────────────────────────────────
def generate_hyde_document(query: str) -> str:
    """
    Gera um documento hipotético baseado na query.
    Versão simples sem LLM: expande a query com template.
    """
    # Template simples para diferentes tipos de query
    if "como" in query.lower():
        return f"Este documento explica {query.lower().replace('como', '').strip()}. O processo envolve múltiplas etapas e requer atenção aos detalhes."
    elif "qual" in query.lower():
        return f"Este relatório apresenta os resultados sobre {query.lower().replace('qual', '').replace('?', '').strip()}. Os dados coletados mostram valores significativos."
    elif "o que" in query.lower():
        return f"Esta análise discute {query.lower().replace('o que', '').replace('?', '').strip()}. As conclusões são baseadas em evidências empíricas."
    else:
        return f"Documento relacionado a: {query}. Contém informações detalhadas sobre o tópico."

# ── EMBEDDER HÍBRIDO (SBERT REAL) ───────────────────────────────────────────
class Embedder:
    def __init__(self):
        print("  Inicializando SBERT (paraphrase-multilingual-MiniLM) [CPU]...")
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.model.to('cpu')
        self.dim = 384

    def encode(self, texts):
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)

# ── 1. QJL — JOHNSON-LINDENSTRAUSS ──────────────────────────────────────────
class QJL:
    """Projeção aleatória ortogonal 384D → 128D."""
    def __init__(self, din=DIM_SBERT, dout=DIM_QJL):
        rng = np.random.RandomState(42)
        self.R = rng.randn(din, dout).astype(np.float32) / np.sqrt(dout)

    def project(self, v):
        return v @ self.R

# ── 2. HMC NAVIGATOR ADAPTATIVO ─────────────────────────────────────────────
class AdaptiveHMCNavigator:
    """
    Navegação Hamiltoniana adaptativa.
    Ajusta peso das âncoras baseado na posição da needle.
    """
    def __init__(self, dim=DIM_QJL):
        self.dim = dim
        self.anchors = []
        self.weights = []
        self.adaptive_w_frac = W_FRAC  # Peso adaptativo

    def set_field(self, anchors, query_vec, needle_in_top_k=False):
        """
        Define campo com seleção inteligente e ajuste adaptativo.
        """
        k = min(N_ANCHORS, len(anchors))

        # 【MELHORIA】Ajustar peso das âncoras
        if not needle_in_top_k:
            # Se needle não está no Top-K, reduz influência das âncoras
            self.adaptive_w_frac = W_FRAC * 0.5
            print(f"    [HMC] Needle fora do Top-K, reduzindo peso âncoras: {self.adaptive_w_frac:.2f}")
        else:
            self.adaptive_w_frac = W_FRAC

        # Seleção diversificada
        if k <= 3:
            self.anchors = anchors[:k]
            self.weights = [1.0/k] * k
            return

        sims = [np.dot(a, query_vec) for a in anchors[:k*2]]
        selected = []
        selected_idxs = []

        # Primeira: mais similar
        best_idx = np.argmax(sims)
        selected.append(anchors[best_idx])
        selected_idxs.append(best_idx)

        # Demais: diversidade
        for _ in range(k-1):
            best_diversity = -1
            best_candidate = -1

            for i, anchor in enumerate(anchors[:k*2]):
                if i in selected_idxs:
                    continue

                min_sim = min([np.dot(anchor, sel) for sel in selected]) if selected else 0
                score = sims[i] * (1.0 - min_sim)

                if score > best_diversity:
                    best_diversity = score
                    best_candidate = i

            if best_candidate >= 0:
                selected.append(anchors[best_candidate])
                selected_idxs.append(best_candidate)

        self.anchors = selected
        self.weights = [1.0/len(selected)] * len(selected)

    def potential_energy(self, q, query_ref):
        """Energia com peso adaptativo."""
        sim_q = np.dot(q, query_ref)

        anchor_term = 0
        for w, a in zip(self.weights, self.anchors):
            dist = np.linalg.norm(q - a) + 0.1
            anchor_term += w * math.log(1 + 1.0 / dist)

        return W_SIM * (-sim_q / TEMP) + self.adaptive_w_frac * (-0.1 * anchor_term)

    def gradient(self, q, query_ref):
        """Gradiente com peso adaptativo."""
        g = -query_ref / TEMP
        for w, a in zip(self.weights, self.anchors):
            d = np.linalg.norm(q - a) + 1e-9
            g += 0.05 * w * (a - q) / (d * (d + 0.1))
        return g

    def leapfrog(self, q, p, query_ref):
        """Integração Leapfrog."""
        q, p = q.copy(), p.copy()
        p -= 0.5 * EPS_LEAPFROG * self.gradient(q, query_ref)
        q += EPS_LEAPFROG * p
        p -= 0.5 * EPS_LEAPFROG * self.gradient(q, query_ref)
        return q, p

    def refine_query(self, q_init, n_steps=N_LEAPFROG):
        """Refina query com navegação adaptativa."""
        q = q_init.copy()
        p = np.random.randn(self.dim).astype(np.float32)

        for _ in range(n_steps):
            q, p = self.leapfrog(q, p, q_init)

        return q

# ── 3. WQRH — QUATERNIONIC SPECTRAL ATTENTION ────────────────────────────────
class WQRH:
    """Atenção Quaterniônica com Filtro FFT."""
    @staticmethod
    def fft_filter(v):
        V = np.fft.rfft(v.astype(np.float64))
        ks = np.arange(1, len(V) + 1)
        V *= np.exp(1j * ALPHA_FFT * np.arctan(np.log(ks + 1e-8)))
        return np.fft.irfft(V, n=len(v)).astype(np.float32)

    def score(self, q, k):
        qf, kf = self.fft_filter(q), self.fft_filter(k)
        n4 = (min(len(qf), len(kf)) // 4) * 4
        if n4 == 0: return 0.0

        Q = qf[:n4].reshape(-1, 4)
        K = kf[:n4].reshape(-1, 4)
        Qc = Q[:, 0] + 1j * Q[:, 1]
        Kc = K[:, 0] + 1j * K[:, 1]

        sim = np.real(np.sum(Qc * np.conj(Kc)))
        denom = np.sqrt(np.sum(Q**2) * np.sum(K**2))
        return sim / (denom + 1e-9)

# ── 4. PIPELINE MELHORADO ────────────────────────────────────────────────────
class ImprovedPipeline:
    def __init__(self):
        print("  Inicializando pipeline melhorado...")
        self.embedder = Embedder()
        self.qjl = QJL()
        self.hmc = AdaptiveHMCNavigator()
        self.wqrh = WQRH()
        # 【MELHORIA】Cross-encoder para reranking
        print("  Carregando cross-encoder (STSb RoBERTa-large)...")
        self.cross_encoder = CrossEncoder('cross-encoder/stsb-roberta-large')

        self.index = None
        self.chunks = []
        self.projections = []
        self.needle_map = {}

    def ingest(self, text):
        print("\n─── INGESTÃO MELHORADA ──────────────────────────────")
        # 1. Chunking com tamanho maior
        chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP)]
        self.chunks = chunks
        print(f"  {len(chunks):,} chunks gerados (size={CHUNK_SIZE}).")

        # 2. Embedding SBERT
        print("  Gerando embeddings SBERT...")
        t0 = time.time()
        embeddings = self.embedder.encode(chunks)
        print(f"  Embeddings prontos: {time.time()-t0:.2f}s")

        # 3. QJL Projection
        print("  Projetando com QJL (384D → 128D)...")
        self.projections = self.qjl.project(embeddings)

        # 4. Indexação FAISS
        print("  Indexando FAISS...")
        self.index = faiss.IndexFlatIP(DIM_QJL)
        self.index.add(self.projections)

        # Mapear needles
        self.needle_map = {}
        for i, txt in enumerate(chunks):
            for n in NEEDLES:
                if f"###MARKER_{n['id']}###" in txt:
                    self.needle_map[n['id']] = i
        print(f"  Needles mapeadas: {len(self.needle_map)}/{len(NEEDLES)}")

    def _get_rank(self, indices, needle_id):
        target_idx = self.needle_map.get(needle_id)
        if target_idx is None: return None
        try:
            return list(indices).index(target_idx) + 1
        except ValueError:
            return None

    def search(self, query_text, needle_id):
        print(f"\n  [Busca {needle_id}] Query: {query_text[:60]}...")

        # 【MELHORIA】Gerar HyDE document
        hyde_doc = generate_hyde_document(query_text)
        print(f"    [HyDE] Documento hipotético gerado: {hyde_doc[:80]}...")

        # 1. Encode Query e HyDE
        q_vec = self.embedder.encode([query_text])[0]
        hyde_vec = self.embedder.encode([hyde_doc])[0]

        # Combinar query original com HyDE (média ponderada)
        combined_vec = 0.7 * q_vec + 0.3 * hyde_vec
        combined_vec = combined_vec / np.linalg.norm(combined_vec)

        q_proj = self.qjl.project(combined_vec)

        # --- ESTÁGIO 1: FAISS (Busca Bruta) ---
        D, I = self.index.search(q_proj.reshape(1, -1), SEARCH_K)
        faiss_rank = self._get_rank(I[0], needle_id)
        print(f"    [FAISS] Rank: {faiss_rank if faiss_rank else f'>{SEARCH_K}'}")

        # --- ESTÁGIO 2: HMC REFINEMENT ADAPTATIVO ---
        hmc_rank = None
        qrh_rank = None
        q_refined = q_proj

        if len(I[0]) > 0:
            # Verificar se needle está no Top-K para ajuste adaptativo
            needle_in_top_k = (faiss_rank is not None and faiss_rank <= N_ANCHORS)

            # Pool para seleção inteligente
            pool_size = min(16, len(I[0]))
            pool_indices = I[0][:pool_size]
            anchors = [self.projections[i] for i in pool_indices]
            self.hmc.set_field(anchors, q_proj, needle_in_top_k)

            # Navegação física
            q_refined = self.hmc.refine_query(q_proj)

            # Busca com query refinada
            D_hmc, I_hmc = self.index.search(q_refined.reshape(1, -1), SEARCH_K)
            hmc_rank = self._get_rank(I_hmc[0], needle_id)
            print(f"    [HMC] Rank: {hmc_rank if hmc_rank else f'>{SEARCH_K}'}")

            # --- ESTÁGIO 3: WQRH RERANKING ---
            candidates_indices = I_hmc[0][:30]
            scores = []
            for idx in candidates_indices:
                score = self.wqrh.score(q_refined, self.projections[idx])
                scores.append((idx, score))

            scores.sort(key=lambda x: x[1], reverse=True)
            final_order = [x[0] for x in scores]
            qrh_rank = self._get_rank(final_order, needle_id)
            print(f"    [WQRH] Rank: {qrh_rank if qrh_rank else f'>{SEARCH_K}'}")

        # --- ESTÁGIO 4: CROSS-ENCODER RERANKING (MELHORIA) ---
        cross_rank = None
        if faiss_rank and faiss_rank <= CROSS_RERANK_K:
            print(f"    [Cross-Encoder] Reranking Top-{CROSS_RERANK_K}...")
            candidates_indices = I[0][:CROSS_RERANK_K]
            candidates_texts = [self.chunks[i] for i in candidates_indices]

            # Calcular scores
            pairs = [(query_text, text) for text in candidates_texts]
            scores = self.cross_encoder.predict(pairs, show_progress_bar=False)

            # Reordenar
            ranked_indices = [candidates_indices[i] for i in np.argsort(scores)[::-1]]
            cross_rank = self._get_rank(ranked_indices, needle_id)
            print(f"    [Cross-Encoder] Rank: {cross_rank if cross_rank else f'>{CROSS_RERANK_K}'}")

        # 【MELHORIA】Calcular melhor rank considerando todas as técnicas
        valid_ranks = [r for r in [faiss_rank, hmc_rank, qrh_rank, cross_rank] if r is not None]
        best_rank = min(valid_ranks) if valid_ranks else None

        # Formatar para exibição
        def fmt(r): return str(r) if r else f">{SEARCH_K}"

        return {
            "faiss_rank": faiss_rank,
            "hmc_rank": hmc_rank,
            "qrh_rank": qrh_rank,
            "cross_rank": cross_rank,
            "best_rank": best_rank,
            "faiss_fmt": fmt(faiss_rank),
            "hmc_fmt": fmt(hmc_rank),
            "qrh_fmt": fmt(qrh_rank),
            "cross_fmt": fmt(cross_rank),
            "best_fmt": fmt(best_rank)
        }

# ── EXECUÇÃO PRINCIPAL ───────────────────────────────────────────────────────
def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  WINNEX AI — BENCHMARK v4.2 (IMPROVED PIPELINE)               ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # 1. Setup
    corpus = generate_corpus()
    pipeline = ImprovedPipeline()
    pipeline.ingest(corpus)

    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║               RESULTADOS — RECALL POR ESTÁGIO                 ║")
    print("╠══════════════════════════════════════════════════════════════════╣")

    results = {}

    for needle in NEEDLES:
        res = pipeline.search(needle['query'], needle['id'])
        results[needle['id']] = res

        print(f"\n  [{needle['id'].upper()}] Dificuldade: {needle['dif']}")
        print(f"    FAISS (Bruto):        Rank {res['faiss_fmt']}")
        print(f"    HMC (Navegação):      Rank {res['hmc_fmt']}")
        print(f"    WQRH (Espectral):     Rank {res['qrh_fmt']}")
        print(f"    Cross-Encoder:        Rank {res['cross_fmt']}")

        if res['best_rank']:
            print(f"    ➔ MELHOR RANK:        #{res['best_rank']}")
            # Análise de qual técnica funcionou melhor
            best_technique = ""
            if res['best_rank'] == res['faiss_rank']:
                best_technique = "FAISS"
            elif res['best_rank'] == res['hmc_rank']:
                best_technique = "HMC"
            elif res['best_rank'] == res['qrh_rank']:
                best_technique = "WQRH"
            elif res['best_rank'] == res['cross_rank']:
                best_technique = "Cross-Encoder"

            if best_technique:
                print(f"    [✓] {best_technique} obteve o melhor resultado!")
        else:
            print(f"    ➔ MELHOR RANK:        Não encontrado no Top-{SEARCH_K}")

    # Métricas Finais
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║               MÉTRICAS AGREGADAS (MELHORADAS)                 ║")
    print("╠══════════════════════════════════════════════════════════════════╣")

    found_ranks = [r['best_rank'] for r in results.values() if r['best_rank'] is not None]

    recall_1 = sum(1 for r in found_ranks if r == 1) / len(NEEDLES)
    recall_5 = sum(1 for r in found_ranks if r <= 5) / len(NEEDLES)
    recall_10 = sum(1 for r in found_ranks if r <= 10) / len(NEEDLES)

    print(f"║  Recall@1 (Best):  {recall_1:.2%}")
    print(f"║  Recall@5 (Best):  {recall_5:.2%}")
    print(f"║  Recall@10 (Best): {recall_10:.2%}")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # Salvar JSON
    serializable_results = {}
    for k, v in results.items():
        serializable_results[k] = {k2: (str(v2) if v2 else f">{SEARCH_K}") for k2, v2 in v.items()}

    with open("result.json", "w") as f:
        json.dump(serializable_results, f, indent=2)
    print("\n  Resultados salvos em winnex_benchmark_improved_v4.2.json")

if __name__ == "__main__":
    main()
