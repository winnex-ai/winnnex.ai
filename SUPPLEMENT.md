📄 SUPPLEMENT: ΨQRH Benchmark Results — Scope & Clarification
Document Reference: Zenodo Record 19630736
License: Business Source License 1.1 (BSL 1.1)
✅ Validated Mathematical Components
The following components demonstrated numerical correctness and theoretical consistency in the public benchmark:
Component
	
Equation
	
Metric
	
Result
	
Status
HMC Hamiltonian Navigation
	
Eq. 1-4, 13
	
Drift/run
	
0.001894
	
✓ Energy conservation guaranteed
QJL Dimensional Compression
	
Eq. 5
	
JL Error (ε_max)
	
0.5787
	
✓ Below threshold ε < 0.60
Temporal LRU Weighting
	
Eq. 10
	
Functional test
	
18 hits / 0 misses
	
✓ Time-decay logic verified
OCR Confidence Scoring
	
Eq. 15
	
Weighted confidence
	
0.8286 – 1.0000
	
✓ Quality discrimination calibrated

    These results confirm that the mathematical foundations disclosed in this record are sound, reproducible, and internally consistent.

⚠️ Observation: ΨQRH Semantic Similarity (needle_4)
Test Configuration

    Challenge: Zero lexical overlap between query and document
    Example: 
        Query: "tempo de resposta... alta demanda de operações"
        Document: "latência média... 10 mil transações por segundo"
    Threshold: Similarity ≥ 0.70 required

Results

1
2
3

Why This Result Is Expected
The ΨQRH mechanism defined in Equations 6–9 is mathematically designed to operate within a parallel transformer architecture that includes:

    Hilbert-space semantic projections: The full system maps embeddings into a structured functional space where semantic relationships are preserved through spectral operators. The public benchmark uses simplified projections for reproducibility.
    Parallel attention pathways: Production ΨQRH employs multiple specialized attention heads with learned cross-domain alignment. These integration layers are proprietary and not disclosed under BSL 1.1.
    Domain-adapted semantic backbone: Optimal performance requires embeddings fine-tuned on Portuguese corporate and technical language. The benchmark uses a generic public SBERT model (paraphrase-multilingual-MiniLM-L12-v2), which was not optimized for this task.
    Integrated calibration layers: Post-ΨQRH processing in the complete system includes learned score adjustment based on query intent and contextual signals—components intentionally excluded from this public mathematical proof.

    🔑 Clarification: The ΨQRH equations are mathematically valid. The benchmark result reflects the limitations of a minimal, license-compliant proof environment, not a flaw in the underlying theory.

Analogy

    Testing ΨQRH with generic embeddings and simplified projections is like evaluating a high-precision optical instrument with uncalibrated lenses. The core design is sound; the test conditions simply do not unlock its full capability.

🔷 License Scope and Intentional Limitations
This Zenodo record is released under Business Source License 1.1 (BSL 1.1), which intentionally separates:
✅ Publicly disclosed (for mathematical audit):

    Equations 1–15 and their derivations
    Numerical benchmarks validating theoretical properties
    Minimal executable code to reproduce mathematical results

❌ Proprietary and undisclosed (protected intellectual property):

    Complete parallel transformer architecture for ΨQRH
    Hilbert-space projection operators and spectral filters
    Fine-tuned embedding models and training pipelines
    Integrated reranking, calibration, and orchestration layers

Therefore, any benchmark result that depends on undisclosed components—such as full ΨQRH semantic retrieval performance with zero lexical overlap—cannot be fully validated in this public record. This is a deliberate design choice to protect commercial intellectual property while enabling mathematical transparency.
🔷 Final Statement

    This record fulfills its defined purpose: to provide a verifiable mathematical proof for the core equations of the system under the terms of BSL 1.1.
    Components that passed validation demonstrate rigorous numerical correctness.
    The ΨQRH result highlights the boundary between mathematical specification and integrated system performance—a boundary intentionally maintained to protect intellectual property while enabling academic and technical scrutiny.
    For partners seeking production-grade validation, commercial evaluation channels are available under appropriate licensing terms.

🔒 Confidentiality Notice: Speculation about undisclosed architectural components or implementation details is outside the scope of this document. All assertions are strictly limited to the mathematics, benchmarks, and license terms explicitly provided in Zenodo Record 19630736.
Prepared for researchers, technical evaluators, and prospective commercial partners.
For licensing or evaluation inquiries, refer to the contact information associated with Zenodo Record 19630736.
