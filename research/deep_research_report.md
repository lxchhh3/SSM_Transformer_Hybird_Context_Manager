# Deep-research sweep — SSM / spike advantage surface (cited)

Online literature sweep, 2026-07-01. 3 angles → 16 sources fetched → 78 claims → top
25 adversarially verified (need 2/3 refutes to kill) → **21 confirmed, 4 refuted**.
This file is the raw cited record (it superseded the earlier curated `advantage_surface.md` menu).

## Bottom line

The **SSM side is well-evidenced** and transfers to a small (2.8–7B), local, streaming
target. The proven native strength is a **constant-size recurrent state → O(1)/token
inference, no growing KV-cache** — exactly the fixed-size streaming-cache primitive.
The equally-proven **wall is a memory-vs-recall tradeoff**: a finite state caps how much
context it compresses, so pure SSMs suffer exponential recency bias, fail associative
recall (MQAR), degrade on early-context needle-in-haystack, and are bounded to TC0
(can't do permutation/state-tracking). The field's validated fix is a **sparse hybrid:
~1 attention layer per 7–8 SSM layers (~8%)** recovers ICL / format-adherence / retrieval
while keeping up to ~3× throughput at 7–8B. The **spiking branch is thin** — no SNN
energy or long-range claim survived verification.

## Confirmed findings (all 3-0 unless noted)

1. **Constant-size state → O(1)/token, no KV-cache.** Mamba unrolls autoregressively in
   constant time per step, "does not require a cache of previous elements." Holds for
   selective S6. Per-step O(d_model·d_state), constant in sequence length. — Mamba (Gu &
   Dao 2023), arXiv:2312.00752.

2. **Selectivity (S6) is what recovers content-based recall** vs time-invariant S4/Hyena
   — controlled ablation isolates it as causal. Selective Copying: S4 18.3, Hyena 30.1 →
   S6 97–99.8. Induction heads solved by S6. (Recovery vs LTI SSMs, not parity with
   transformers.) — arXiv:2312.00752.

3. **Small-scale competitive — but NOT on recall.** Mamba-3B matches transformers ~2×
   its size on LM/common-sense; scaling laws validated 125M–1.3B vs Transformer++. BUT
   follow-ups show Mamba **weaker than same-size transformers on retrieval/ICL** — the
   exact regime a compressed-cache-over-event-stream lives in. — arXiv:2312.00752,
   2402.04248.

4. **Unifying theory: SSMs as amortized online learners.** The recurrence emerges as the
   closed-form solution to an online associative-recall objective, unifying SSMs, linear
   attention, RetNet, GLA, Griffin. — Longhorn, arXiv:2407.14207 (NeurIPS 2024).

5. **The honest wall — memory-vs-recall Pareto.** "Recurrent models are efficient because
   they have a finite state… their effectiveness is limited by how well this state has
   compressed the context." BASED: "a key tradeoff between state size and recall ability";
   you can dial state size (window / feature dim) to traverse the Pareto frontier. MQAR
   theory: gated-conv models need width scaling ~linearly with sequence length for
   associative recall (information-theoretic lower bound). Derived at 355M–1.3B → **transfers
   to small scale.** — arXiv:2312.00752, 2402.18668 (BASED), 2312.04927 (Zoology/MQAR).

6. **Empirical manifestation: retrieval failure.** SSMs match transformers on standard
   regression ICL, beat them on sparse parity, but "fall short in tasks involving
   non-standard retrieval functionality." — Park et al., arXiv:2402.04248 (ICML 2024).

7. **Hard expressivity ceiling: TC0.** Standard S4/Mamba/S6 lie in TC0 → provably cannot
   do state-tracking / permutation composition (S5 word problem), regardless of state
   size. Corroborated for LRNNs with positive/[0,1] eigenvalues. **Scope:** assumes
   TC0≠NC1 (believed, unproven) + log-precision; *modified* SSMs (negative eigenvalues,
   input-dependent/non-triangular transitions) provably escape it. — "The Illusion of
   State in SSMs," arXiv:2404.08819; Grazzi et al., arXiv:2411.12537 (ICLR 2025).

8. **Provable exponential recency bias.** Token-pair influence decays exponentially with
   distance (Theorem 3.1, only assumes diagonal A_t∈(0,1) — holds for S6/Mamba).
   Empirically, Mamba-Codestral-7B (pure Mamba-2, target scale) needle-in-haystack is
   accurate near the END of context, drops near the BEGINNING. **Mitigable** via
   near-1-eigenvalue channels / polarization, so it's a tendency, not an unfixable wall.
   — arXiv:2501.00658 (ICLR 2025).

9. **Sparse hybrid recovers recall at small scale.** BASED (Taylor-feature linear attn +
   ~64-tok sliding window) matches Mamba perplexity, beats it on recall by **6.22 pts**;
   a 64-tok window alone recovers 90.8% of full-attention recall; ~24× throughput vs
   FlashAttn-2 at 1.3B. MambaFormer (interleave attention + Mamba) "surpasses individual
   models in tasks where they struggle independently," nearly closes MQAR gap. Trained
   ≤1.3B → holds at small scale. — arXiv:2402.18668, 2402.04248.

10. **Working ratio: ~1 attention : 7–8 SSM (~8%).** Jamba released 1:7; ablation at 1.3B
    found 1:3 vs 1:7 "virtually no performance difference." Nemotron-H ~8% attention
    (4/52 in 8B, 10/118 in 56B), evenly dispersed. — arXiv:2403.19887 (Jamba, ICLR 2025),
    2504.03624 (Nemotron-H).

11. **What the few attention layers recover: ICL, output-format adherence, induction
    heads, retrieval.** Pure Mamba fails format adherence (IMDB 48.8 vs 84.1; outputs
    "Very Good" instead of "Positive"); **1 attention layer per 8 restores successful
    ICL**; 12 induction heads found in the hybrid's attention layers. — arXiv:2403.19887.

12. **7–8B hybrid: no quality loss + big throughput.** Nemotron-H-8B matches/beats
    Qwen-2.5-7B / Llama-3.1-8B while running up to **~3× faster**; RULER 91.5%@16K →
    81.7%@128K. (Vendor self-report; 8B slightly above target; margins small in the Jamba
    ablation.) — arXiv:2504.03624, 2403.19887.

13. **Spiking is THIN (low confidence, 1-2).** All surveyed SNN energy / long-range
    claims were **refuted at the verification bar** — treat as unsubstantiated, not
    established. — OpenReview 35gF1nqsgw (single insufficient source).

## Refuted — do NOT claim these

- **Train-short → infer-long 16× extrapolation.** (1-2) The "unbounded/streaming context"
  strength is substantiated only for the **O(1) mechanism**, NOT for reliable length
  extrapolation. — arXiv:2407.14207.
- **"SSM recurrent state grants no extra power over transformers."** (1-2) Over-stated
  phrasing of the TC0 result; see finding 7 for the correct, scoped version. — arXiv:2404.08819.
- **P-SpikeSSM LRA avg 79.03% preserves long-range memory.** (1-2) — OpenReview 35gF1nqsgw.
- **Spiking ~36× energy efficiency (0.9pJ acc vs 4.6pJ MAC).** (1-2) — OpenReview 35gF1nqsgw.

## Open questions (from the sweep)

1. Do hybrid quality/throughput conclusions hold at the **LOW end (2.8–3B)** on a single
   consumer GPU? All strong hybrid evidence is 7–8B/56B; the memory-bound small-batch
   regime of local inference may erode the O(1) throughput edge.
2. For an always-fresh cache over an **unbounded** stream, what's the practical usable
   memory depth before recency collapse, and can mitigations (eigenvalue polarization,
   DeltaNet-style updates, larger d_state) extend it without a KV cache — quantitatively?
3. Is **train-short/infer-long streaming extrapolation** actually reliable? (16× refuted.)
4. Does the **spiking-SSM intersection** preserve long-range memory AND real energy
   savings? None survived verification → genuinely open; needs dedicated primary sourcing.

## Caveat on sourcing

12/13 findings rest on primary peer-reviewed papers (Mamba; BASED/Zoology from Stanford
Hazy Research; Illusion-of-State from Merrill et al.; Jamba/AI21; Nemotron-H/NVIDIA; Park
et al. ICML) with unanimous 3-0. Vendor results (Jamba, Nemotron-H) are published
ablations but carry mild vested interest and cite throughput in the most SSM-favorable
long-context regime. TC0 "proofs" are conditional (TC0≠NC1 + log-precision, standard
diagonal SSMs only). Small-scale transfer is well-supported for the tradeoff/recall
findings (355M–1.3B) and hybrid ratios (1.3B), but strongest quality/throughput evidence
is 7–8B with no test at 2.8B. Field is fast-moving (2024–2026); newer update-rule work
(Gated DeltaNet, Mamba-3, StateX) refines but does not overturn the tradeoff.

## Primary sources

- arXiv:2312.00752 — Mamba (Gu & Dao)
- arXiv:2402.18668 — BASED (Arora et al., Stanford Hazy Research)
- arXiv:2312.04927 — Zoology / MQAR
- arXiv:2402.04248 — "Can Mamba Learn How to Learn?" / MambaFormer (Park et al., ICML 2024)
- arXiv:2404.08819 — "The Illusion of State in SSMs" (Merrill, Petty, Sabharwal)
- arXiv:2411.12537 — Grazzi et al. (ICLR 2025), LRNN state-tracking
- arXiv:2501.00658 — recency-bias theorem + Codestral-7B needle test (ICLR 2025)
- arXiv:2407.14207 — Longhorn (online-learner unification)
- arXiv:2403.19887 — Jamba (AI21, ICLR 2025)
- arXiv:2504.03624 — Nemotron-H (NVIDIA)
- arXiv:2406.07522 — Samba (Mamba + sliding-window attention)
- research.ibm.com/blog/bamba-ssm-transformer-model — Bamba-9B
- zyphra.com/post/zamba2-small — Zamba2-2.7B
- OpenReview 35gF1nqsgw — P-SpikeSSM (spiking, refuted at bar)
