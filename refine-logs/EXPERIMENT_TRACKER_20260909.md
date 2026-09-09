# Phase 1 Experiment Tracker

Status values: `TODO`, `RUNNING`, `PASS`, `FAIL`, `BLOCKED`, `EXCLUDED`.

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R000 | legacy exploratory | KV-Cloak × KIVI exploratory matrix already launched | Origin/KIVI/KV-Cloak/KV-Cloak+KIVI, old short dataset | 20/1000 | existing repo “BERTScore”/ROUGE | EXCLUDED | RUNNING | Không dùng làm Phase 1 evidence: thiếu FQ, ngoài scope, dataset length-biased |
| R001 | M0 | Freeze public dataset provenance and length-stratified IDs | filtered-LMSYS, 20 dev + 100 main | dev/main | source revision, SHA-256, token lengths | MUST | TODO | Dataset public thay cho gated LMSYS; ghi external-validity limitation |
| R002 | M0 | Freeze environment and protocol | Llama-3.2-1B, FP16, seed 42 | all | model/tokenizer/package/GPU/CUDA/git hashes | MUST | TODO | Không quantize weights |
| R003 | M1 | FQ lifecycle unit tests | KIVI4-FQ/KIVI2-FQ | synthetic + short prompt | zero FP residual prompt tokens, shape, finite, packed metadata | MUST | TODO | Phải xử lý prompt ngắn hơn residual length |
| R004 | M1 | Native/dequantized cache correctness | KIVI4/2 STD and FQ | 20 dev | K/V MSE, cosine, relative L2, storage bytes | MUST | TODO | Không overwrite canonical Origin |
| R005 | M2 | Reproduce Shadow baseline | FP16 | 20 dev | inversion/collision/collision+/injection, layer breakdown | MUST | TODO | Raw per-sample JSONL |
| R006 | M3 | Minimum FQ leakage screen, Protocol A | FP16 vs KIVI4-FQ/KIVI2-FQ | 20 dev | semantic, ROUGE-L, token acc, exact match, runtime | MUST | TODO | Gate G3 |
| R007 | M4 | Main leakage matrix, Protocol A | FP16/KIVI4-STD/FQ/KIVI2-STD/FQ | 100 main | all attack families, layers, paired CI | MUST | TODO | Gate G4 |
| R008 | M4 | Utility guardrail | same five conditions | fixed validation + same prompt set | PPL/NLL, generation agreement, task score | MUST | TODO | Gate G5 |
| R009 | M5 | Residual ablation | KIVI4 STD/FQ, KIVI2 STD/FQ | 100 main | collision/injection + residual token count | MUST | TODO | Table B / Plot 3 |
| R010 | M5 | Attacker knowledge Protocol B | target KIVI, local FP vs same KIVI | 20 then 100 if needed | Collision/Collision+ first/mid/last | CONDITIONAL | TODO | Bắt buộc nếu Protocol A giảm rõ |
| R011 | M6 | Distortion/leakage mechanism | FP16 vs FQ | 100 main | K/V MSE, cosine, relative L2, leakage delta | MUST | TODO | Plot 5 |
| R012 | M6 | Aggregate and plots | all completed main runs | main | bootstrap CI, mean/median/std, per-sample delta | MUST | TODO | Release bundle |
| R013 | follow-up | Larger robustness set | selected winning conditions | 200–500 | same core metrics | NICE | TODO | Chỉ sau Phase 1 decision |
| R014 | follow-up | Secondary sanity datasets | Alpaca/GSM8K | 20 then 100 | selected attacks + utility | NICE | TODO | Không block initial Phase 1 |

## Decision log

| Date | Gate | Decision | Evidence | Next action |
|---|---|---|---|---|
| 2026-09-09 | pre-run audit | `NO-GO` for current cache as Phase 1 evidence | Current adapter has STD but no FQ; current 1k set has only 5/1000 prompts >32 tokens | Implement FQ + stratified sampler before M1 |

## Required raw artifacts

- `dataset/*.provenance.json`
- environment snapshot (`torch`, `transformers`, CUDA, GPU, model/tokenizer hashes)
- per-condition native KIVI payload metadata
- per-sample attack JSONL
- utility JSONL/CSV
- storage and distortion reports
- aggregate CSV, Markdown tables, bootstrap seed and plots
