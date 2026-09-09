# Phase 1 GO/NO-GO Experiment Plan

**Problem**: Đo tác động của KV-cache quantization lên prompt leakage trong cùng một pipeline Shadow/KIVI.

**Method thesis**: Giữ nguyên model, prompt, tokenizer và leakage evaluator; chỉ thay representation của prompt KV-cache bằng KIVI-4/KIVI-2 để phân biệt leakage thực sự bị mất với trường hợp attack chỉ gặp representation mismatch.

**Date**: 2026-09-09

**Phase boundary**: Phase 1 chỉ xác nhận hiện tượng. Không weight quantization, QAT, fine-tuning, prompt optimization, attack mới, defense mới hoặc sweep lớn.

## 1. Quyết định cần đưa ra

Phase 1 phải trả lời hai câu hỏi theo thứ tự:

1. KIVI có làm giảm leakage trên cùng prompt set mà vẫn giữ utility chấp nhận được không?
2. Nếu Collision được cho biết quantizer và dùng cùng representation ở local candidate cache, leakage có quay lại không?

Kết quả chỉ được dùng để chọn hướng Phase 2:

- **GO — quantization-as-defense**: leakage giảm cả dưới attacker-aware protocol và utility không collapse.
- **GO — attack-specific robustness**: chỉ một số attack giảm; nghiên cứu tiếp theo phải tập trung vào cơ chế khác nhau, không claim KIVI là defense tổng quát.
- **NO-GO — privacy defense claim**: naive attack fail nhưng quantization-aware attack recover, hoặc leakage giảm chỉ vì utility hỏng.
- **NO-GO — pipeline**: một gate correctness/provenance/metric thất bại; dừng mở rộng cho đến khi sửa.

## 2. Claim map

| Claim | Vì sao quan trọng | Bằng chứng tối thiểu | Block |
|---|---|---|---|
| C1. KIVI quantization làm thay đổi prompt leakage | Đây là hypothesis trung tâm của Phase 1 | FP16, KIVI4-FQ, KIVI2-FQ trên cùng ít nhất 100 prompt, raw per-sample JSONL, paired bootstrap CI, tách từng attack/layer | B1, B2 |
| C2. Mức giảm leakage là do mất thông tin hay chỉ do representation mismatch | Tránh kết luận sai từ Collision gốc | Protocol A: local FP16; Protocol B: local cùng KIVI-b; Collision và Collision+ report riêng | B3 |
| C3. Leakage–utility trade-off có ý nghĩa thực tế | Privacy không có giá trị nếu model bị hỏng | PPL/NLL, generation agreement, task score hoặc exact match, KV storage và distortion | B2, B4 |

**Anti-claim phải loại trừ**:

- “KIVI là privacy defense” chỉ vì attack gốc không tương thích với cache quantized.
- “KIVI-FQ” khi cache vẫn còn prompt token ở FP residual.
- “BERTScore” nếu implementation thực tế chỉ tính cosine của `all-mpnet-base-v2`.
- “Compression ratio” nếu chỉ đo dequantized FP attack view thay vì packed native payload.

## 3. Điều kiện bất biến

| Thành phần | Quy định Phase 1 |
|---|---|
| Model | `Llama-3.2-1B`, weights không quantize, cùng checkpoint/hash |
| Dtype | FP16 cho model/cache baseline và attack view |
| Dataset | Cùng sample IDs, tokenizer, prompt formatting và thứ tự ở mọi condition |
| Main size | 100 prompt; 20 prompt chỉ là development/screening |
| KIVI | Public pinned implementation; `group_size=32`; K/V bit-width 4/4 và 2/2 |
| Randomness | Seed 42; ghi cả Python/NumPy/PyTorch/CUDA seed |
| Attacks | Shadow implementation; không đổi candidate policy, distance, pruning hoặc instruction set |
| Conditions | FP16, KIVI4-STD, KIVI4-FQ, KIVI2-STD, KIVI2-FQ |
| KV source | Mọi condition bắt đầu từ cùng một canonical prefill cache |
| Evidence | Raw JSONL bắt buộc; aggregate CSV/Markdown/plots chỉ là derived outputs |

### Dataset decision

Dataset gated `lmsys/lmsys-chat-1m` chưa có quyền truy cập trong environment hiện tại. Nếu tiếp tục với dataset public, phải ghi rõ external-validity limitation và provenance của [natong19/lmsys-chat-1m-filtered](https://huggingface.co/datasets/natong19/lmsys-chat-1m-filtered); không gọi đây là bản gốc LMSYS-Chat-1M.

Bộ `dataset/lmsys-chat-1m_1k.jsonl` đang có đã chọn prompt ngắn: chỉ 5/1.000 prompt có token length lớn hơn 32 theo Llama tokenizer. Bộ này phù hợp smoke nhưng **không đủ tốt cho residual ablation** vì KIVI-STD phần lớn chỉ giữ nguyên residual FP. Main Phase 1 phải tạo lại sample set có length strata cố định, khuyến nghị:

- 20 dev: 10 prompt dưới 32 token và 10 prompt từ 32–128 token;
- 100 main: 25 prompt dưới 32, 50 prompt 32–63, 25 prompt 64–128 token;
- loại duplicate first-user prompt, cap 128 token, lưu `sample_id`, source row ID và token length.

FQ vẫn phải quantize cả prompt ngắn; padding metadata phải phân biệt rõ padding với prompt token.

## 4. Cache conditions và lifecycle

| ID | Representation | Prompt FP residual? | Mục đích |
|---|---|---:|---|
| `FP16` | canonical Shadow cache | N/A | baseline |
| `KIVI4-STD` | KIVI 4/4, residual mặc định 32 | Có | deployment-style behavior |
| `KIVI4-FQ` | KIVI 4/4, flush residual trước evaluation | Không | isolate 4-bit effect |
| `KIVI2-STD` | KIVI 2/2, residual mặc định 32 | Có | deployment-style aggressive quantization |
| `KIVI2-FQ` | KIVI 2/2, flush residual trước evaluation | Không | isolate 2-bit effect |

`FQ` không phải quantizer mới và không được cài bằng cách đơn giản đặt `residual_length=0` nếu điều đó làm thay đổi công thức hoặc gây chia cho zero. Lifecycle cần:

1. lấy toàn bộ K/V của prompt sau prefill;
2. chạy public KIVI quantizer trên toàn bộ vùng prompt, kể cả vùng residual;
3. nếu chiều quantization cần padding, chỉ pad bên ngoài prompt và lưu valid length;
4. lưu packed code, scale/minimum metadata, valid lengths và dequantized attack/attention view;
5. assert số prompt token còn trong FP residual bằng 0;
6. tính distortion từ FP16 source trước quantization, không tính từ một cache đã bị quantize trước đó.

Native `packed` representation là object đo memory. Dequantized view chỉ dùng cho attention/attack evaluator và không được dùng để claim storage saving.

## 5. Attack protocol

### Attacks bắt buộc

- **Inversion**: first/mid/last nếu implementation hỗ trợ; report semantic score, ROUGE-L, token accuracy và full-prompt exact recovery.
- **Collision**: first/mid/last; report semantic score, ROUGE-L, token accuracy, full sequence recovery, candidate evaluations/token và runtime.
- **Collision+**: report hoàn toàn riêng khỏi Collision, cùng ba layer; threshold/calibration policy phải cố định trước main run và không tune theo bit-width/evaluation sample.
- **Injection**: instruction set của Shadow, không search prompt mới; report semantic score và ROUGE-L.

Metric hiện tại của repo gắn nhãn `BERTScore` nhưng dùng cosine similarity của sentence embedding. Trong output mới phải ghi tên chính xác là `semantic_cosine_mpnet`; nếu muốn claim canonical BERTScore thì chạy thêm implementation BERTScore riêng và ghi rõ model/version.

### Protocol A — representation mismatch / naive attacker

- Target cache: KIVI-b, dequantized view tương ứng packed cache.
- Local candidate cache: FP16 output từ base model.
- Giữ nguyên Shadow candidate policy và pruning.

Đây là out-of-the-box robustness, không phải bằng chứng KIVI xóa thông tin.

### Protocol B — quantization-aware attacker

- Target cache: KIVI-b.
- Local candidate cache: cùng KIVI-b config và cùng FQ/STD lifecycle với target.
- Chỉ thêm cache transform quanh candidate output; không đổi candidate policy, distance hoặc threshold logic.

Protocol B ưu tiên chạy cho Collision và Collision+ sau khi Protocol A cho thấy giảm leakage. Quantizer/dequantizer phải là public KIVI functions; không dùng original FP16 candidate cache ở bước tính distance.

## 6. GO/NO-GO gates

Các ngưỡng dưới đây được pre-register cho quyết định Phase 1; chúng là engineering/research gates của project, không phải universal definition của privacy.

### Gate G0 — data, environment, provenance

**GO nếu**:

- dataset manifest có source/revision/license, local SHA-256 và sample IDs;
- model/tokenizer/checkpoint hash và package/CUDA/GPU versions được ghi;
- mọi condition dùng cùng sample IDs và `input_ids`;
- không có weight quantization hoặc fine-tuning;
- raw output có seed, config, layer, protocol và git commits.

**NO-GO nếu** thiếu bất kỳ trường nào, sample bị lệch giữa condition, hoặc không chứng minh được cache source là chung.

### Gate G1 — representation correctness, trước mọi attack

**GO nếu** 20 dev samples pass:

- native KIVI payload tồn tại cho 4/4 và 2/2;
- FQ có `fp_residual_prompt_tokens = 0` cho mọi layer/sample;
- packed code/metadata load được bằng `weights_only=True`;
- dequantized view giữ đúng layer count, shape, dtype policy và finite values;
- `MSE(K_fp,K_dequant)` hoặc `MSE(V_fp,V_dequant)` khác 0 trên vùng đã quantize;
- STD và FQ khác nhau đúng ở lifecycle residual, không khác do model/prompt/seed;
- utility prefill/generation chạy ít nhất một prompt.

**NO-GO nếu** FQ còn residual prompt, native payload không được dùng trong storage report, hoặc dequantization không tái tạo được shape/finite tensor.

### Gate G2 — Shadow baseline

**GO nếu** FP16 20 samples tạo đủ raw records cho inversion, Collision, Collision+ và injection; first/mid/last không bị mất; runtime và metric đều finite. Không yêu cầu khớp từng số trong paper, nhưng attack không được trả toàn bộ empty/zero do wiring lỗi.

**NO-GO nếu** baseline không chạy được hoặc metric implementation không xác định được.

### Gate G3 — minimum signal, 20 samples

Chạy trước `FP16`, `KIVI4-FQ`, `KIVI2-FQ` với Protocol A.

**GO để scale lên 100 nếu**:

- tất cả condition có raw records đủ attack/layer;
- ít nhất một attack/layer có paired delta khác 0 và CI không bị lỗi;
- KIVI4-FQ không làm utility smoke test collapse;
- không phát hiện prompt/seed/cache mismatch.

**NO-GO để scale** nếu FQ chưa verified, output chỉ có aggregate không có sample records, hoặc utility/metric pipeline hỏng. Nếu signal bằng 0 nhưng pipeline đúng, vẫn scale lên 100 để kết luận “no observable effect”, không được dừng chỉ vì kết quả âm.

### Gate G4 — main 100 samples

Chạy đầy đủ 5 conditions, Protocol A, bốn attack families và layer breakdown.

**Evidence threshold cho một leakage reduction**:

- paired mean/median delta `quantized - FP16` có bootstrap 95% CI không chứa 0;
- relative reduction tối thiểu 20% trên metric semantic primary ở aggregate attack đó;
- không có hơn 1/3 layer positions bị regression lớn hơn 5 percentage points;
- báo cáo riêng từng sample và length stratum, không chỉ average.

Ngưỡng 20% là ngưỡng “signal đáng đào sâu”, không phải permission để claim defense.

### Gate G5 — utility guardrail

Để gọi một condition là practical:

- KIVI4-FQ: PPL tăng không quá 5%, generation agreement không thấp hơn 95%, task score giảm không quá 2 điểm phần trăm;
- KIVI2-FQ: PPL tăng không quá 10%, generation agreement không thấp hơn 90%, task score giảm không quá 5 điểm phần trăm;
- nếu metric nào không có ground truth, phải ghi `not_available`, không thay bằng số 0.

Nếu condition vượt guardrail, vẫn report numerical result nhưng đánh dấu **utility-fail**, không dùng cho claim privacy–utility.

### Gate G6 — attacker knowledge

Nếu Protocol A cho thấy reduction ≥20% ở Collision hoặc Collision+, bắt buộc chạy Protocol B.

- **B phục hồi về ≥90% FP16 leakage**: NO-GO cho claim “quantization removes leakage”; kết luận đúng là representation mismatch.
- **B vẫn thấp hơn FP16 ít nhất 20% với CI không chứa 0** và utility pass: GO cho đào sâu quantization-aware privacy.
- **B không ổn định/không chạy**: NO-GO claim; giữ kết quả A là exploratory בלבד.

## 7. Scientific decision tree

| Quan sát sau G4–G6 | Quyết định |
|---|---|
| KIVI4-FQ và KIVI2-FQ giảm ít nhất 3/4 attack families, Protocol B vẫn giảm, utility pass | **GO mạnh**: nghiên cứu quantization-aware privacy; thêm model/bit allocation ở Phase 2 |
| Naive A giảm nhưng B phục hồi gần FP16 | **NO-GO defense claim**; **GO attack-robustness**: phân tích representation mismatch |
| Collision robust nhưng Injection giảm | **GO attack-specific**: token matching còn, semantic usability giảm |
| Collision giảm nhưng Injection robust | **GO attack-specific**: lossy cache ảnh hưởng token-level matching nhiều hơn semantic leakage |
| Chỉ KIVI2 giảm và utility KIVI2 fail | **NO-GO practical privacy claim**; cân nhắc mixed precision/selective layers |
| Không attack nào thay đổi đáng kể, utility pass | **GO robustness conclusion**: leakage robust với practical KIVI quantization |
| Leakage giảm cùng utility collapse | **NO-GO privacy benefit** |

“GO” ở bảng trên là GO cho hướng nghiên cứu tương ứng; không được biến mixed/robustness result thành claim defense tổng quát.

## 8. Experiment blocks

### Block B0 — Integrity and representation sanity

- **Claim tested**: pipeline thật sự quantize đúng prompt KV.
- **Dataset**: 20 dev, length-stratified.
- **Systems**: FP16, KIVI4-STD/FQ, KIVI2-STD/FQ.
- **Metrics**: residual prompt token count, packed payload, shape/finite, K/V MSE, cosine, relative L2.
- **Success**: G0 và G1 pass.
- **Failure interpretation**: dừng; chưa được chạy attack để suy luận privacy.
- **Target**: appendix correctness table.
- **Priority**: MUST-RUN.

### Block B1 — Minimum leakage screen

- **Claim tested**: FQ có signal leakage trên 20 prompt.
- **Systems**: FP16, KIVI4-FQ, KIVI2-FQ.
- **Attacks**: inversion, Collision, Collision+, injection; Collision first/mid/last.
- **Protocol**: A trước; B chỉ khi A giảm.
- **Metrics**: semantic cosine/ROUGE-L/token accuracy/exact match/runtime; candidate evals/token cho Collision.
- **Success**: G2 và G3 pass.
- **Failure interpretation**: pipeline NO-GO nếu correctness fail; otherwise scale để xác nhận null effect.
- **Target**: screen table.
- **Priority**: MUST-RUN.

### Block B2 — Main leakage × utility result

- **Claim tested**: quantization effect có lặp lại trên 100 prompt và có utility guardrail.
- **Systems**: đủ 5 conditions.
- **Utility**: fixed validation subset cho NLL/PPL; same privacy prompt set cho generation agreement; GSM8K small subset nếu đã chuẩn bị.
- **Metrics**: G4/G5, per-length stratum, storage native.
- **Success**: leakage CI + utility thresholds.
- **Target**: Main Table A và Table D; Plot 1, Plot 4.
- **Priority**: MUST-RUN.

### Block B3 — Residual và attacker knowledge ablation

- **Claim tested**: residual FP và local candidate representation giải thích leakage reduction.
- **Systems**: STD vs FQ ở 4/4 và 2/2; target KIVI vs local FP/same-KIVI.
- **Attacks**: Collision và Collision+ first/mid/last; injection để kiểm tra semantic effect.
- **Success**: G6 quyết định claim có được phép hay không.
- **Target**: Table B, Table C, Plot 3.
- **Priority**: MUST-RUN sau khi A cho thấy signal; nếu A không giảm thì vẫn chạy STD/FQ trên 20 để xác nhận residual mechanism.

### Block B4 — Mechanism analysis and reporting

- **Claim tested**: numerical distortion có liên hệ với leakage change.
- **Metrics**: K/V MSE, cosine, relative L2 theo layer/token position; paired delta và 95% bootstrap CI.
- **Outputs**: CSV, Markdown tables, raw JSONL, plots 1–5.
- **Success**: mọi con số truy ngược được về sample raw.
- **Target**: appendix + Plot 5.
- **Priority**: MUST-RUN for final Phase 1 decision; nice-to-have nếu chỉ mới ở 20-sample screen.

## 9. Run order and milestones

| Milestone | Runs | Decision gate | Cost/risk | Output |
|---|---|---|---|---|
| M0. Freeze | dataset stratification, manifest, environment snapshot, FQ design | G0 | thấp; risk dataset provenance | manifest + config |
| M1. Correctness | 20 × 5 cache conditions, roundtrip/distortion | G1 | thấp–trung bình | native/view + sanity JSONL |
| M2. Baseline | FP16 20, all four attacks | G2 | trung bình; Collision slow | baseline raw |
| M3. Screen | FQ 4/4 and 2/2, 20, Protocol A | G3 | trung bình | screen table |
| M4. Main | 100, 5 conditions, Protocol A | G4/G5 | trung bình–cao | main raw + utility |
| M5. Knowledge | Protocol B Collision/Collision+; residual STD/FQ | G6 | cao; quantizing each candidate expensive | attacker table |
| M6. Polish | bootstrap, plots, distortion, audit | final decision | thấp–trung bình | release bundle |

**Must-run**: M0–M4, M6 tối thiểu. M5 bắt buộc nếu G3/G4 cho thấy reduction ở Collision/Collision+.

**Không chạy 1.000 mẫu trong Phase 1**: 100 là main criterion trong file nguồn; 200–500 chỉ là robustness follow-up sau khi M5 pass. Job 1.000 mẫu đang chạy từ protocol cũ là exploratory và không được trộn vào bảng Phase 1.

## 10. Output contract

Mỗi sample/attack/layer/protocol phải có một JSONL record tối thiểu:

```json
{
  "sample_id": 17,
  "source_row_id": "...",
  "model": "Llama-3.2-1B",
  "dataset": "natong19/lmsys-chat-1m-filtered",
  "dataset_revision": "...",
  "kv_method": "kivi",
  "kv_bits": 4,
  "mode": "full_prompt_quantized",
  "residual_length": 32,
  "fp_residual_prompt_tokens": 0,
  "attack": "collision",
  "attacker_protocol": "quantization_mismatch",
  "layer_position": "first",
  "layer_index": 0,
  "input_token_length": 64,
  "semantic_cosine_mpnet": 0.0,
  "rouge_l": 0.0,
  "token_accuracy": 0.0,
  "exact_match": false,
  "full_prompt_exact_match": false,
  "attack_time_sec": 0.0,
  "candidate_evals": 0,
  "candidate_evals_per_token": 0.0,
  "k_mse": 0.0,
  "v_mse": 0.0,
  "k_relative_l2": 0.0,
  "v_relative_l2": 0.0,
  "seed": 42,
  "torch_version": "...",
  "kivi_commit": "...",
  "kvcloak_commit": "..."
}
```

Tên file có thể chứa condition để tiện vận hành, nhưng schema không được hard-code theo attack. Aggregate phải có mean, median, standard deviation và 95% paired bootstrap CI. Bắt buộc plot `leakage_quantized - leakage_FP16` theo từng sample và length stratum.

## 11. Implementation work required before M1

Repo hiện tại đã có adapter KIVI-STD và native/dequantized cache view, nhưng còn các blocker sau:

1. `KIVIConfig` hiện reject `residual_length <= 0`; cần thêm lifecycle FQ mà không thay KIVI math.
2. `defense/baseline/kivi_kvcache.py` chưa có STD/FQ mode và chưa log residual prompt token count.
3. Collision chưa có hook để transform local candidate cache bằng cùng KIVI config cho Protocol B.
4. Collision+ threshold/calibration policy cần tách khỏi Collision và cố định trước main evaluation.
5. Attack output hiện thiếu token accuracy, exact-match, candidate evaluations/token, distortion và protocol metadata.
6. Utility evaluator chưa có PPL/NLL + generation agreement report theo cùng condition.
7. Dataset sampler cần length strata và mapping `sample_id -> cache hash`.
8. Metric field hiện gọi “BERTScore” phải đổi thành tên chính xác hoặc bổ sung canonical BERTScore.

Không triển khai M1 bằng cache hiện tại cho đến khi blocker 1, 2 và 7 được giải quyết; nếu không, kết quả chỉ là exploratory smoke.

## 12. Final checklist

- [ ] G0 provenance/data/environment pass
- [ ] FQ có zero prompt FP residual và test ngắn hơn residual length
- [ ] KIVI-4 và KIVI-2 dùng public quantizer/dequantizer
- [ ] FP16 baseline reproduced trên 20 dev
- [ ] 100 identical prompts cho main
- [ ] Inversion first/mid/last (nếu supported)
- [ ] Collision first/mid/last
- [ ] Collision+ first/mid/last, report riêng
- [ ] Injection result
- [ ] Protocol A và Protocol B theo trigger G6
- [ ] STD-vs-FQ residual ablation
- [ ] PPL/NLL và generation agreement
- [ ] Native memory, K/V distortion
- [ ] Mean/median/std/95% paired bootstrap CI
- [ ] Raw JSONL + aggregate CSV + Markdown table + plots
- [ ] Quyết định cuối ghi rõ GO hướng nào hoặc NO-GO claim nào
