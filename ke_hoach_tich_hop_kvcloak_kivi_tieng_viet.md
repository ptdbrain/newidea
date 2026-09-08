# Kế hoạch tích hợp KV-Cloak + KIVI

> Mục tiêu: sử dụng **KV-Cloak/Shadow** làm baseline và bộ khung đánh giá, sử dụng **implementation public của KIVI** cho lượng tử hóa KV-cache, và **chỉ viết glue/adapter code** để nối hai hệ thống với nhau.

---

## 1. Mục tiêu tổng thể

Kiến trúc tích hợp cần tuân thủ nguyên tắc:

- **Shadow/KV-Cloak** chịu trách nhiệm:
  - sinh/extract KV-cache;
  - chạy inversion attack;
  - chạy collision / collision+ attack;
  - chạy injection attack;
  - tính các metric đánh giá privacy;
  - chạy evaluator utility như MMLU, SQuAD;
  - cung cấp implementation gốc của KV-Cloak.

- **KIVI** chịu trách nhiệm:
  - quantize Key cache;
  - quantize Value cache;
  - pack dữ liệu lượng tử hóa;
  - dequantize dữ liệu từ representation của KIVI.

- **Code tự viết** chỉ chịu trách nhiệm:
  - chuyển đổi representation;
  - gọi đúng API/hàm public từ KIVI;
  - tổ chức pipeline;
  - lưu metadata;
  - ghép KIVI vào evaluation harness của Shadow;
  - đo storage, latency và các chỉ số tích hợp.

Không tự viết lại thuật toán tấn công của Shadow và không tự viết lại công thức quantization của KIVI.

---

# 2. Kiến trúc thí nghiệm mục tiêu

Hệ thống cuối cùng nên hỗ trợ ít nhất bốn điều kiện:

| Điều kiện | Ý nghĩa |
|---|---|
| `origin` | KV-cache gốc, chưa bảo vệ, chưa quantize |
| `kivi_k2_v2_g32_r32` | KV-cache được lượng tử hóa bằng KIVI |
| `kvcloak` | KV-cache được bảo vệ bằng KV-Cloak |
| `kvcloak_kivi_k2_v2_g32_r32` | KV-Cloak trước, sau đó lượng tử hóa bằng KIVI |

Pipeline tổng quát:

```text
                        SHADOW
                           │
                    KV-cache FP16 gốc
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
       Origin          KV-Cloak           KIVI
                           │                │
                           │          KIVI quantizer
                           │                │
                           ▼                ▼
                    KV đã bảo vệ       KV đã quantize
                           │
                           ▼
                     KIVI quantizer
                           │
                           ▼
                 KV-Cloak + KIVI
```

Khi dùng Shadow để tấn công KIVI:

```text
Representation native của KIVI
          │
          ▼
KIVI public dequantizer
          │
          ▼
Approximate FP16 K/V
          │
          ▼
Shadow inversion / collision / injection
```

### Nguyên tắc quan trọng

**Không sửa Shadow attack để attack trực tiếp packed `int32` của KIVI.**

Threat model hợp lý là attacker biết thuật toán KIVI và các metadata như scale/min/bit-width. Vì vậy attacker có thể dequantize representation rồi áp dụng attack trên approximate K/V.

---

# 3. Đóng băng phiên bản upstream

Nên pin chính xác commit của cả hai repo để đảm bảo khả năng tái lập.

Ví dụ:

```text
KV-Cloak:
6b40f36edb2f337557543e7e60b10022308883d4

KIVI:
876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6
```

Clone baseline:

```bash
git clone https://github.com/SiO-2/kvcloak.git
cd kvcloak

git checkout 6b40f36edb2f337557543e7e60b10022308883d4
git checkout -b feat/kivi-adapter
```

Thêm KIVI dưới dạng submodule:

```bash
mkdir -p third_party

git submodule add \
  https://github.com/jy-yuan/KIVI.git \
  third_party/KIVI

git -C third_party/KIVI checkout \
  876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6
```

Lợi ích:

- không copy source KIVI sang repo;
- dễ kiểm tra provenance;
- dễ tái lập thí nghiệm;
- có thể ghi rõ trong báo cáo rằng implementation KIVI không bị thay đổi.

---

# 4. Không cài chồng toàn bộ môi trường của KIVI

KV-Cloak và KIVI có dependency khác nhau.

KV-Cloak hiện dùng phiên bản mới hơn, trong khi requirements của KIVI vẫn pin một số phiên bản cũ hơn.

Do đó **không nên** chạy trực tiếp:

```bash
pip install -r third_party/KIVI/requirements.txt
```

trong cùng environment của KV-Cloak.

Thay vào đó, ưu tiên:

```bash
export PYTHONPATH=$PWD/third_party/KIVI:$PYTHONPATH
```

và chỉ import các primitive cần dùng từ KIVI, ví dụ:

```python
from quant.new_pack import (
    triton_quantize_and_pack_along_last_dim,
    unpack_and_dequant_vcache,
)
```

Mục tiêu là tránh biến integration task thành một dự án port toàn bộ KIVI sang stack Transformers mới.

---

# 5. Task 1 — Kiểm tra tương thích KIVI

## File tạo mới

```text
tests/test_kivi_import.py
```

## Mục tiêu

Xác nhận các primitive của KIVI có thể import và chạy được trong environment hiện tại.

Ví dụ tensor kiểm thử:

```text
[B, H, T, D] = [1, 4, 32, 64]
bits = 2
group_size = 32
```

Test tối thiểu:

```python
from quant.new_pack import (
    triton_quantize_and_pack_along_last_dim,
    unpack_and_dequant_vcache,
)
```

Sau đó chạy quantize → dequantize.

## Điều kiện đạt

```text
output.shape == input.shape
torch.isfinite(output).all()
```

Đây là **GO / NO-GO gate**.

Nếu KIVI Triton code không chạy với Torch/Triton hiện tại thì phải xử lý compatibility trước. Không được tự viết một quantizer khác để thay KIVI.

---

# 6. Task 2 — Xây dựng KIVI adapter

## File tạo mới

```text
src/kivi_adapter.py
tests/test_kivi_adapter.py
```

Đây là phần glue code chính.

## Cấu hình

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class KIVIConfig:
    k_bits: int
    v_bits: int
    group_size: int
    residual_length: int
```

## Interface đề xuất

```python
def quantize_cache(
    past_key_values,
    config: KIVIConfig,
):
    ...
```

```python
def dequantize_cache(
    quantized_cache,
    config: KIVIConfig,
):
    ...
```

```python
def roundtrip_cache(
    past_key_values,
    config: KIVIConfig,
):
    quantized = quantize_cache(past_key_values, config)
    return dequantize_cache(quantized, config)
```

### Ràng buộc

`src/kivi_adapter.py` không được chứa implementation mới của thuật toán lượng tử hóa.

Adapter chỉ được:

- chia tensor thành quantized region và residual region;
- transpose/reshape đúng format;
- gọi hàm public của KIVI;
- ghép kết quả lại;
- trả về representation chuẩn hóa.

---

# 7. Representation của KIVI

Không nên chỉ lưu reconstructed FP cache.

Mỗi layer nên có representation dạng:

```python
{
    "key_code": ...,
    "key_scale": ...,
    "key_min": ...,
    "key_residual": ...,

    "value_code": ...,
    "value_scale": ...,
    "value_min": ...,
    "value_residual": ...,

    "seq_len": ...,
}
```

Lý do:

- `native.pt` phản ánh memory footprint thật của KIVI;
- `past_key_values.pt` chỉ là attack view sau dequantization;
- nếu chỉ lưu reconstructed cache thì không đo được compression thật.

---

# 8. Task 3 — Partition Key/Value theo đúng KIVI

KIVI xử lý Key và Value không giống nhau.

## Key cache

Key được lượng tử hóa theo hướng per-channel.

Pipeline:

```text
K gốc
  │
  ├── phần có thể quantize
  │       │
  │       ▼
  │   transpose(2, 3)
  │       │
  │       ▼
  │   KIVI public quantizer
  │
  └── phần residual full precision
```

Representation tương ứng nên giữ:

```text
key_states_quant_trans
key_states_full
key_scale_trans
key_mn_trans
```

## Value cache

Value được lượng tử hóa theo kiểu per-token.

```text
V gốc
  │
  ├── old tokens
  │       │
  │       ▼
  │   KIVI public quantizer
  │
  └── recent residual tokens FP16
```

Representation:

```text
value_code
value_scale
value_min
value_residual
```

## Cấu hình khởi đầu đề xuất

```text
k_bits = 2
v_bits = 2
group_size = 32
residual_length = 32
```

Đây là cấu hình phù hợp cho smoke/reproduction đầu tiên.

---

# 9. Task 4 — Dequantization adapter cho Shadow attack

Shadow attack cần nhận lại dạng:

```text
tuple[(K_layer, V_layer), ...]
```

Do đó adapter phải chuyển KIVI representation về approximate FP16 K/V.

Ví dụ với Key:

```python
dequant_k_transposed = unpack_and_dequant_vcache(
    key_code,
    key_scale.unsqueeze(-1),
    key_min.unsqueeze(-1),
    group_size,
    k_bits,
)

dequant_k = dequant_k_transposed.transpose(2, 3)

K = torch.cat(
    [dequant_k, key_residual],
    dim=2,
)
```

Value:

```python
dequant_v = unpack_and_dequant_vcache(
    value_code,
    value_scale.unsqueeze(-1),
    value_min.unsqueeze(-1),
    group_size,
    v_bits,
)

V = torch.cat(
    [dequant_v, value_residual],
    dim=2,
)
```

### Nguyên tắc

Không tự viết lại dequantization formula.

Adapter phải gọi chính hàm public từ KIVI.

---

# 10. Dtype của thí nghiệm

Để so sánh công bằng, thí nghiệm chính nên dùng:

```text
FP16
```

cho tất cả condition:

```text
Origin FP16
KIVI FP16
KV-Cloak FP16
KV-Cloak + KIVI FP16
```

Không nên so:

```text
Origin FP32
vs
KIVI dequantized FP16
```

vì khi đó kết quả bị trộn giữa:

- lỗi do giảm precision FP32 → FP16;
- lỗi do KIVI quantization.

Mục tiêu là isolate tác động của KIVI.

---

# 11. Task 5 — Giữ nguyên Shadow KV extraction

Không sửa:

```text
inference/get_kvcache.py
inference/pdsplit.py
```

Shadow tiếp tục là nguồn sinh canonical KV-cache duy nhất.

Chạy ví dụ:

```bash
python inference/get_kvcache.py \
  --model-name Llama-3.2-1B \
  --dataset ./dataset/lmsys-chat-1m_1k.jsonl \
  --dtype float16 \
  --device cuda:0 \
  --max-samples 20
```

Output cơ bản:

```text
cache/float16/<dataset>/<model>/<hash>/
└── origin/
    └── past_key_values.pt
```

Tất cả condition khác phải bắt đầu từ chính file `origin/past_key_values.pt` này.

Điều này đảm bảo Origin, KIVI và KV-Cloak cùng xuất phát từ một cache gốc.

---

# 12. Task 6 — Batch processor cho KIVI

## File tạo mới

```text
defense/baseline/kivi_kvcache.py
```

## CLI đề xuất

```bash
python defense/baseline/kivi_kvcache.py \
  --model-name Llama-3.2-1B \
  --dataset-path ./dataset/lmsys-chat-1m_1k.jsonl \
  --source-protect-type origin \
  --k-bits 2 \
  --v-bits 2 \
  --group-size 32 \
  --residual-length 32 \
  --dtype float16 \
  --device cuda:0
```

## Quy trình

Với mỗi sample:

```text
origin/past_key_values.pt
      │
      ▼
KIVI quantize
      │
      ├── native.pt
      │
      ▼
KIVI dequantize
      │
      ▼
past_key_values.pt
```

## Output

```text
<hash>/
└── kivi_k2_v2_g32_r32/
    ├── native.pt
    ├── past_key_values.pt
    └── metadata.json
```

### `native.pt`

Lưu actual compressed representation.

### `past_key_values.pt`

Lưu approximate FP16 cache dùng cho Shadow attack.

### `metadata.json`

Ví dụ:

```json
{
  "method": "kivi",
  "k_bits": 2,
  "v_bits": 2,
  "group_size": 32,
  "residual_length": 32,
  "source": "origin",
  "kivi_commit": "876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6",
  "attack_view": "dequantized_from_native"
}
```

---

# 13. Task 7 — Tích hợp với Shadow attack mà gần như không sửa code

Không sửa:

```text
attack/attacks.py
attack/inversion.py
attack/collision.py
attack/injection.py
```

Shadow hiện đã hỗ trợ đọc cache theo dạng:

```text
<sample>/<protect_type>/past_key_values.pt
```

Do đó chỉ cần tạo đúng thư mục và tên `protect_type`.

Ví dụ chạy KIVI:

```bash
python attack/attacks.py \
  --target-model-name Llama-3.2-1B \
  --dataset-path ./dataset/lmsys-chat-1m_1k.jsonl \
  --protect-type kivi_k2_v2_g32_r32 \
  --dtype float16 \
  --device cuda:0 \
  --run-inversion \
  --run-collision \
  --run-injection \
  --i-understand-risks
```

Điều này giúp bảo toàn tuyên bố:

> Shadow attack implementation được giữ nguyên.

---

# 14. Task 8 — Calibration riêng cho Collision+

Nếu chạy Collision+ với:

```bash
--enhance
```

thì mỗi protection type phải có threshold/config riêng.

Không được dùng config của Origin cho KIVI.

Cần generate riêng cho:

```text
origin
kivi
kvcloak
kvcloak+kivi
```

Ví dụ:

```bash
python attack/get_collision_threshold.py \
  --model_path ~/model/Llama-3.2-1B \
  --target_data_path cache/.../<hash>/kivi_k2_v2_g32_r32/past_key_values.pt \
  --protect_type kivi_k2_v2_g32_r32 \
  --target_model_name Llama-3.2-1B \
  --dtype float16 \
  --device cuda:0
```

Sau đó mới chạy:

```bash
python attack/attacks.py ... \
  --protect-type kivi_k2_v2_g32_r32 \
  --run-collision \
  --enhance
```

---

# 15. Task 9 — Kết hợp KV-Cloak + KIVI

Thứ tự đề xuất:

```text
Original KV
    │
    ▼
KV-Cloak obfuscate
    │
    ▼
KIVI quantize
    │
    ▼
Stored cache
```

Khi model cần dùng lại cache:

```text
Stored KIVI cache
    │
    ▼
KIVI dequantize
    │
    ▼
KV-Cloak deobfuscate
    │
    ▼
Model decode
```

### Không nên dùng thứ tự

```text
KIVI
→ dequantize
→ KV-Cloak
→ lưu FP16
```

vì như vậy representation lưu cuối cùng không còn lợi ích memory compression của KIVI.

## Batch processing

Có thể reuse `kivi_kvcache.py`:

```bash
python defense/baseline/kivi_kvcache.py \
  ... \
  --source-protect-type kvcloak
```

Output:

```text
kvcloak_kivi_k2_v2_g32_r32/
├── native.pt
├── past_key_values.pt
└── metadata.json
```

Trong đó `past_key_values.pt` là:

```text
dequantized(KIVI(KV-Cloak(KV)))
```

Nó vẫn đang ở không gian đã obfuscate, nên đây là representation phù hợp để Shadow attacker quan sát.

---

# 16. Task 10 — Tích hợp utility evaluator

Các file có thể sửa rất nhẹ:

```text
defense/eval/mmlu_eval.py
defense/eval/squad_eval.py
```

Không sửa:

- prompt construction;
- dataset;
- metric;
- scoring;
- prediction logic.

Chỉ thêm một generic cache hook:

```python
cache_adapter=None
```

Sau khi evaluator lấy được `past_key_values`:

```python
if self.cache_adapter is not None:
    past_key_values = self.cache_adapter.roundtrip(
        past_key_values
    )
```

Các mode:

```text
origin:
    identity

kivi:
    quantize
    → dequantize

kvcloak:
    obfuscate
    → deobfuscate

kvcloak_kivi:
    kvcloak.obfuscate
    → kivi.quantize
    → kivi.dequantize
    → kvcloak.deobfuscate
```

Cách làm này giữ evaluator gốc gần như nguyên vẹn.

---

# 17. Task 11 — Tạo CachePipeline

## File tạo mới

```text
src/cache_pipeline.py
tests/test_cache_pipeline.py
```

Interface tối thiểu:

```python
class CachePipeline:
    def roundtrip(self, past_key_values):
        ...
```

Các implementation:

```text
IdentityPipeline
KIVIPipeline
KVCloakKIVIPipeline
```

Không cần xây một framework abstraction quá lớn.

Mục tiêu chỉ là chuẩn hóa cách evaluator gọi các phương pháp xử lý cache.

---

# 18. Task 12 — Bộ test bắt buộc

## 18.1. Test cấu trúc adapter

```text
test_kivi_quantized_cache_preserves_layer_count
test_kivi_roundtrip_preserves_shape
test_kivi_roundtrip_returns_finite_values
```

## 18.2. Test partition

```text
test_key_residual_partition
test_value_residual_partition
test_sequence_shorter_than_residual
test_exact_residual_multiple
```

Ví dụ:

```text
T = 130
residual = 32

Key:
quantized prefix = 128
FP tail = 2

Value:
quantized prefix = 98
FP tail = 32
```

Key và Value có logic partition khác nhau nên bắt buộc phải test riêng.

## 18.3. Test chất lượng quantization

Ví dụ:

```text
MSE(KIVI-4bit) <= MSE(KIVI-2bit)
```

Không cần hard-code một mức MSE tuyệt đối quá chặt.

## 18.4. Provenance test

Monkeypatch các hàm:

```python
triton_quantize_and_pack_along_last_dim
unpack_and_dequant_vcache
```

Sau đó assert adapter thực sự gọi các hàm này.

Mục tiêu:

> chứng minh code của mình đang dùng implementation public của KIVI thay vì tự viết quantizer riêng.

---

# 19. Task 13 — Smoke test trên cache thật

Sau synthetic unit test, dùng đúng một KV-cache thật do Shadow sinh.

Input:

```text
origin/past_key_values.pt
```

Pipeline:

```text
Origin
→ KIVI quantize
→ KIVI dequantize
```

Kiểm tra từng layer:

```text
shape(K_original) == shape(K_reconstructed)
shape(V_original) == shape(V_reconstructed)

không NaN
không Inf
```

Sau đó chạy Shadow attack trên đúng một sample.

## Acceptance

```text
Inversion chạy xong
Collision chạy xong
Injection chạy xong
JSONL kết quả được tạo
```

Ở bước này chưa quan trọng attack score tăng hay giảm. Mục tiêu là kiểm tra integration path hoạt động end-to-end.

---

# 20. Task 14 — Đo memory footprint đúng cách

Không dùng kích thước của reconstructed:

```text
past_key_values.pt
```

để claim KIVI compression.

Phải đo trên:

```text
native.pt
```

## File tạo mới

```text
defense/eval/kivi_storage_eval.py
```

Tính byte dựa trên actual tensor:

```python
tensor.numel() * tensor.element_size()
```

Bao gồm:

```text
packed code
scale
min
full-precision residual
```

Report:

```text
Origin KV bytes
KIVI native bytes
Compression ratio
```

Công thức:

```text
compression_ratio =
    origin_bytes / kivi_native_bytes
```

---

# 21. Task 15 — Đo latency

Nên tách riêng:

```text
T_quant
T_dequant
T_roundtrip
```

Với CUDA nên dùng:

```python
torch.cuda.Event(enable_timing=True)
torch.cuda.synchronize()
```

Có warmup trước khi đo.

### Lưu ý khi viết báo cáo

Offline adapter benchmark không đồng nghĩa với end-to-end throughput của native KIVI inference.

Trong scope glue-only hiện tại chỉ nên claim:

```text
cache size
quantization latency
dequantization latency
roundtrip utility
```

Muốn claim throughput thực của KIVI cần chạy native `LlamaForCausalLM_KIVI` hoặc pipeline inference tương đương.

---

# 22. Ma trận thí nghiệm đề xuất

## Giai đoạn A — Smoke experiment

### Model

```text
Llama-3.2-1B
```

### Dataset

```text
LMSYS — 20 samples
```

### Dtype

```text
FP16
```

### Conditions

```text
Origin
KIVI-2bit
KV-Cloak
KV-Cloak + KIVI-2bit
```

### KIVI config

```text
k_bits = 2
v_bits = 2
group_size = 32
residual_length = 32
```

### Seed

```text
42
```

---

# 23. Giai đoạn B — Ablation theo bit-width

Conditions:

```text
Origin

KIVI:
2/2 bit
4/4 bit

KV-Cloak

KV-Cloak + KIVI:
2/2 bit
4/4 bit
```

Giữ cố định trước:

```text
group_size = 32
residual_length = 32
```

Không nên sweep quá nhiều hyperparameter ngay ở vòng đầu.

Mục tiêu trước hết là hiểu trade-off giữa:

```text
privacy
utility
memory
latency
```

---

# 24. Bảng kết quả chính

Bảng tổng hợp cuối cùng nên có dạng:

| Phương pháp | KV bits | Compression ↓ | MMLU ↑ | SQuAD ↑ | Inversion ↓ | Collision ↓ | Injection ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Origin | FP16 | 1.00× | | | | | |
| KIVI | 2/2 | | | | | | |
| KIVI | 4/4 | | | | | | |
| KV-Cloak | FP16 | | | | | | |
| KV-Cloak + KIVI | 2/2 | | | | | | |
| KV-Cloak + KIVI | 4/4 | | | | | | |

Với inversion/collision nên giữ breakdown theo layer:

```text
First
Middle
Last
```

đúng theo evaluation setup của Shadow.

---

# 25. Lưu ý về metric trong Shadow

Trong implementation Shadow/KV-Cloak hiện tại có một điểm cần ghi chú khi báo cáo kết quả.

Tên biến/output có thể gọi là:

```text
BERTScore
```

nhưng implementation thực tế dùng embedding từ SentenceTransformer rồi tính cosine similarity.

Theo nguyên tắc dùng evaluator public, không cần sửa metric này trong integration.

Tuy nhiên khi viết báo cáo/paper nên mô tả chính xác, ví dụ:

> Semantic similarity được báo cáo dưới tên “BERTScore” trong implementation public của Shadow.

Không nên khẳng định đây là canonical BERTScore nếu implementation không dùng metric BERTScore chuẩn.

---

# 26. Cấu trúc repo đề xuất

```text
kvcloak/
│
├── attack/
│   ├── attacks.py                 # GIỮ NGUYÊN
│   ├── inversion.py               # GIỮ NGUYÊN
│   ├── collision.py               # GIỮ NGUYÊN
│   └── injection.py               # GIỮ NGUYÊN
│
├── inference/
│   ├── get_kvcache.py             # GIỮ NGUYÊN
│   └── pdsplit.py                 # GIỮ NGUYÊN
│
├── defense/
│   ├── core/
│   │   └── kvcloak.py             # GIỮ NGUYÊN
│   │
│   ├── baseline/
│   │   └── kivi_kvcache.py        # GLUE CODE MỚI
│   │
│   └── eval/
│       ├── mmlu_eval.py            # thêm adapter hook nhỏ
│       ├── squad_eval.py           # thêm adapter hook nhỏ
│       └── kivi_storage_eval.py    # GLUE CODE MỚI
│
├── src/
│   ├── kivi_adapter.py             # GLUE CODE MỚI
│   └── cache_pipeline.py           # GLUE CODE MỚI
│
├── tests/
│   ├── test_kivi_import.py
│   ├── test_kivi_adapter.py
│   └── test_cache_pipeline.py
│
└── third_party/
    └── KIVI/                       # git submodule, GIỮ NGUYÊN
```

Boundary này cần được giữ chặt để đảm bảo contribution của dự án là integration thay vì fork/re-implementation hai phương pháp gốc.

---

# 27. Thứ tự triển khai thực tế

```text
1. Pin commit của Shadow + KIVI
        ↓
2. Kiểm tra KIVI import/Triton compatibility
        ↓
3. Viết synthetic KIVI quant/dequant adapter
        ↓
4. Viết unit test cho adapter
        ↓
5. Shadow real cache → KIVI
        ↓
6. Lưu native.pt + attack-view past_key_values.pt
        ↓
7. Chạy Shadow attacks trên KIVI
        ↓
8. Sinh calibration riêng cho Collision+
        ↓
9. Xây KV-Cloak → KIVI composition
        ↓
10. Chạy Shadow attacks trên KV-Cloak + KIVI
        ↓
11. Chuẩn hóa CachePipeline
        ↓
12. Tích hợp MMLU/SQuAD
        ↓
13. Đo memory + latency
        ↓
14. Chạy 20-sample smoke experiment
        ↓
15. Chạy full experiment
```

---

# 28. Definition of Done

Integration được xem là hoàn thành khi đạt đủ các điều kiện sau.

## 1. Shadow extraction được giữ nguyên

Không sửa:

```text
inference/get_kvcache.py
inference/pdsplit.py
```

## 2. Shadow attacks được giữ nguyên

Không sửa:

```text
attack/inversion.py
attack/collision.py
attack/injection.py
```

## 3. Không tự implement lại quantization/dequantization của KIVI

Code tự viết chỉ làm adapter/glue.

## 4. KIVI được pin dưới `third_party/KIVI`

Adapter import trực tiếp public functions của KIVI.

## 5. Một Shadow cache gốc chạy được đầy đủ bốn condition

```text
origin
kivi
kvcloak
kvcloak+kivi
```

## 6. Có đủ ba nhóm kết quả

### Privacy

```text
Inversion
Collision / Collision+
Injection
```

### Utility

```text
MMLU
SQuAD
```

### Efficiency

```text
KV storage size
Compression ratio
Quantization latency
Dequantization latency
```

---

# 29. Nguyên tắc nghiên cứu cần giữ xuyên suốt

Toàn bộ dự án nên giữ một thông điệp kỹ thuật rất rõ:

> **Shadow/KV-Cloak được dùng làm framework baseline và threat-model/evaluation implementation. KIVI được dùng nguyên bản cho KV-cache quantization. Phần code mới chỉ cung cấp adapter giữa hai representation và orchestration cho thí nghiệm.**

Điều này giúp:

- giảm nguy cơ sai khác implementation so với paper gốc;
- tăng khả năng tái lập;
- dễ audit;
- dễ chứng minh fairness khi so sánh;
- làm rõ contribution thực sự của nghiên cứu.

---

# 30. Kết quả kỳ vọng của giai đoạn đầu

Sau giai đoạn triển khai đầu tiên, ta cần trả lời được bốn câu hỏi:

1. **KIVI riêng có làm giảm khả năng khôi phục thông tin từ KV-cache hay không?**
2. **Mức giảm privacy leakage đó có đến từ 2-bit/4-bit quantization hay chỉ từ precision loss thông thường?**
3. **KV-Cloak + KIVI có giữ được hiệu quả bảo vệ của KV-Cloak hay không?**
4. **Đổi lại privacy improvement, ta mất bao nhiêu utility, memory và latency?**

Nếu trả lời rõ được bốn câu hỏi này bằng cùng một evaluation harness của Shadow, ta sẽ có một baseline tích hợp đủ sạch để tiếp tục phát triển ý tưởng nghiên cứu mới phía trên nó.
