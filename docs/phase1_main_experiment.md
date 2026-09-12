# Phase 1 main experiment — one-command run

Tài liệu này dành cho người chạy thực nghiệm chính thức. Pipeline tạo đúng
`100` prompt mới từ revision đã pin của public mirror, chia `25 short / 50
medium / 25 long`, rồi dùng cùng prompt/cache nguồn cho năm điều kiện:

- `FP16`
- `KIVI4-STD` — K/V 4-bit, residual 32
- `KIVI4-FQ` — K/V 4-bit, quantize toàn bộ prompt
- `KIVI2-STD` — K/V 2-bit, residual 32
- `KIVI2-FQ` — K/V 2-bit, quantize toàn bộ prompt

Mỗi condition chạy inversion, Collision và injection trên toàn bộ 100 mẫu;
launcher cũng đo packed-native storage, K/V distortion, prompt PPL
teacher-forced và generation agreement. Collision+ có calibration riêng và
có thể bật bằng cùng một lệnh.

## Lệnh chính thức

Sau khi checkout repo, trên máy Linux có GPU CUDA chạy:

```bash
cd /workspace/newidea
export HF_TOKEN=hf_...  # Llama-3.2-1B yêu cầu quyền truy cập Hugging Face
PYTORCH_CUDA_VARIANT=cu128 RUN_COLLISION_PLUS=1 bash scripts/run_phase_1
```

`cu128` là mặc định. Nếu node chạy CUDA 13.0, dùng:

```bash
PYTORCH_CUDA_VARIANT=cu130 RUN_COLLISION_PLUS=1 bash scripts/run_phase_1
```

Launcher chỉ chấp nhận `cu128` hoặc `cu130`. Nó cài `torch==2.9.1` từ
đúng PyTorch index trước `requirements.txt`, rồi chạy một roundtrip KIVI thật
trên GPU trước khi tạo dataset hoặc cache.

`scripts/run_phase_1` tự tạo `.venv`, cài PyTorch CUDA đã chọn rồi cài
`requirements.txt`, init
`third_party/KIVI`, tải `meta-llama/Llama-3.2-1B` và
`sentence-transformers/all-mpnet-base-v2` vào `.models/`, rồi chạy toàn bộ
pipeline. Chạy lại cùng lệnh sẽ dùng lại các dependency/checkpoint đã có.

Nếu đã có checkpoint local, truyền path để bỏ qua download tương ứng:

```bash
MODEL_PATH=/abs/path/to/Llama-3.2-1B \
EMBEDDING_PATH=/abs/path/to/all-mpnet-base-v2 \
RUN_COLLISION_PLUS=1 bash scripts/run_phase_1
```

Dataset revision public mặc định không cần token. Nếu dùng mirror/dataset
khác, phải thay cả `SOURCE_DATASET` và `SOURCE_REVISION`,
đồng thời ghi nhận license/provenance trước khi diễn giải kết quả.

`RUN_COLLISION_PLUS=1` làm thêm calibration CPA cho từng condition; đây là
bước tốn thời gian nhất. Nếu chỉ cần main Protocol-A screening, bỏ biến này:

```bash
cd /workspace/newidea && bash scripts/run_phase_1
```

## Bootstrap riêng

Có thể chỉ chuẩn bị environment và model trước bằng:

```bash
bash scripts/run_phase_1 --bootstrap-only
```

Muốn dùng một model Hugging Face public khác, đổi `MODEL_ID` và giữ
`MODEL_NAME` là một cấu hình đã được hỗ trợ trong `src/config.py`, hoặc bổ sung
cấu hình batch tương ứng:

```bash
MODEL_ID=... MODEL_NAME=Llama-3.2-1B bash scripts/run_phase_1
```

Model và embedding phải là checkpoint đầy đủ, không phải model-weight
quantized:

```text
MODEL_PATH     = checkpoint Llama-3.2-1B FP16/BF16/FP32 nguyên bản
EMBEDDING_PATH = all-mpnet-base-v2 local checkpoint
DEVICE         = cuda:0 (khuyến nghị cho Collision)
DTYPE          = float16
```

`HF_TOKEN` chỉ được dùng bởi Hugging Face Hub trong lúc tải và không được ghi
vào `environment.txt`; launcher cũng không đưa secret vào log.

## Những gì lệnh thực hiện

1. Freeze environment, commit upstream, GPU và file manifest vào `run_dir`.
2. Stream public dataset ở revision cố định, deduplicate prompt và tạo manifest
   100 mẫu.
3. Prefill một canonical FP16 cache; chỉ lưu KV + `decode.json` để tránh lưu
   attention/hidden-state không cần thiết.
4. Materialize bốn KIVI condition từ đúng canonical cache.
5. Validate layer/shape/finite/FQ residual, K/V MSE, relative L2 và native
   compression.
6. Chạy toàn bộ attack Protocol A với cờ `--i-understand-risks` đã được
   launcher truyền tường minh.
7. Chạy generation agreement và prompt PPL trên cùng cache views.
8. Nếu bật `RUN_COLLISION_PLUS=1`, tạo calibration cache Bitter Lesson, freeze
   threshold cho từng condition, rồi chạy Collision+ riêng.
9. Tạo normalized JSONL, CSV/JSON/Markdown summary và PNG plots.

Protocol A giữ local candidate cache FP16. Vì vậy kết quả này đo
`quantization-mismatch / naive attacker` và không tự nó chứng minh KIVI là
privacy defense. Nếu G4 cho thấy reduction đáng kể, cần chạy Protocol B với
local candidate cache cùng KIVI lifecycle trước khi claim
quantization-as-defense.

## Output

Mỗi run được đặt ở `logs/phase1_main_<timestamp>/`:

```text
run.log
environment.txt
model.files
embedding.files
phase1_main100_<timestamp>.jsonl
dataset.provenance.json
cache_validation.json
raw_attacks/{FP16,KIVI4_STD,KIVI4_FQ,KIVI2_STD,KIVI2_FQ}.jsonl
raw_attacks/*_collision_plus.jsonl       # chỉ khi bật CPA
generation_agreement.jsonl
prompt_ppl.jsonl
aggregate/normalized_records.jsonl
aggregate/summary.{json,csv,md}
plots/{leakage_delta_vs_fp16,native_storage_compression,kv_distortion}.png
FINAL_STATUS.txt
```

Cache tensor nặng nằm ở `cache/<dtype>/<dataset-stem>/<model-name>/`. Native
KIVI payload là nguồn duy nhất để đọc compression ratio; file
`past_key_values.pt` trong condition chỉ là attack/utility view đã
reconstruct.

## Resume và kiểm tra sau khi chạy

Nếu job bị ngắt, chạy lại cùng `RUN_DIR`; các stage đã có marker sẽ được bỏ
qua:

```bash
RESUME=1 RUN_DIR=/workspace/newidea/logs/phase1_main_<timestamp> bash scripts/run_phase1_main.sh
```

Launcher giữ lại output dở dang bằng hậu tố `.partial.<timestamp>`, không xóa
cache hay raw result cũ. Kiểm tra nhanh:

```bash
cat logs/phase1_main_<timestamp>/FINAL_STATUS.txt
cat logs/phase1_main_<timestamp>/aggregate/summary.md
```

Dry-run không cài package, download, load model hay sửa artifact:

```bash
DRY_RUN=1 bash scripts/run_phase_1
```

## Diễn giải đúng phạm vi

- `BERTScore` trong raw output của upstream là tên legacy; aggregate gọi đúng
  là `semantic_cosine_mpnet`, vì code dùng cosine của `all-mpnet-base-v2`.
- Dataset mặc định là `natong19/lmsys-chat-1m-filtered`, không phải gated
  `lmsys/lmsys-chat-1m`; phải giữ external-validity limitation này trong
  report.
- 20 mẫu dev trước đó chỉ là smoke/correctness, không được gộp vào main 100.
- Main Protocol A không thay thế Protocol B. Nếu A giảm leakage nhưng B phục
  hồi về gần FP16, kết luận đúng là representation mismatch, không phải
  privacy defense.
- Các output attack chỉ được chạy trên hệ thống và dữ liệu mà người chạy có
  quyền kiểm thử.
