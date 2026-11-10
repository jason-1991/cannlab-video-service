#!/bin/bash
# One-shot setup for the Ascend video service inside the CANNLab environment.
# Run:  bash setup_video.sh [t2v|i2v]     (default t2v)
# Strategy: first run the OFFICIAL Wan2.1 generate.py CLI to prove a real clip
# end-to-end (cheapest possible proof), then install the HTTP service.
set -u
MODE="${1:-t2v}"
BASE="$HOME/canvas-service"
mkdir -p "$BASE/models" "$BASE/logs" "$BASE/outputs"
cd "$BASE" || exit 1

echo "=== env probe ==="
uname -m
npu-smi info 2>/dev/null | head -25 || echo "npu-smi NOT FOUND"
python3 -c "import torch, torch_npu; print('torch', torch.__version__, 'npu_available', torch.npu.is_available())" 2>&1 | tail -2
ls /shared_assets 2>/dev/null | head -30 || echo "no /shared_assets"
df -h "$HOME" | tail -1; free -g | head -2

echo "=== token ==="
if [ ! -f "$BASE/token.txt" ]; then
  (openssl rand -hex 24 || head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n') > "$BASE/token.txt"
  chmod 600 "$BASE/token.txt"
fi
echo "token file: $BASE/token.txt (value not printed)"

echo "=== wan code ==="
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export TOKENIZERS_PARALLELISM=false
if [ ! -d "$BASE/Wan2.1" ]; then
  git clone https://github.com/Wan-Video/Wan2.1.git "$BASE/Wan2.1" \
    || git clone https://gitcode.com/gh_mirrors/wa/Wan2.1.git "$BASE/Wan2.1"
fi
cd "$BASE/Wan2.1" && python3 -m pip install --user -q -r requirements.txt 2>&1 | tail -2

echo "=== weights ($MODE) ==="
if [ "$MODE" = "i2v" ]; then
  MS_MODEL="Wan-AI/Wan2.1-I2V-14B-480P"; DIR_NAME="wan2.1-i2v-1.3b"
  # NOTE: 1.3B I2V exists only in the original (non-diffusers) layout as
  # Wan2.1-I2V-1.3B via the both-weights release; diffusers layout handled below.
  CKPT_DIR="$BASE/models/Wan2.1-I2V-1.3B"
else
  MS_MODEL="Wan-AI/Wan2.1-T2V-1.3B"; CKPT_DIR="$BASE/models/Wan2.1-T2V-1.3B"
fi
if [ ! -d "$CKPT_DIR" ]; then
  python3 -m pip install --user -q modelscope 2>&1 | tail -1
  modelscope download --model "$MS_MODEL" --local_dir "$CKPT_DIR" 2>&1 | tail -2 \
    || huggingface-cli download "$MS_MODEL" --local-dir "$CKPT_DIR" 2>&1 | tail -2
fi
ls "$CKPT_DIR" | head

echo "=== proof clip via official CLI (t2v-1.3B, 832*480, 5s) ==="
cd "$BASE/Wan2.1"
time python3 generate.py --task t2v-1.3B --size 832*480 --ckpt_dir "$CKPT_DIR" \
  --sample_steps 30 --base_seed 42 \
  --prompt "A lighthouse on rocky cliffs at night during a storm, huge waves crashing, lightning in the distance, cinematic" \
  --save_file "$BASE/outputs/proof_t2v.mp4" 2>&1 | tail -15
ls -la "$BASE/outputs/"

echo "=== http service ==="
python3 -m pip install --user -q fastapi "uvicorn[standard]" diffusers transformers accelerate safetensors pillow imageio imageio-ffmpeg requests 2>&1 | tail -2
# diffusers-layout weights for the HTTP service (T2V 1.3B)
DIFF_DIR="$BASE/models/wan2.1-t2v-1.3b-diffusers"
if [ ! -f "$DIFF_DIR/model_index.json" ]; then
  modelscope download --model "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" --local_dir "$DIFF_DIR" 2>&1 | tail -2 \
    || huggingface-cli download "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" --local-dir "$DIFF_DIR" 2>&1 | tail -2
fi
ls "$DIFF_DIR" | head -5
for f in /usr/local/Ascend/ascend-toolkit/set_env.sh "$HOME/Ascend/ascend-toolkit/set_env.sh"; do
  [ -f "$f" ] && . "$f" && echo "sourced $f"
done
pkill -f video_service.py 2>/dev/null; sleep 1
MODE=t2v nohup python3 -m uvicorn video_service:app --host 127.0.0.1 --port 8900 \
  --app-dir "$BASE" >> "$BASE/logs/video_service.log" 2>&1 &
echo "video service pid $!"
nohup bash "$BASE/watchdog.sh" >> "$BASE/logs/watchdog.log" 2>&1 &
echo "watchdog pid $!"
sleep 5
curl -s http://127.0.0.1:8900/health || echo "health not up yet, check logs/video_service.log"
echo "=== setup_video done ==="
