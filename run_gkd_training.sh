#!/bin/bash

# GKD Training Script with Optimized Settings
# 用于单机8卡的 GKD 训练脚本，包含已修复的配置

set -e

# ============================================================================
# 配置部分 - 请根据实际情况修改
# ============================================================================

# 模型路径
MODEL_PATH="/home/ma-user/work/nlp/***/models/Qwen2.5-1.5B-Instruct"

# 数据文件路径
DATA_FILE="/home/ma-user/work/nlp/***/data/sftdata/raw/Countdown-Task-GOLD/verified_Qwen3-4B-Instruct-2507/train-00000-of-00001.parquet"

# Megatron-LM 路径
MEGATRON_PATH="/home/ma-user/work/nlp/***/code/megatron/Megatron-LM-0.12.1"

# 输出目录
OUTPUT_DIR="/home/ma-user/work/nlp/***/grpo/Outputs/${EXP_NAME:-gkd_test}"

# Teacher 服务配置
TEACHER_IP="127.0.0.1"
TEACHER_PORT="15555"

# ============================================================================
# 环境设置
# ============================================================================

echo "[*] Setting up environment..."

# 设置 Python 路径
export PYTHONPATH="${PYTHONPATH}:${MEGATRON_PATH}"

# 启用诊断日志（NCCL、PyTorch 分布式、vLLM）
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO
export VLLM_LOGGING_LEVEL=INFO

# 关键：设置 NCCL 超时（防止卡死）
export NCCL_TIMEOUT=1200  # 20 分钟

# 可选：其他 NCCL 优化
export NCCL_CUMEM_ENABLE=0  # 防止权重同步期间的卡死或崩溃
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 显示环境变量
echo "PYTHONPATH: $PYTHONPATH"
echo "NCCL_DEBUG: $NCCL_DEBUG"
echo "NCCL_TIMEOUT: $NCCL_TIMEOUT"
echo "TORCH_DISTRIBUTED_DEBUG: $TORCH_DISTRIBUTED_DEBUG"

# ============================================================================
# 验证配置
# ============================================================================

echo ""
echo "[*] Verifying configuration..."

if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ MODEL_PATH not found: $MODEL_PATH"
    exit 1
fi
echo "✓ Model path: $MODEL_PATH"

if [ ! -f "$DATA_FILE" ]; then
    echo "❌ DATA_FILE not found: $DATA_FILE"
    exit 1
fi
echo "✓ Data file: $DATA_FILE"

mkdir -p "$OUTPUT_DIR"
echo "✓ Output directory: $OUTPUT_DIR"

# ============================================================================
# 启动训练
# ============================================================================

echo ""
echo "=========================================="
echo "  Starting GKD Training"
echo "=========================================="
echo ""

cd /home/ma-user/work/nlp/***/verl/

# 运行训练（所有参数在同一行）
nohup python3 -m recipe.gkd.main_gkd \
  --config-path=recipe/gkd/config \
  --config-name=on_policy_distill_trainer \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  data.train_files="$DATA_FILE" \
  trainer.total_epochs=1 \
  trainer.n_gpus_per_node=1 \
  rollout.n_gpus_per_node=3 \
  actor_rollout_ref.teacher.server_ip="$TEACHER_IP" \
  actor_rollout_ref.teacher.server_port=$TEACHER_PORT \
  actor_rollout_ref.teacher.n_server_workers=4 \
  actor_rollout_ref.teacher.num_microbatches=4 \
  actor_rollout_ref.nccl_timeout=1200 \
  trainer.scheduler=one_step_off \
  > "$OUTPUT_DIR/train.log" 2>&1 &

TRAIN_PID=$!
echo "✓ Training started with PID: $TRAIN_PID"
echo "✓ Log file: $OUTPUT_DIR/train.log"

echo ""
echo "=========================================="
echo "  Tips for Monitoring"
echo "=========================================="
echo ""
echo "1. View real-time logs:"
echo "   tail -f $OUTPUT_DIR/train.log"
echo ""
echo "2. Monitor weight synchronization:"
echo "   tail -f $OUTPUT_DIR/train.log | grep 'weight sync'"
echo ""
echo "3. Check for errors:"
echo "   tail -f $OUTPUT_DIR/train.log | grep -i 'error\\|failed\\|timeout'"
echo ""
echo "4. Monitor GPU usage:"
echo "   watch -n 1 nvidia-smi"
echo ""
echo "5. Check process status:"
echo "   ps aux | grep $TRAIN_PID"
echo ""
echo "=========================================="
echo ""

# ============================================================================
# 后续操作
# ============================================================================

echo "[*] Training process started. You can now:"
echo ""
echo "1. Monitor the logs:"
echo "   tail -f $OUTPUT_DIR/train.log | grep -i 'weight sync\\|error\\|timeout'"
echo ""
echo "2. If training hangs, debug with:"
echo "   python3 diagnose_gkd_deadlock.py $OUTPUT_DIR/train.log"
echo ""
echo "3. Kill training if needed:"
echo "   kill $TRAIN_PID"
echo ""
echo "[*] Script completed. Training is running in background."
