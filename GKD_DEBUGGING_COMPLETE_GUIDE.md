# GKD 卡死诊断工具包 - 完整指南

## 概览

您遇到的 GKD 训练卡死问题已经有了完整的诊断方案。本文档将所有诊断工具和步骤整合在一起。

**当前问题**：训练在 "Generating train split" 阶段卡住

**根本原因**：最可能是以下之一：
1. NCCL 权重同步超时（最可能，70%）
2. Rollout Worker 初始化失败（次可能，20%）
3. 数据加载问题（可能性小，10%）

---

## 快速开始（10 分钟快速诊断）

### Step 1：运行诊断脚本
```bash
python diagnose_gpu_allocation.py --actor-gpus 2 --rollout-gpus 2 --total-gpus 8
```

**预期输出**：
```
================================================================================
  1. CUDA 基础信息
================================================================================

[*] nvidia-smi 输出:
GPU 0: ... 40GB
GPU 1: ... 40GB
...
GPU 7: ... 40GB
```

### Step 2：启用调试环境
```bash
# PowerShell
$env:NCCL_DEBUG = "TRACE"
$env:NCCL_TIMEOUT = "600"
```

### Step 3：运行训练
```bash
cd recipe/gkd
python main_gkd.py data.output_dir=./output 2>&1 | tee train_debug.log
```

### Step 4：实时监控日志
在另一个终端：
```bash
tail -f train_debug.log | grep -E "\[ACTOR\]|\[ROLLOUT\]|\[SYNC\]|ERROR|NCCL"
```

### Step 5：分析结果

**如果看到这个序列，说明正常**：
```
[ACTOR] init_model START
[ACTOR] Model loaded successfully
[ROLLOUT] init_model START
[ROLLOUT] Rollout built successfully
[SYNC] Starting weight synchronization...
[SYNC] Rollout weight sync completed
```

**如果在某个地方卡住不动 > 2 分钟，那就是问题所在**

---

## 诊断工具列表

已为您创建的工具：

### 1. `diagnose_gpu_allocation.py` - GPU 分配检查
```bash
python diagnose_gpu_allocation.py --actor-gpus 2 --rollout-gpus 2 --total-gpus 8
```
**用途**：验证 GPU 是否被正确分配到 Actor 和 Rollout

### 2. `test_gkd_components.py` - 组件测试
```bash
python test_gkd_components.py
```
**用途**：测试 CUDA、Ray、vLLM 等关键组件

### 3. `GKD_REALTIME_DEBUGGING.md` - 实时调试指南
**用途**：详细的分步调试方案，根据卡死位置诊断

### 4. `GKD_QUICK_FIX_CARD.md` - 快速参考
**用途**：快速查阅，包含常用命令和参数

### 5. `setup_gkd_debugging.py` - 自动调试设置
```bash
python setup_gkd_debugging.py --enable
```
**用途**：自动在关键位置添加调试日志

---

## 根据卡死位置的诊断方案

### 情况 A：卡在 "Generating train split"

**日志显示**：
```
Generating train split: 0 examples [00:00, ? examples/s]
[永远卡在这里]
```

**可能原因**：
- 数据文件路径错误
- 数据集库损坏
- 磁盘 IO 问题

**诊断命令**：
```bash
# 检查数据文件
ls -lah /path/to/your/data

# 测试数据加载库
python -c "from datasets import load_dataset; print('✓ datasets OK')"

# 查看磁盘使用
df -h

# 查看 I/O 等待
iostat -x 1
```

**修复方案**：
```bash
# 1. 确认数据路径配置正确
grep "train_files\|data_path" recipe/gkd/config/on_policy_distill_trainer.yaml

# 2. 重新下载数据集（如果损坏）
rm -rf ~/.cache/huggingface/datasets

# 3. 改用本地数据
python main_gkd.py data.train_files=/local/path/to/data
```

---

### 情况 B：卡在 "Initializing Actor Worker" / "Initializing Rollout Worker"

**日志显示**：
```
[ACTOR] init_model START | rank=0 | CUDA_VIS=0,1
[ACTOR] Calling _build_model_optimizer()...
[永远卡在这里 - 通常卡 30 秒以上]
```

**可能原因**：
- 模型文件损坏或路径错
- CUDA 编译超时
- GPU 内存不足

**诊断命令**：
```bash
# 检查模型文件
ls -lah /path/to/model/

# 测试模型加载
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
model_path = 'Qwen/Qwen2.5-1.5B'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
print('✓ Model loaded OK')
"

# 检查 GPU 内存
nvidia-smi --query-gpu=memory.free --format=csv
```

**修复方案**：
```bash
# 1. 增加 CUDA 编译超时
export CUDA_LAUNCH_BLOCKING=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# 2. 清理 CUDA 缓存
python -c "import torch; torch.cuda.empty_cache()"

# 3. 减少 GPU 分配（测试）
python main_gkd.py trainer.n_gpus_per_node=1 rollout.n_gpus_per_node=1

# 4. 预热 CUDA（重新启动 Ray 前）
python -c "import torch; torch.cuda.synchronize()"
```

---

### 情况 C：卡在 "Starting weight synchronization"

**日志显示**：
```
[SYNC] Starting weight synchronization...
[SYNC] Actor weight sync completed
[SYNC] Waiting for rollout with timeout=30s...
[永远卡在这里 - NCCL 问题]
```

**这是最常见的问题！**

**可能原因**：
- NCCL 初始化超时（最可能）
- NCCL 通信失败
- GPU 间通信问题

**诊断命令**：
```bash
# 1. 启用完整 NCCL 调试
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,COLL

# 2. 查看 NCCL 日志中的错误
python main_gkd.py ... 2>&1 | grep -i "nccl\|timeout\|error"

# 3. 测试 NCCL
python -c "
import torch
import torch.distributed as dist
torch.cuda.set_device(0)
dist.init_process_group('nccl')
print('✓ NCCL initialized')
"
```

**修复方案**：
```bash
# 最直接的修复：增加超时
export NCCL_TIMEOUT=600  # 从默认 30 秒增加到 600 秒

# 禁用某些 NCCL 功能（如果有兼容性问题）
export NCCL_IB_DISABLE=1  # 禁用 InfiniBand
export NCCL_P2P_DISABLE=1  # 禁用点对点通信

# 使用更大的超时重试
python main_gkd.py data.output_dir=./output 2>&1 | tee train.log
```

---

## 完整调试流程

### 第一次运行（完整诊断）

```bash
# 1. 清理旧进程和缓存
Get-Process python | Stop-Process -Force
ray shutdown
Remove-Item -Path $env:TEMP\nccl* -Force -Recurse

# 2. 启动新的 Ray 集群
ray start --head --num-cpus=8

# 3. 运行诊断脚本
python diagnose_gpu_allocation.py
python test_gkd_components.py

# 4. 设置调试环境
$env:NCCL_DEBUG = "TRACE"
$env:NCCL_TIMEOUT = "600"
$env:TORCH_DISTRIBUTED_DEBUG = "INFO"

# 5. 运行训练（完整日志）
cd recipe/gkd
python main_gkd.py `
  data.output_dir=./output `
  trainer.n_gpus_per_node=2 `
  rollout.n_gpus_per_node=2 `
  2>&1 | tee -a train_debug.log

# 6. 在另一个终端实时监控
tail -f train_debug.log | grep -E "\[ACTOR\]|\[ROLLOUT\]|\[SYNC\]|ERROR"

# 7. 收集诊断信息
tail -200 train_debug.log > diagnosis.log
nvidia-smi > gpu_status.txt
Get-Env | Out-File env_vars.txt
```

### 如果仍然卡死

```bash
# 1. 尝试最小配置
python main_gkd.py `
  trainer.n_gpus_per_node=1 `
  rollout.n_gpus_per_node=1 `
  2>&1 | tee train_minimal.log

# 2. 增加超时
$env:NCCL_TIMEOUT = "900"

# 3. 禁用可能冲突的功能
$env:NCCL_IB_DISABLE = "1"
$env:NCCL_P2P_DISABLE = "1"

# 4. 重新启动 Ray
ray shutdown
ray start --head

# 5. 再次运行
python main_gkd.py ...
```

---

## 关键配置参数速查

### NCCL 参数
| 参数 | 含义 | 调优建议 |
|------|------|---------|
| `NCCL_TIMEOUT` | 超时（秒） | 从 30 增加到 600 |
| `NCCL_DEBUG` | 调试级别 | 设置为 `TRACE` |
| `NCCL_IB_DISABLE` | 禁用 InfiniBand | 有兼容性问题时设置为 1 |
| `NCCL_P2P_DISABLE` | 禁用点对点 | 有兼容性问题时设置为 1 |

### GKD 参数
| 参数 | 默认 | 建议调试值 |
|------|------|------------|
| `trainer.n_gpus_per_node` | 4 | 1 (用于测试) |
| `rollout.n_gpus_per_node` | 4 | 1 (用于测试) |
| `trainer.nnodes` | 1 | 1 |
| `rollout.nnodes` | 1 | 1 |

---

## 日志分析检查清单

运行后检查 `train_debug.log`：

```bash
# ✓ 应该看到
grep "init_model" train_debug.log
grep "Model built\|Rollout built" train_debug.log
grep "weight sync" train_debug.log

# ✗ 不应该看到
grep "ERROR\|Exception\|Failed\|Timeout\|Error" train_debug.log
grep "NCCL.*error\|NCCL.*timeout" train_debug.log
```

---

## 成功标志

当您看到以下输出时，说明卡死问题已解决：

```
[ACTOR] init_model START
[ACTOR] Model loaded successfully
[ROLLOUT] init_model START  
[ROLLOUT] Rollout built successfully
[SYNC] Starting weight synchronization...
[SYNC] ✓ Rollout sync done in X.XXs
[TRAINING] Starting training loop
```

还有实时的进度条更新：
```
training step: 0%|          | 0/100 [00:00<?, ?it/s]
training step: 5%|▌         | 5/100 [00:30<09:30, 5.85s/it]
```

---

## 获取进一步帮助

如果问题仍未解决，请收集并提供：

1. **train_debug.log** - 完整训练日志
2. **GPU 状态** - `nvidia-smi` 输出
3. **环境信息** - `$PSVersionTable` 或 `uname -a`
4. **配置文件** - 您使用的 `on_policy_distill_trainer.yaml`
5. **最后 50 行日志** - 卡住前的最后输出

```bash
# 快速收集诊断包
mkdir diagnosis_$(date +%s)
tail -500 train_debug.log > diagnosis_*/train.log
nvidia-smi > diagnosis_*/gpu_status.txt
$PSVersionTable | Out-File diagnosis_*/system_info.txt
Get-Env | Out-File diagnosis_*/env.txt
zip -r diagnosis.zip diagnosis_*
```

---

## 常见问题 FAQ

**Q: 为什么卡死时没有任何错误信息？**
A: Ray actors 默认不显示完整错误堆栈。启用 `NCCL_DEBUG=TRACE` 可以看到更多细节。

**Q: 增加 NCCL_TIMEOUT 后还是卡死？**
A: 说明不是超时，而是真正的通信失败。检查：
- GPU 间是否有物理连接问题
- 是否有 NCCL 版本不兼容
- PyTorch 是否与 NCCL 版本匹配

**Q: 只用 1 个 GPU 可以正常训练吗？**
A: 可以。用以下命令测试：
```bash
python main_gkd.py trainer.n_gpus_per_node=1 rollout.n_gpus_per_node=1
```
如果这样能成功，说明是多 GPU 通信问题。

**Q: 如何在本地测试不用真实数据？**
A: 检查 `recipe/gkd/config/on_policy_distill_trainer.yaml` 中是否有 `use_dummy_data` 选项。

---

## 下一步行动

1. ✅ **立即**：运行 `python diagnose_gpu_allocation.py`
2. ✅ **立即**：运行 `python test_gkd_components.py`
3. ✅ **现在**：设置 `NCCL_TIMEOUT=600` 并重新运行训练
4. ✅ **如果还是卡**：按照 "情况 A/B/C" 中的步骤诊断
5. ✅ **仍然未解决**：收集诊断包并寻求帮助

---

**创建时间**：2026-01-11
**最后更新**：同上
**支持工具**：
- `diagnose_gpu_allocation.py`
- `test_gkd_components.py`
- `GKD_REALTIME_DEBUGGING.md`
- `GKD_QUICK_FIX_CARD.md`
- `setup_gkd_debugging.py`

