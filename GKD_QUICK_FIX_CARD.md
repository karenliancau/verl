# GKD 卡死快速诊断卡片

## 当前情况
```
(TaskRunner pid=133947) Generating train split: 0 examples [00:00, ? examples/s]
```
✗ 程序卡在数据生成阶段 → 可能是 IO 或初始化问题

---

## 快速诊断（5 分钟）

### 命令 1：环境检查
```bash
python diagnose_gpu_allocation.py --actor-gpus 2 --rollout-gpus 2 --total-gpus 8
```
预期：✓ 8 张 GPU 都可见，Ray 正常

### 命令 2：运行测试
```bash
python test_gkd_components.py
```
预期：✓ CUDA、Ray、vLLM 都可用

### 命令 3：启用调试日志
```bash
$env:NCCL_DEBUG = "TRACE"
$env:NCCL_TIMEOUT = "600"
```

### 命令 4：运行训练（带日志）
```bash
cd recipe/gkd
python main_gkd.py data.output_dir=./output 2>&1 | tee train.log
```

### 命令 5：实时监控
在另一个终端：
```bash
tail -f train.log | grep -E "[ACTOR]|[ROLLOUT]|[SYNC]|ERROR|卡|超时"
```

---

## 卡死症状识别

| 卡死迹象 | 位置 | 原因 |
|---------|------|------|
| `Generating train split` 不动 | 数据加载 | 数据路径错或 IO 慢 |
| `init_model START` 后卡住 | Worker 初始化 | 模型加载或 NCCL 失败 |
| `Waiting for rollout` 后卡住 | 权重同步 | NCCL 通信超时 |

---

## 根据卡死位置快速修复

### 如果卡在数据生成
```bash
# 检查数据路径
ls -la /path/to/your/data

# 检查数据加载
python -c "from datasets import load_dataset; print('✓ datasets library OK')"
```

### 如果卡在 Actor/Rollout 初始化
```bash
# 增加 GPU 初始化超时
$env:NCCL_TIMEOUT = "900"

# 启用完整 NCCL 调试
$env:NCCL_DEBUG = "TRACE"

# 减少 GPU 分配（测试用）
python main_gkd.py trainer.n_gpus_per_node=1 rollout.n_gpus_per_node=1
```

### 如果卡在权重同步
```bash
# 这是最常见的 NCCL 问题
export NCCL_IB_DISABLE=1  # 禁用 InfiniBand 如果有问题
export NCCL_SOCKET_IFNAME=eth0  # 指定网络接口（需要调整）
$env:NCCL_TIMEOUT = "600"  # 增加超时到 10 分钟
```

---

## 日志检查清单

运行训练后，检查 `train.log`：

```bash
# ✓ 应该看到这些
grep "\[ACTOR\]" train.log
grep "\[ROLLOUT\]" train.log
grep "\[SYNC\]" train.log

# ✗ 不应该看到这些
grep "ERROR" train.log
grep "TIMEOUT" train.log
grep "Exception" train.log
grep "Failed" train.log
```

---

## 核心参数

| 参数 | 默认 | 说明 | 调优 |
|------|------|------|------|
| `trainer.n_gpus_per_node` | 4 | Actor GPU 数 | 减少到 1-2 进行测试 |
| `rollout.n_gpus_per_node` | 4 | Rollout GPU 数 | 减少到 1-2 进行测试 |
| `NCCL_TIMEOUT` | 30s | NCCL 超时 | 增加到 600s (10 min) |
| `NCCL_DEBUG` | 关闭 | 调试日志 | 设置为 `TRACE` |

---

## 完整训练命令（推荐配置）

```bash
# 清理旧进程
ray shutdown
sleep 2

# 设置环境
$env:NCCL_DEBUG = "TRACE"
$env:NCCL_TIMEOUT = "600"

# 启动 Ray（可选，通常自动启动）
ray start --head

# 运行训练
cd recipe/gkd
python main_gkd.py `
  data.output_dir=./output `
  trainer.n_gpus_per_node=2 `
  rollout.n_gpus_per_node=2 `
  2>&1 | tee -a train.log
```

---

## 如果 10 分钟后还是卡住

```bash
# 1. 杀死进程
Get-Process python | Stop-Process -Force

# 2. 关闭 Ray
ray shutdown

# 3. 清理缓存
Remove-Item -Path $env:TEMP\nccl* -Force -Recurse
Remove-Item -Path $env:USERPROFILE\.cache\nccl* -Force -Recurse

# 4. 减少配置重试
python main_gkd.py `
  trainer.n_gpus_per_node=1 `
  rollout.n_gpus_per_node=1 `
  2>&1 | tee train.log
```

---

## 生成诊断包

如果仍未解决，生成诊断信息：

```bash
# 收集所有信息
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
New-Item -ItemType Directory -Path "diagnosis_$timestamp" | Out-Null

# 日志
Copy-Item train.log "diagnosis_$timestamp\"

# GPU 状态
nvidia-smi | Out-File "diagnosis_$timestamp\gpu_status.txt"

# Python 环境
python -m pip list | Out-File "diagnosis_$timestamp\packages.txt"

# 配置
Copy-Item recipe/gkd/config/on_policy_distill_trainer.yaml "diagnosis_$timestamp\"

Write-Host "Diagnostic package created: diagnosis_$timestamp"
```

---

## 常用命令速查

```bash
# 监控日志
tail -f train.log | grep -v "^$"

# 查看 GPU 使用
watch -n 1 nvidia-smi

# 查看进程
ps aux | grep python

# 杀死训练
pkill -f main_gkd.py

# Ray 状态
python -c "import ray; ray.get_runtime_context()" 2>/dev/null || echo "Ray not initialized"

# NCCL 检查
python -c "import torch; torch.distributed.launch --help" 2>/dev/null && echo "✓ NCCL available"
```

---

## 何时寻求帮助

收集以下信息并提交：

1. **train.log** - 完整训练日志（最后 500 行）
2. **nvidia-smi 输出** - GPU 状态
3. **配置文件** - 你的 `on_policy_distill_trainer.yaml`
4. **系统信息** - `uname -a` 或 Windows 版本
5. **错误信息** - 任何 ERROR 或 Exception

---

**最后更新：2026-01-11**
