# 🎯 GKD 卡死诊断工具包 - 使用说明

## 👋 欢迎

您遇到的 GKD 训练卡死问题（"Generating train split" 后无响应）已经有了完整的诊断解决方案。

本文档将指导您快速诊断和解决这个问题。

---

## ⚡ 极速开始（推荐）

### 第 1 步：快速诊断（2 分钟）

```bash
python diagnose_gkd_quick.py
```

这个脚本会：
- ✅ 检查 GPU 分配
- ✅ 验证关键组件
- ✅ 显示下一步操作

### 第 2 步：设置调试环境

**在 Windows PowerShell 中：**

```powershell
$env:NCCL_DEBUG = "TRACE"
$env:NCCL_TIMEOUT = "600"
```

### 第 3 步：运行训练并观察

```bash
cd recipe/gkd
python main_gkd.py data.output_dir=./output 2>&1 | tee train.log
```

### 第 4 步：在另一个终端实时监控

```bash
# PowerShell
Get-Content -Path train.log -Wait | Select-String -Pattern "[ACTOR]|[ROLLOUT]|[SYNC]|ERROR"

# 或使用 bash
tail -f train.log | grep -E "[ACTOR]|[ROLLOUT]|[SYNC]|ERROR"
```

### 第 5 步：分析结果

**✅ 正常情况**（应该看到这些日志）：
```
[ACTOR] init_model START
[ACTOR] Model loaded successfully
[ROLLOUT] init_model START
[ROLLOUT] Rollout built successfully
[SYNC] Starting weight synchronization...
[SYNC] ✓ Rollout sync done in XXs
```

**❌ 卡死情况**（某一步长时间不输出）：
```
[ACTOR] init_model START
[ACTOR] Model loaded successfully
[ROLLOUT] init_model START
[永远卡在这里 - 2 分钟以上没有下一行]
```

---

## 📊 根据卡死位置快速修复

### 情况 1️⃣：卡在 "Generating train split"

```bash
# 检查数据文件是否存在
ls -lah /path/to/your/data

# 测试数据加载库
python -c "from datasets import load_dataset; print('✓ OK')"
```

### 情况 2️⃣：卡在 Actor/Rollout 初始化（最常见）

```bash
# 最常见的修复：增加超时和禁用 InfiniBand
$env:NCCL_TIMEOUT = "600"
$env:NCCL_IB_DISABLE = "1"

# 重新运行训练
python main_gkd.py data.output_dir=./output
```

### 情况 3️⃣：卡在权重同步（第二常见）

```bash
# 这通常是 NCCL 超时问题
$env:NCCL_TIMEOUT = "600"
$env:NCCL_DEBUG = "TRACE"

# 如果还是不行，减少 GPU 用于测试
python main_gkd.py \
  trainer.n_gpus_per_node=1 \
  rollout.n_gpus_per_node=1 \
  2>&1 | tee train_test.log
```

---

## 📚 详细诊断资源

### 快速参考（推荐）
📄 **`GKD_QUICK_FIX_CARD.md`**
- 快速查阅
- 常见问题和快速修复
- 参数速查表
- ~200 行，5 分钟读完

### 详细诊断指南
📄 **`GKD_REALTIME_DEBUGGING.md`**
- 分步骤诊断流程
- 四种卡死情况（A/B/C/D）
- 每种情况的诊断命令和修复方案
- ~400 行，详细但容易跟随

### 完整参考手册
📄 **`GKD_DEBUGGING_COMPLETE_GUIDE.md`**
- 综合所有诊断方法
- FAQ 常见问题
- 完整流程检查清单
- ~500 行，全面但篇幅较长

### GPU 架构分析
📄 **`GKD_GPU_ALLOCATION_ANALYSIS.md`**
- 解释 GPU 分配机制
- ResourcePoolManager 工作原理
- 模型加载分析
- 如果需要理解"为什么"

### 代码参考
📄 **`gkd_diagnostic_patches.py`**
- 代码补丁示例
- 如何添加日志到您的代码
- 诊断函数示例

---

## 🔧 诊断工具清单

| 工具 | 用途 | 运行方式 |
|------|------|---------|
| `diagnose_gkd_quick.py` | 一键诊断 | `python diagnose_gkd_quick.py` |
| `diagnose_gpu_allocation.py` | 详细 GPU 检查 | `python diagnose_gpu_allocation.py` |
| `test_gkd_components.py` | 组件可用性测试 | `python test_gkd_components.py` |
| `setup_gkd_debugging.py` | 设置调试模式 | `python setup_gkd_debugging.py --enable` |

---

## 💡 关键知识

### 最常见的卡死原因：NCCL 超时

NCCL（NVIDIA Collective Communications Library）在多 GPU 权重同步时可能超时。

**快速修复**：
```powershell
$env:NCCL_TIMEOUT = "600"  # 从 30 秒增加到 600 秒
```

### 第二常见的原因：GPU 初始化

vLLM/SGLang 推理引擎初始化可能缓慢。

**快速修复**：
```powershell
$env:CUDA_LAUNCH_BLOCKING = "1"  # 同步 GPU 操作
$env:NCCL_DEBUG = "TRACE"  # 查看详细日志
```

### 第三常见的原因：多 GPU 通信问题

某些系统上 InfiniBand 可能有兼容性问题。

**快速修复**：
```powershell
$env:NCCL_IB_DISABLE = "1"  # 禁用 InfiniBand，使用以太网
```

---

## ✅ 成功标志

当您看到类似这样的输出时，说明卡死已解决：

```
[ACTOR] init_model START | rank=0 | CUDA_VIS=0,1
[ACTOR] Model loaded successfully
[ROLLOUT] init_model START | rank=0 | CUDA_VIS=2,3
[ROLLOUT] Rollout built successfully
[SYNC] Starting weight synchronization...
[SYNC] ✓ Rollout sync done in 2.34s
[TRAINING] Starting training loop

training step:   0%|          | 0/100 [00:00<?, ?it/s]
training step:   5%|▌         | 5/100 [00:30<09:30, 5.85s/it]
training step:  10%|█         | 10/100 [01:00<09:00, 5.85s/it]
...
```

---

## 🆘 如果仍然卡死

### 尝试顺序

1. **增加超时**（最有效）
   ```powershell
   $env:NCCL_TIMEOUT = "600"
   ```

2. **启用完整调试**
   ```powershell
   $env:NCCL_DEBUG = "TRACE"
   $env:TORCH_DISTRIBUTED_DEBUG = "INFO"
   ```

3. **禁用 InfiniBand**
   ```powershell
   $env:NCCL_IB_DISABLE = "1"
   ```

4. **清理缓存并重启**
   ```powershell
   ray shutdown
   Remove-Item -Path $env:TEMP\nccl* -Force -Recurse
   ray start --head
   ```

5. **减少 GPU 配置测试**
   ```powershell
   python main_gkd.py trainer.n_gpus_per_node=1 rollout.n_gpus_per_node=1
   ```

### 收集诊断信息

如果上述方法都不行，请收集这些信息：

```bash
# 最后 200 行日志
tail -200 train.log > diagnosis_log.txt

# GPU 状态
nvidia-smi > gpu_status.txt

# 您的配置文件
copy recipe/gkd/config/on_policy_distill_trainer.yaml diagnosis_config.yaml

# 系统信息
systeminfo > system_info.txt

# 创建诊断包
mkdir gkd_diagnosis_$(date +%s)
move diagnosis_*.txt gkd_diagnosis_*/
move gpu_status.txt gkd_diagnosis_*/
move system_info.txt gkd_diagnosis_*/
```

---

## 📞 何时寻求帮助

当您：
- ✅ 已运行 `diagnose_gkd_quick.py`
- ✅ 已设置 `NCCL_TIMEOUT=600` 和 `NCCL_DEBUG=TRACE`
- ✅ 已收集完整日志
- ✅ 仍然卡死 > 5 分钟

那么请参考 `GKD_DEBUGGING_COMPLETE_GUIDE.md` 的"获取进一步帮助"部分。

---

## 🎓 学习资源

### 理解 GPU 分配

如果您想了解 GKD 如何分配 GPU，请阅读：
- `GKD_GPU_ALLOCATION_ANALYSIS.md`

这个文档解释：
- ResourcePoolManager 如何工作
- Actor 和 Rollout 如何使用不同的 GPU
- 为什么 Teacher 是独立启动的

### 理解诊断日志

日志中的关键标记：
- `[ACTOR]` - Actor Worker 日志
- `[ROLLOUT]` - Rollout Worker 日志
- `[SYNC]` - 权重同步日志
- `[NCCL]` - NCCL 通信日志
- `ERROR` - 错误信息
- `TIMEOUT` - 超时警告

---

## 🚀 快速命令速查

```bash
# 一键诊断
python diagnose_gkd_quick.py

# 设置超时（最常见的修复）
$env:NCCL_TIMEOUT = "600"

# 运行训练
cd recipe/gkd
python main_gkd.py data.output_dir=./output 2>&1 | tee train.log

# 实时监控
Get-Content -Path train.log -Wait | Select-String -Pattern "ACTOR|ROLLOUT|SYNC|ERROR"

# 清理缓存
ray shutdown ; Remove-Item -Path $env:TEMP\nccl* -Force -Recurse

# 重启 Ray
ray start --head --num-cpus=8

# 减少 GPU 配置测试
python main_gkd.py trainer.n_gpus_per_node=1 rollout.n_gpus_per_node=1
```

---

## ✨ 最后提示

1. **不要急**：诊断通常需要几分钟，仔细查看日志
2. **记录日志**：使用 `2>&1 | tee train.log` 保存日志便于后续分析
3. **查看关键日志**：重点关注 `[ACTOR]`, `[ROLLOUT]`, `[SYNC]`, `ERROR` 这些关键词
4. **如果不确定**：默认增加 `NCCL_TIMEOUT` 总是有帮助的

---

## 📖 文档导航

```
开始诊断
    ↓
运行 diagnose_gkd_quick.py
    ↓
是否成功？
    ├─ 是 → 继续设置训练
    └─ 否 → 参考 GKD_QUICK_FIX_CARD.md
    
运行训练
    ↓
是否卡死？
    ├─ 否 → 恭喜！训练成功
    └─ 是 → 查看日志，找出卡死位置
    
根据卡死位置
    ├─ 快速答案？ → GKD_QUICK_FIX_CARD.md
    ├─ 详细诊断？ → GKD_REALTIME_DEBUGGING.md
    ├─ 完整指南？ → GKD_DEBUGGING_COMPLETE_GUIDE.md
    └─ 架构问题？ → GKD_GPU_ALLOCATION_ANALYSIS.md
```

---

## 📝 文件清单

本诊断工具包包含：

### 脚本文件
- ✅ `diagnose_gkd_quick.py` - 一键诊断脚本
- ✅ `diagnose_gpu_allocation.py` - GPU 检查脚本
- ✅ `test_gkd_components.py` - 组件测试脚本
- ✅ `setup_gkd_debugging.py` - 调试设置脚本

### 文档文件
- ✅ `GKD_QUICK_FIX_CARD.md` - 快速参考卡
- ✅ `GKD_REALTIME_DEBUGGING.md` - 实时调试指南
- ✅ `GKD_DEBUGGING_COMPLETE_GUIDE.md` - 完整诊断指南
- ✅ `GKD_GPU_ALLOCATION_ANALYSIS.md` - GPU 架构分析
- ✅ `GKD_DIAGNOSTIC_TOOLKIT_INDEX.md` - 工具包索引
- ✅ `GKD_DIAGNOSIS_README.md` - 本文件

---

**工具包版本**：1.0
**创建日期**：2026-01-11
**最后更新**：同上

**祝您诊断顺利！** 🎉

如有问题，请按照本文档的指引进行诊断。
大多数情况下，增加 `NCCL_TIMEOUT` 就能解决问题。

