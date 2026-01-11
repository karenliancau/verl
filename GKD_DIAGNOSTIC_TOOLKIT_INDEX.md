# GKD 诊断工具包清单

## 📦 已创建的诊断文件

### 核心诊断脚本

#### 1. `diagnose_gpu_allocation.py` ✅
**位置**：`d:\code\verl\diagnose_gpu_allocation.py`

**功能**：
- 检查 CUDA 和 GPU 可用性
- 验证 Ray 集群配置
- 模拟 ResourcePool GPU 分配
- 检查 PyTorch 分布式设置
- 分析模型加载内存需求

**使用**：
```bash
python diagnose_gpu_allocation.py --actor-gpus 2 --rollout-gpus 2 --total-gpus 8
```

**输出**：
- CUDA 设备信息
- Ray 集群资源
- GPU 分配模拟
- 预期 vs 实际 GPU 分配
- 内存需求分析

---

#### 2. `test_gkd_components.py` ✅
**位置**：`d:\code\verl\test_gkd_components.py` (由 setup_gkd_debugging.py 创建)

**功能**：
- 测试 CUDA 可用性
- 测试 Ray 集群
- 测试 vLLM/SGLang
- 测试模型加载（可选）

**使用**：
```bash
python test_gkd_components.py
```

---

#### 3. `setup_gkd_debugging.py` ✅
**位置**：`d:\code\verl\setup_gkd_debugging.py`

**功能**：
- 自动检测调试就绪状态
- 创建调试包装脚本
- 创建组件测试脚本

**使用**：
```bash
python setup_gkd_debugging.py --enable
```

**创建的文件**：
- `run_gkd_debug.py` - 调试包装脚本
- `test_gkd_components.py` - 组件测试脚本

---

#### 4. `recipe/gkd/gkd_diagnostic_patches.py` ✅
**位置**：`d:\code\verl\recipe\gkd\gkd_diagnostic_patches.py`

**功能**：
- 提供代码补丁示例
- 展示如何添加日志
- 提供诊断函数

**内容**：
- `log_gpu_info()` - GPU 信息日志函数
- `init_model_actor_with_logging()` - Actor 初始化示例
- `init_model_rollout_with_logging()` - Rollout 初始化示例
- `sync_rollout_weights_with_logging()` - 权重同步示例

**使用方式**：
参考文件中的注释，手动复制相应部分到实际代码中

---

### 诊断文档

#### 5. `GKD_REALTIME_DEBUGGING.md` ✅
**位置**：`d:\code\verl\GKD_REALTIME_DEBUGGING.md`

**内容**：
- Phase 1：定位卡死的确切位置
  - 添加时间戳日志
  - 观察关键函数
  - 识别卡死点
  
- Phase 2：根据卡死位置诊断
  - 情况 A：数据生成卡死
  - 情况 B：Actor 初始化卡死
  - 情况 C：Rollout 初始化卡死
  - 情况 D：权重同步卡死
  
- Phase 3：根据诊断结果采取行动

**篇幅**：详细的分步骤诊断指南（约 400 行）

---

#### 6. `GKD_QUICK_FIX_CARD.md` ✅
**位置**：`d:\code\verl\GKD_QUICK_FIX_CARD.md`

**内容**：
- 快速诊断（5 分钟）
- 卡死症状识别表
- 根据卡死位置的快速修复
- 日志检查清单
- 核心参数速查表
- 常用命令速查
- 何时寻求帮助

**篇幅**：精简参考卡片（约 200 行）

---

#### 7. `GKD_DEBUGGING_COMPLETE_GUIDE.md` ✅
**位置**：`d:\code\verl\GKD_DEBUGGING_COMPLETE_GUIDE.md`

**内容**：
- 概览和快速开始
- 诊断工具列表
- 三种情况的完整诊断方案（A/B/C）
- 完整调试流程
- 关键参数速查
- 日志分析清单
- 成功标志
- 常见问题 FAQ
- 下一步行动

**篇幅**：综合参考指南（约 500 行）

---

#### 8. `GKD_GPU_ALLOCATION_ANALYSIS.md` ✅
**位置**：`d:\code\verl\GKD_GPU_ALLOCATION_ANALYSIS.md`

**内容**（之前创建的）：
- GPU 分配架构详解
- ResourcePoolManager 工作原理
- 模型加载分析
- 诊断方法
- 性能优化建议

---

### 之前创建的文件（参考）

- `GKD_DEADLOCK_DIAGNOSIS.md` - 早期诊断（已部分过时）
- `GKD_COMPLETE_SOLUTION.md` - 完整解决方案汇总

---

## 🎯 使用流程

### 快速诊断流程（推荐）

```
1. 运行诊断脚本
   ↓
   python diagnose_gpu_allocation.py
   
2. 查看结果
   ↓
   GPU 分配是否正确？
   
3. 如果 GPU 分配正确
   ↓
   参考 GKD_QUICK_FIX_CARD.md
   
4. 如果需要详细诊断
   ↓
   参考 GKD_REALTIME_DEBUGGING.md
   
5. 如果需要完整指南
   ↓
   参考 GKD_DEBUGGING_COMPLETE_GUIDE.md
```

### 完整诊断流程（遇到卡死时）

```
1. 设置调试环境
   $env:NCCL_DEBUG = "TRACE"
   $env:NCCL_TIMEOUT = "600"
   
2. 运行训练
   python main_gkd.py data.output_dir=./output 2>&1 | tee train.log
   
3. 实时监控
   tail -f train.log | grep -E "[ACTOR]|[ROLLOUT]|[SYNC]|ERROR"
   
4. 找到卡死位置
   检查日志何处停止
   
5. 根据位置参考 GKD_REALTIME_DEBUGGING.md
   找到对应的"情况 A/B/C/D"
   
6. 应用对应的诊断和修复方案
   按步骤执行
   
7. 如果仍未解决
   收集诊断包并寻求帮助
```

---

## 📋 文件用途矩阵

| 文件 | 脚本? | 诊断? | 参考? | 用于何时 |
|------|--------|--------|--------|---------|
| diagnose_gpu_allocation.py | ✅ | ✅ | | 快速检查 GPU 分配 |
| test_gkd_components.py | ✅ | ✅ | | 测试关键组件 |
| setup_gkd_debugging.py | ✅ | | | 设置调试环境 |
| gkd_diagnostic_patches.py | | | ✅ | 参考代码补丁 |
| GKD_REALTIME_DEBUGGING.md | | ✅ | ✅ | 详细分步诊断 |
| GKD_QUICK_FIX_CARD.md | | | ✅ | 快速查阅参考 |
| GKD_DEBUGGING_COMPLETE_GUIDE.md | | ✅ | ✅ | 综合诊断指南 |
| GKD_GPU_ALLOCATION_ANALYSIS.md | | | ✅ | GPU 架构分析 |

---

## 🚀 立即开始

### 第 1 步（2 分钟）

```bash
# 快速检查 GPU
python diagnose_gpu_allocation.py --actor-gpus 2 --rollout-gpus 2 --total-gpus 8
```

**预期**：✓ 8 张 GPU 都被识别

### 第 2 步（5 分钟）

```bash
# 测试组件
python test_gkd_components.py
```

**预期**：✓ CUDA、Ray、vLLM 都可用

### 第 3 步（设置环境）

```bash
# PowerShell
$env:NCCL_DEBUG = "TRACE"
$env:NCCL_TIMEOUT = "600"
```

### 第 4 步（运行训练）

```bash
cd recipe/gkd
python main_gkd.py data.output_dir=./output 2>&1 | tee train.log
```

### 第 5 步（实时监控）

在另一个终端：

```bash
tail -f train.log | grep -E "[ACTOR]|[ROLLOUT]|[SYNC]|ERROR"
```

---

## 📊 诊断决策树

```
开始训练
  ↓
程序卡死？
  ├─ 否 → ✅ 成功！
  └─ 是
     ↓
     查看日志，找出卡死位置
     ↓
     ├─ 卡在"数据生成"
     │  └─ 参考 GKD_REALTIME_DEBUGGING.md → 情况 A
     │
     ├─ 卡在"Actor 初始化"
     │  └─ 参考 GKD_REALTIME_DEBUGGING.md → 情况 B
     │
     ├─ 卡在"Rollout 初始化"
     │  └─ 参考 GKD_REALTIME_DEBUGGING.md → 情况 C
     │
     ├─ 卡在"权重同步"
     │  └─ 参考 GKD_REALTIME_DEBUGGING.md → 情况 D
     │     → 最可能：增加 NCCL_TIMEOUT=600
     │
     └─ 其他位置
        └─ 参考 GKD_DEBUGGING_COMPLETE_GUIDE.md
```

---

## ✨ 核心诊断命令速查

```bash
# 1. GPU 检查
python diagnose_gpu_allocation.py

# 2. 组件测试
python test_gkd_components.py

# 3. 设置调试
$env:NCCL_DEBUG = "TRACE"
$env:NCCL_TIMEOUT = "600"

# 4. 运行训练
cd recipe/gkd
python main_gkd.py data.output_dir=./output 2>&1 | tee train.log

# 5. 实时监控
tail -f train.log | grep -E "[ACTOR]|[ROLLOUT]|[SYNC]|ERROR"

# 6. Ray 状态
python -c "import ray; print(ray.cluster_resources())"

# 7. 进程监控
ps aux | grep python | grep main_gkd

# 8. 清理缓存（如需要）
ray shutdown
Remove-Item -Path $env:TEMP\nccl* -Force -Recurse
```

---

## 💡 关键提示

1. **最常见的卡死原因**：NCCL 超时
   - 快速修复：`$env:NCCL_TIMEOUT = "600"`

2. **如果不确定从哪里开始**：
   - 先运行 `diagnose_gpu_allocation.py`
   - 然后按照 `GKD_QUICK_FIX_CARD.md` 操作

3. **如果需要详细诊断**：
   - 参考 `GKD_REALTIME_DEBUGGING.md` 的 Phase 1-2-3

4. **如果仍然未解决**：
   - 收集 train.log、nvidia-smi、配置文件
   - 参考 `GKD_DEBUGGING_COMPLETE_GUIDE.md` 的"获取进一步帮助"部分

---

## 📞 支持

如遇问题，请按以下顺序参考：

1. **快速答案** → `GKD_QUICK_FIX_CARD.md`
2. **分步诊断** → `GKD_REALTIME_DEBUGGING.md`
3. **完整指南** → `GKD_DEBUGGING_COMPLETE_GUIDE.md`
4. **架构分析** → `GKD_GPU_ALLOCATION_ANALYSIS.md`
5. **代码参考** → `gkd_diagnostic_patches.py`

---

**工具包创建时间**：2026-01-11
**包含文件数**：8 个
**总行数**：~2000 行诊断代码和文档

