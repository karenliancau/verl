# GKD 卡死实时调试指南

## 您的情况

从日志看，程序卡在了 **"Generating train split"** 阶段。这是在数据加载后，但在实际训练开始前。

```
(TaskRunner pid=133947)   Generating train split: 0 examples [00:00, ? examples/s]
```

这表明卡死发生在**数据集生成或初始化阶段**，而不是训练循环。

---

## 实时调试计划

### Phase 1：定位卡死的确切位置

**目标**：找出是哪一步卡住了

#### 步骤 1.1：在关键位置添加时间戳日志

编辑 `recipe/gkd/main_gkd.py`，在主函数中添加：

```python
import time
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(name)s - %(message)s')

def main():
    logger.info("="*70)
    logger.info("GKD TRAINING START")
    logger.info("="*70)
    
    # ... 现有代码：配置加载 ...
    logger.info("✓ Configuration loaded")
    
    # ... 现有代码：数据加载 ...
    start_time = time.time()
    logger.info("Starting dataset generation...")
    
    # data_manager = ...
    logger.info(f"✓ Dataset generation completed in {time.time() - start_time:.1f}s")
    
    # ... 现有代码：资源池创建 ...
    start_time = time.time()
    logger.info("Creating ResourcePool...")
    # resource_pool_manager = ...
    logger.info(f"✓ ResourcePool created in {time.time() - start_time:.1f}s")
    
    # ... 现有代码：Actor Worker 初始化 ...
    start_time = time.time()
    logger.info("Initializing Actor Worker...")
    # actor_wg = ...
    # ray.get(actor_wg.init_model.remote())
    logger.info(f"✓ Actor Worker initialized in {time.time() - start_time:.1f}s")
    
    # ... 现有代码：Rollout Worker 初始化 ...
    start_time = time.time()
    logger.info("Initializing Rollout Worker...")
    # rollout_wg = ...
    # ray.get(rollout_wg.init_model.remote())
    logger.info(f"✓ Rollout Worker initialized in {time.time() - start_time:.1f}s")
    
    # ... 现有代码：权重同步 ...
    start_time = time.time()
    logger.info("Starting weight synchronization...")
    # ray.get(rollout_wg.sync_rollout_weights.remote(actor_wg))
    logger.info(f"✓ Weight synchronization completed in {time.time() - start_time:.1f}s")
    
    # ... 现有代码：训练循环 ...
    logger.info("Starting training loop...")
    
if __name__ == "__main__":
    main()
```

#### 步骤 1.2：运行训练并观察日志

```bash
cd recipe/gkd
python main_gkd.py data.output_dir=./output 2>&1 | tee train_debug.log
```

然后在另一个终端监控：

```bash
# 实时查看关键日志
tail -f train_debug.log | grep -E "✓|Starting|ERROR|NCCL|Traceback"
```

**预期输出序列**：
```
[2026-01-11 10:00:00] root - GKD TRAINING START
[2026-01-11 10:00:05] root - ✓ Configuration loaded
[2026-01-11 10:00:10] root - Starting dataset generation...
[2026-01-11 10:00:30] root - ✓ Dataset generation completed in 20.1s
[2026-01-11 10:00:30] root - Creating ResourcePool...
[2026-01-11 10:00:35] root - ✓ ResourcePool created in 5.2s
[2026-01-11 10:00:35] root - Initializing Actor Worker...
[2026-01-11 10:00:50] root - ✓ Actor Worker initialized in 15.1s
[2026-01-11 10:00:50] root - Initializing Rollout Worker...
[2026-01-11 10:01:20] root - ✓ Rollout Worker initialized in 30.5s
[2026-01-11 10:01:20] root - Starting weight synchronization...
[2026-01-11 10:01:25] root - ✓ Weight synchronization completed in 5.2s
[2026-01-11 10:01:25] root - Starting training loop...
```

**如果卡在某一步，日志会停止**，比如：
```
[2026-01-11 10:00:50] root - Initializing Rollout Worker...
[永不停止 - 表示卡在这里]
```

---

### Phase 2：根据卡死位置诊断

#### 情况 A：卡在 "Generating train split"

**日志显示**：
```
Starting dataset generation...
Generating train split: 0 examples [00:00, ? examples/s]
[卡住，没有下一行]
```

**可能原因**：
1. 数据文件路径错误
2. 数据集加载库（HF datasets）卡住
3. 磁盘 IO 问题

**诊断命令**：
```bash
# 检查数据文件是否存在
ls -lah /path/to/your/data

# 检查数据加载库
python -c "from datasets import load_dataset; ds = load_dataset('wikitext', 'wikitext-2'); print(len(ds))"
```

---

#### 情况 B：卡在 "Initializing Actor Worker"

**日志显示**：
```
Creating ResourcePool...
✓ ResourcePool created in 5.2s
Initializing Actor Worker...
[卡住]
```

**可能原因**：
1. Actor Worker init_model() 失败
2. Megatron 分布式初始化卡住
3. NCCL 超时

**诊断方法**：

在 `recipe/gkd/megatron_workers.py` 的 `MegatronOnPolicyDistillActorWorker.init_model()` 中添加：

```python
def init_model(self):
    import logging
    import time
    logger = logging.getLogger(__name__)
    
    logger.info(f"[ACTOR] init_model START | pid={os.getpid()} | rank={self.local_rank}")
    logger.info(f"[ACTOR] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
    
    start = time.time()
    try:
        logger.info("[ACTOR] Calling _build_model_optimizer()...")
        self._build_model_optimizer()
        logger.info(f"[ACTOR] ✓ Model built in {time.time()-start:.1f}s")
    except Exception as e:
        logger.error(f"[ACTOR] ✗ _build_model_optimizer failed: {e}", exc_info=True)
        raise
    
    start = time.time()
    try:
        logger.info("[ACTOR] Calling _init_distributed()...")
        self._init_distributed()
        logger.info(f"[ACTOR] ✓ Distributed initialized in {time.time()-start:.1f}s")
    except Exception as e:
        logger.error(f"[ACTOR] ✗ _init_distributed failed: {e}", exc_info=True)
        raise
    
    logger.info("[ACTOR] init_model DONE")
```

---

#### 情况 C：卡在 "Initializing Rollout Worker"

**日志显示**：
```
✓ Actor Worker initialized in 15.1s
Initializing Rollout Worker...
[卡住 - 通常卡 1-2 分钟]
```

**这是最常见的情况**。可能原因：
1. vLLM/SGLang 初始化卡住
2. NCCL 初始化超时
3. 推理引擎编译超时

**诊断方法**：

在 `recipe/gkd/megatron_workers.py` 的 `MegatronOnPolicyDistillRolloutWorker.init_model()` 中添加：

```python
def init_model(self):
    import logging
    import time
    logger = logging.getLogger(__name__)
    
    logger.info(f"[ROLLOUT] init_model START | pid={os.getpid()} | rank={self.local_rank}")
    logger.info(f"[ROLLOUT] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
    
    start = time.time()
    try:
        logger.info("[ROLLOUT] Calling _build_rollout()...")
        self._build_rollout()
        logger.info(f"[ROLLOUT] ✓ Rollout built in {time.time()-start:.1f}s")
    except Exception as e:
        logger.error(f"[ROLLOUT] ✗ _build_rollout failed: {e}", exc_info=True)
        raise
    
    logger.info("[ROLLOUT] init_model DONE")
```

---

#### 情况 D：卡在 "Starting weight synchronization"

**日志显示**：
```
✓ Actor Worker initialized in 15.1s
✓ Rollout Worker initialized in 30.5s
Starting weight synchronization...
[卡住 - 表示 ray.get() 没有返回]
```

**最可能的原因**：NCCL 权重同步超时

**诊断方法**：

在 `recipe/gkd/ray_trainer.py` 的 `sync_rollout_weights()` 中修改：

```python
def sync_rollout_weights(self):
    assert not self.hybrid_engine
    import logging
    import time
    logger = logging.getLogger(__name__)
    
    logger.info("[SYNC] Starting weight synchronization")
    
    start = time.time()
    try:
        logger.info("[SYNC] Syncing actor weights...")
        self.actor_wg.sync_rollout_weights()
        logger.info(f"[SYNC] ✓ Actor sync done in {time.time()-start:.1f}s")
    except Exception as e:
        logger.error(f"[SYNC] ✗ Actor sync failed: {e}")
        raise
    
    start = time.time()
    try:
        logger.info("[SYNC] Waiting for rollout weights (timeout=30s)...")
        # 👇 关键：添加超时！
        timeout = 30
        ray.get(self.rollout_wg.sync_rollout_weights(), timeout=timeout)
        logger.info(f"[SYNC] ✓ Rollout sync done in {time.time()-start:.1f}s")
    except ray.exceptions.GetTimeoutError as e:
        logger.error(f"[SYNC] ✗ Rollout sync TIMEOUT after {timeout}s")
        logger.error("[SYNC] This likely means NCCL communication is stuck")
        raise
    except Exception as e:
        logger.error(f"[SYNC] ✗ Rollout sync failed: {e}")
        raise
```

---

### Phase 3：根据诊断结果采取行动

| 卡死位置 | 原因 | 解决方案 |
|---------|------|---------|
| 数据生成 | 数据路径错 | 检查 `data.train_files` 配置 |
| Actor 初始化 | 模型加载失败 | 检查模型文件，增加日志到 `_build_model_optimizer()` |
| Rollout 初始化 | vLLM 卡住 | 启用 `NCCL_DEBUG=TRACE`，增加超时 |
| 权重同步 | NCCL 超时 | 设置 `export NCCL_TIMEOUT=600` |

---

## 完整调试命令集

### 1. 清理旧进程

```bash
# 杀死所有 Python 训练进程
ps aux | grep python | grep -v grep | awk '{print $2}' | xargs kill -9

# 关闭 Ray 集群
ray shutdown

# 清理 NCCL 缓存
rm -rf /tmp/nccl* ~/.cache/nccl*
```

### 2. 设置调试环境

```bash
# PowerShell
$env:NCCL_DEBUG = "TRACE"
$env:NCCL_TIMEOUT = "600"
$env:TORCH_DISTRIBUTED_DEBUG = "INFO"
$env:VLLM_LOGGING_LEVEL = "DEBUG"
```

### 3. 启动 Ray 集群

```bash
ray shutdown
ray start --head --num-cpus=8
```

### 4. 运行训练（带完整日志）

```bash
cd recipe/gkd
python main_gkd.py \
  data.output_dir=./output \
  trainer.n_gpus_per_node=2 \
  rollout.n_gpus_per_node=2 \
  2>&1 | tee -a train_debug.log
```

### 5. 在另一个终端实时监控

```bash
# 监控关键日志
tail -f train_debug.log | grep -E "\[ACTOR\]|\[ROLLOUT\]|\[SYNC\]|✓|✗|ERROR|NCCL"

# 或者查看完整日志
tail -100 train_debug.log
```

### 6. 监控 GPU 使用

```bash
# 在第三个终端
watch -n 1 nvidia-smi
```

---

## 日志分析检查清单

运行后检查 `train_debug.log`：

- [ ] 有 `[ACTOR] init_model START` 日志吗？
- [ ] 有 `[ACTOR] ✓ Model built` 日志吗？
- [ ] 有 `[ROLLOUT] init_model START` 日志吗？
- [ ] 有 `[ROLLOUT] ✓ Rollout built` 日志吗？
- [ ] 有 `[SYNC] Starting weight synchronization` 日志吗？
- [ ] 有 `ERROR` 或 `Exception` 字样吗？
- [ ] 有 `NCCL` 错误吗？
- [ ] 有 `Timeout` 字样吗？

---

## 收集信息用于进一步诊断

如果卡死，请收集：

```bash
# 1. 完整日志
tail -200 train_debug.log > diagnosis_log.txt

# 2. GPU 状态
nvidia-smi > gpu_status.txt

# 3. Ray 状态
python -c "import ray; print(ray.get_runtime_context())" > ray_status.txt

# 4. 环境变量
env | grep -i "cuda\|nccl\|rank" > env_vars.txt
```

然后分享这些文件。

---

## 预期卡死时间

- **正常情况**：从启动到训练循环 < 3 分钟
- **缓慢情况**：3-10 分钟（模型加载或编译慢）
- **超过 10 分钟**：肯定有问题，是卡死而不是慢

如果看到进度条长时间不动（> 2 分钟），就是卡死了。

---

## 快速修复清单

如果确实卡死，尝试以下顺序：

1. **增加超时**：`export NCCL_TIMEOUT=600`
2. **启用 NCCL 调试**：`export NCCL_DEBUG=TRACE`
3. **减少 GPU 分配**：改用 `trainer.n_gpus_per_node=1 rollout.n_gpus_per_node=1`
4. **清理缓存**：`ray shutdown && rm -rf /tmp/nccl* ~/.cache/nccl*`
5. **重启 Ray**：`ray start --head --num-cpus=8`
6. **重新运行**

---

## 下一步

1. **立即执行** Phase 1 和 Phase 2 的步骤
2. **收集日志** 并分析卡死位置
3. **根据情况** 应用相应的诊断或修复
4. **反馈结果** 和日志内容

