## 🎯 GKD训练卡死 - 根本原因与快速修复

### 核心发现

**你的判断是对的！** 卡死确实发生在**Rollout阶段**，而非Teacher。问题是：

```python
# recipe/gkd/ray_trainer.py 第329行
def sync_rollout_weights(self):
    ray.get(self.rollout_wg.sync_rollout_weights())  # ❌ 这里永久阻塞
```

这个 `ray.get()` 是**同步阻塞调用**，会无限期等待所有3个Rollout Worker完成权重同步。

### 卡死流程

```
数据加载 ✅
  ↓
_async_gen_next_batch() 被调用
  ↓
sync_rollout_weights() 被调用
  ↓
ray.get() 等待 3 个 rollout worker 同步权重
  ↓
如果任何 worker 出现问题：
  - 初始化失败
  - NCCL 通信超时
  - GPU 内存不足
  - 权重同步错误
  ↓
永久阻塞 ❌ 没有超时机制
  ↓
程序卡死
```

### 已应用的修复

#### 1️⃣ 添加超时和诊断日志
**文件**: `recipe/gkd/ray_trainer.py`

```python
def sync_rollout_weights(self):
    assert not self.hybrid_engine
    import time
    logger.info("Starting rollout weight synchronization...")
    start_time = time.time()
    
    try:
        self.actor_wg.sync_rollout_weights()
        logger.info("Actor weight sync completed")
    except Exception as e:
        logger.error(f"Actor weight sync failed: {e}")
        raise
    
    try:
        # 关键：添加超时防止无限期阻塞
        timeout = self.config.actor_rollout_ref.get("nccl_timeout", 600)
        logger.info(f"Waiting for rollout weight sync with timeout={timeout}s...")
        ray.get(self.rollout_wg.sync_rollout_weights(), timeout=timeout)
        logger.info(f"Rollout weight sync completed in {time.time() - start_time:.2f}s")
    except Exception as e:
        logger.error(f"Rollout weight sync failed after {time.time() - start_time:.2f}s: {e}")
        raise
```

#### 2️⃣ 修复 TeacherClient 初始化
**文件**: `recipe/gkd/ray_trainer.py`

添加缺失的 `num_microbatches` 参数，提高效率

#### 3️⃣ 修复 TeacherClient 队列处理
**文件**: `recipe/gkd/teacher/client.py`

改为非阻塞式处理，支持部分批次

### 立即测试

```bash
# 1. 启用诊断日志
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO

# 2. 运行训练
nohup python3 -m recipe.gkd.main_gkd \
  --config-path=recipe/gkd/config \
  --config-name=on_policy_distill_trainer \
  actor_rollout_ref.model.path=/path/to/Qwen2.5-1.5B-Instruct \
  data.train_files=/path/to/data.parquet \
  trainer.total_epochs=1 \
  trainer.n_gpus_per_node=1 rollout.n_gpus_per_node=3 \
  actor_rollout_ref.teacher.server_ip=127.0.0.1 \
  actor_rollout_ref.teacher.server_port=15555 \
  actor_rollout_ref.nccl_timeout=300 \
  trainer.scheduler=one_step_off \
  > train.log 2>&1 &

# 3. 实时监控日志
tail -f train.log | grep -i "weight sync\|error\|timeout"
```

### 预期输出

**成功**：
```
Starting rollout weight synchronization...
Actor weight sync completed
Waiting for rollout weight sync with timeout=300s...
Rollout weight sync completed in 2.34s
Generating train split: 27667 examples [00:00, 100235.65 examples/s]
...生成开始...
```

**失败（有诊断信息）**：
```
Starting rollout weight synchronization...
Rollout weight sync failed after 300.15s: timeout
# 这说明需要增加 nccl_timeout 或检查 GPU 通信问题
```

### 关键配置参数

```bash
# NCCL 超时（对于大模型很重要）
actor_rollout_ref.nccl_timeout=1200  # 20分钟，默认600s

# Rollout Worker 数量（根据 GPU 数量调整）
rollout.n_gpus_per_node=3  # 用3张卡做rollout生成

# Teacher 配置
actor_rollout_ref.teacher.n_server_workers=4
actor_rollout_ref.teacher.num_microbatches=4
```

### 如果仍然超时

1. **检查 Rollout Worker 日志**
   ```bash
   grep "rollout" train.log | grep -i "error\|failed"
   ```

2. **检查 GPU 内存**
   ```bash
   nvidia-smi  # 查看是否有卡住或内存溢出
   ```

3. **增加超时时间**
   ```bash
   actor_rollout_ref.nccl_timeout=2400  # 40分钟
   ```

4. **减少并行度**
   ```bash
   rollout.n_gpus_per_node=2  # 改用2张卡
   ```

### 文件修改完整清单

✅ `recipe/gkd/ray_trainer.py`
- 添加超时和诊断日志到 `sync_rollout_weights()`
- 添加 `num_microbatches` 到 TeacherClient 初始化

✅ `recipe/gkd/teacher/client.py`  
- 修复队列阻塞，支持部分批次

✅ `recipe/gkd/config/on_policy_distill_trainer.yaml`
- 添加 `num_microbatches: null` 配置项

---

**下一步**: 运行修复后的代码，观察 `sync_rollout_weights` 的日志输出，这样可以立即诊断问题所在。
