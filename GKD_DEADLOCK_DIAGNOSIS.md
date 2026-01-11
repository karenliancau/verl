# GKD训练卡死问题诊断与修复

## 问题描述

在运行GKD（Generative Knowledge Distillation）训练时，数据集加载完成后训练立即卡死。日志显示：
```
(TaskRunner pid=133947) Generating train split: 27667 examples [00:00, 100235.65 examples/s]
Generating train split: 27667 examples [00:00, 98703.23 examples/s]
```

然后程序无响应。**重要：Teacher服务未收到任何请求，说明卡死发生在rollout阶段，而非teacher阶段。**

## 真正的根本原因

问题出在**Rollout权重同步的阻塞调用**上：

### 问题1：sync_rollout_weights中的阻塞ray.get()调用
- **位置**: `recipe/gkd/ray_trainer.py` 第329行
- **代码**:
  ```python
  def sync_rollout_weights(self):
      assert not self.hybrid_engine
      self.actor_wg.sync_rollout_weights()
      ray.get(self.rollout_wg.sync_rollout_weights())  # ❌ 同步阻塞调用
  ```
- **问题**: `ray.get()` 是同步阻塞调用，会无限期等待所有rollout workers完成权重同步
- **死锁场景**:
  1. 训练循环调用 `_async_gen_next_batch()`
  2. 其中执行 `self.sync_rollout_weights()`  
  3. `ray.get()` 阻塞等待rollout workers完成
  4. 如果rollout workers出现以下任何问题就会永久阻塞：
     - 初始化失败
     - 权重同步出错
     - NCCL通信卡住（集群通信问题）
     - GPU内存不足

### 问题2：async_generate_sequences标注与实现不一致
- **位置**: `recipe/gkd/megatron_workers.py` 第749行
- **代码**:
  ```python
  @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"), blocking=False)
  def async_generate_sequences(self, *args, **kwargs):
      return self.generate_sequences(*args, **kwargs)  # 实际是同步调用
  ```
- **问题**: 虽然标注了 `blocking=False`，但实际执行的 `generate_sequences` 是同步阻塞的

### 问题3：TeacherClient初始化参数缺失（次要问题）
虽然这不是主卡死原因，但也会导致低效：
- TeacherClient初始化时缺少 `num_microbatches` 参数，导致每次只处理1个微批次
- 当批量大小很大时会导致队列阻塞

## 解决方案

### 修改1：添加超时和诊断日志到sync_rollout_weights
**文件**: `recipe/gkd/ray_trainer.py` 第326行

添加超时和日志，防止无限期阻塞：
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
        # Add timeout to prevent indefinite blocking
        timeout = self.config.actor_rollout_ref.get("nccl_timeout", 600)
        logger.info(f"Waiting for rollout weight sync with timeout={timeout}s...")
        ray.get(self.rollout_wg.sync_rollout_weights(), timeout=timeout)
        logger.info(f"Rollout weight sync completed in {time.time() - start_time:.2f}s")
    except Exception as e:
        logger.error(f"Rollout weight sync failed after {time.time() - start_time:.2f}s: {e}")
        raise
```

### 修改2：修复TeacherClient初始化（已完成）
**文件**: `recipe/gkd/ray_trainer.py` 第170行附近

添加 `num_microbatches` 参数，提高teacher知识蒸馏效率

### 修改3：修复TeacherClient队列处理（已完成）
**文件**: `recipe/gkd/teacher/client.py` 第88行附近

改为非阻塞式处理，支持部分批次

## 快速诊断

### 第一步：启用调试日志
在启动命令前设置：
```bash
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO
export VLLM_LOGGING_LEVEL=INFO
```

### 第二步：运行训练并观察日志
```bash
nohup python3 -m recipe.gkd.main_gkd \
  --config-path=/home/ma-user/work/nlp/***/verl/recipe/gkd/config \
  --config-name=on_policy_distill_trainer \
  actor_rollout_ref.model.path=/path/to/model \
  ... \
  actor_rollout_ref.nccl_timeout=300 \
  > train.log 2>&1 &
```

### 第三步：检查关键日志
```bash
# 查看权重同步日志
tail -f train.log | grep -i "weight sync\|rollout weight\|timeout\|error"

# 查看完整日志
tail -100 train.log
```

### 预期日志输出

**正常情况**：
```
Starting rollout weight synchronization...
Actor weight sync completed
Waiting for rollout weight sync with timeout=600s...
Rollout weight sync completed in 2.34s
```

**异常情况1 - Actor同步失败**：
```
Starting rollout weight synchronization...
Actor weight sync failed: NCCL operation timed out
```

**异常情况2 - Rollout同步超时**：
```
Starting rollout weight synchronization...
Actor weight sync completed
Waiting for rollout weight sync with timeout=600s...
Rollout weight sync failed after 600.15s: timeout
```

**异常情况3 - Rollout worker未初始化**：
```
Starting rollout weight synchronization...
Actor weight sync completed
Waiting for rollout weight sync with timeout=600s...
Rollout weight sync failed: RayActorError: The actor died unexpectedly before finishing this task.
```

## 进阶诊断

### 检查Rollout Worker初始化
```bash
grep -i "rollout.*init\|initialized\|collective" train.log | head -20
```

### 检查GPU资源
```bash
# 在另一个终端运行
watch -n 1 nvidia-smi

# 查找内存泄漏或长期占用
ps aux | grep python | grep -v grep
```

### 检查Ray集群状态
```bash
python3 -c "
import ray
if ray.is_initialized():
    print('Ray resources:', ray.cluster_resources())
    print('Ray nodes:')
    for node in ray.nodes():
        print('  ', node)
else:
    print('Ray not initialized')
"
```

## 配置调优建议

### 对于单机8卡配置（你的场景）

**推荐配置**:
```bash
nohup python3 -m recipe.gkd.main_gkd \
  --config-path=recipe/gkd/config \
  --config-name=on_policy_distill_trainer \
  actor_rollout_ref.model.path=/path/to/model \
  data.train_files=/path/to/data \
  trainer.total_epochs=1 \
  trainer.n_gpus_per_node=1 \
  rollout.n_gpus_per_node=3 \
  actor_rollout_ref.teacher.server_ip=127.0.0.1 \
  actor_rollout_ref.teacher.server_port=15555 \
  actor_rollout_ref.teacher.n_server_workers=4 \
  actor_rollout_ref.teacher.num_microbatches=4 \
  actor_rollout_ref.nccl_timeout=1200 \
  trainer.scheduler=one_step_off \
  > train.log 2>&1 &
```

**参数说明**:
- `trainer.n_gpus_per_node=1` - 1张卡用于actor训练
- `rollout.n_gpus_per_node=3` - 3张卡用于rollout生成
- `actor_rollout_ref.nccl_timeout=1200` - 增加NCCL超时到20分钟（适合大模型）
- `actor_rollout_ref.teacher.num_microbatches=4` - 让teacher并行处理4个微批次

## 后续步骤

1. **应用修复**: 最新的 `recipe/gkd/ray_trainer.py` 已包含超时和日志增强
2. **测试**: 运行修复后的代码，观察是否正常同步权重
3. **收集日志**: 如果仍然超时，收集完整日志用于进一步分析
4. **调整参数**: 根据诊断日志调整 `nccl_timeout` 和 `n_gpus_per_node`

## 相关文件变更

- ✅ `recipe/gkd/ray_trainer.py` 
  - 添加超时和诊断日志到 `sync_rollout_weights()`
  - 添加 `num_microbatches` 参数到 TeacherClient 初始化
  
- ✅ `recipe/gkd/teacher/client.py`
  - 修复队列阻塞问题，支持部分批次处理
  
- ✅ `recipe/gkd/config/on_policy_distill_trainer.yaml`
  - 添加 `num_microbatches` 配置选项
