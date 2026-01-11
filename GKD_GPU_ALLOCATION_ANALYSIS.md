## GKD GPU分配 - 深度分析

### 🎯 问题澄清

你的怀疑**完全正确**！GPU分配确实有问题，关键是要理解GKD中 **Actor 和 Rollout** 的加载方式。

---

## 架构分析

### 1. Resource Pool 分配机制

**配置**: `recipe/gkd/config/on_policy_distill_trainer.yaml`
```yaml
trainer:
  n_gpus_per_node: 2   # Actor 用 2 张卡
  nnodes: 1

rollout:
  n_gpus_per_node: 2   # Rollout 用 2 张卡
  nnodes: 1
```

**GKD 中的 ResourcePoolManager** (`main_gkd.py` 第182-189行):
```python
actor_pool = [config.trainer.n_gpus_per_node] * config.trainer.nnodes  # [2]
rollout_pool = [config.rollout.n_gpus_per_node] * config.rollout.nnodes  # [2]

resource_pool_spec = {
    "rollout_pool": rollout_pool,  # [2] → GPU 0-1
    "actor_pool": actor_pool,       # [2] → GPU 2-3  
}
```

**GPU 分配结果** (8卡机器):
- GPU 0-1: Rollout Worker (生成序列)
- GPU 2-3: Actor Worker (训练模型)
- GPU 4-7: Teacher Server (独立启动，单独进程)

---

## 关键问题：模型加载逻辑

### 问题 1：Actor Worker 加载什么模型？

**文件**: `recipe/gkd/megatron_workers.py` 第480-537行

```python
class MegatronOnPolicyDistillActorWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # 加载 ACTOR 模型（用于训练）
        self.actor_module, self.actor_optimizer, ... = self._build_model_optimizer(
            model_path=self.config.model.path,  # ← 学生模型路径
            optim_config=self.config.actor.optim,
            ...
        )
        
        # 创建 Actor 对象
        self.actor = OnPolicyDistillActor(
            actor_module=self.actor_module,  # 学生模型
            ...
        )
```

**结论**: Actor 加载 **学生模型 (Qwen2.5-1.5B)**

### 问题 2：Rollout Worker 加载什么模型？

**文件**: `recipe/gkd/megatron_workers.py` 第697-707行

```python
class MegatronOnPolicyDistillRolloutWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # 加载 ROLLOUT 模型（用于推理生成）
        self._build_rollout(trust_remote_code=...)
```

关键：`_build_rollout()` 加载的是什么？

**父类实现** (`verl/workers/megatron_workers.py` 第489-550行):

```python
def _build_rollout(self, trust_remote_code=False):
    # 使用同一个模型路径
    model_config: HFModelConfig = omega_conf_to_dataclass(
        OmegaConf.create(model_config_dict), 
        dataclass_type=HFModelConfig
    )
    
    # 创建 vLLM/SGLang 推理引擎
    # 该引擎也加载 self.config.model.path（学生模型）
    self.rollout = vLLMRollout(
        model_hf_config=self.actor_model_config,  # 同学生模型配置
        ...
    )
```

**结论**: Rollout 也加载 **学生模型 (Qwen2.5-1.5B)** 用于生成序列

---

## 内存分析

### 学生模型内存占用

Qwen2.5-1.5B (1.5B 参数，bfloat16):
- 模型参数: 1.5B × 2字节 ≈ **3GB**
- 梯度 (Actor): 3GB
- 优化器状态 (Adam): 3GB × 2 ≈ **6GB**
- KV缓存 + 激活值: 1-2GB
- **Actor Worker 总计**: ~12-13GB

Rollout (推理专用):
- 模型参数: **3GB**
- KV缓存 (推理): 2-3GB
- **Rollout Worker 总计**: ~5-6GB

**总使用**: (13 + 6) × N_workers = 19GB × 1 = 19GB (合理)

---

## 关键发现：为什么配置文件中说 teacher 需要 n_gpus_per_node?

### 答案: 这是**误导**

在 `on_policy_distill_trainer.yaml` 第280-284行:
```yaml
teacher:
  server_ip: localhost
  server_port: 15555
  overlap_rollout: False
  n_server_workers: 1
```

这里的 **`n_server_workers`** 不是 GPU 数量，而是：
- **后台线程数** - Teacher 服务器内部的处理线程数
- 不影响 GPU 分配（Teacher 在独立进程中）
- 取值范围通常: 1-4

---

## 实际问题诊断

### 问题 A：Ray 分配 GPU 不正确

**症状**: Actor 或 Rollout 初始化时卡死

**可能原因**:
1. Ray 没有正确分配 GPU 到进程
2. 多个 Worker 争用同一张 GPU
3. NCCL 通信在错误的 GPU 上进行

**检查方法**:
```python
# 在 Worker 中添加诊断代码
import os
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'NOT SET')}")

import torch
print(f"Available GPUs: {torch.cuda.device_count()}")
print(f"Current GPU: {torch.cuda.current_device()}")
```

### 问题 B：同时加载学生模型太慢

**症状**: init_model 阶段卡死 20+ 秒

**原因**: 虽然 Actor 和 Rollout 分别在不同 GPU 上，但：
1. 都要加载同一个模型（下载/解析/转换）
2. Megatron 初始化很慢（NCCL 通信）
3 没有并行化（顺序初始化）

**优化方式**:
```python
# 可以尝试异步初始化
# 或在 init_workers 中添加并发控制
```

---

## GPU 配置建议

### 对于单机 8 卡 + Teacher 的场景

**Option 1: 当前配置（推荐）**
```yaml
trainer:
  n_gpus_per_node: 1    # Actor 用 1 张卡（训练不需要太多）
  nnodes: 1

rollout:
  n_gpus_per_node: 3    # Rollout 用 3 张卡（推理需要并行）
  nnodes: 1

# Teacher: 另外启动，用 4 张卡
```

GPU 分布:
- GPU 0: Actor (训练学生模型)
- GPU 1-3: Rollout (并行生成序列)
- GPU 4-7: Teacher Server (独立进程)

**优势**: 充分利用 8 张卡

---

## 验证 GPU 分配

运行这个脚本确认实际分配:

```python
import ray
import torch

@ray.remote(num_gpus=1)
def check_gpu():
    import os
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    return os.environ.get('CUDA_VISIBLE_DEVICES')

ray.init()
futures = [check_gpu.remote() for _ in range(4)]
results = ray.get(futures)
print("GPU Assignments:", results)
ray.shutdown()
```

**期望输出**:
```
GPU Assignments: ['0', '1', '2', '3']
```

如果出现重复（如 `['0', '0', '1', '1']`），说明分配有问题。

---

## 总结

1. ✅ **Teacher 配置中的 `n_gpus_per_node` 无效** - Teacher 是独立进程，GPU 分配由启动命令控制
2. ✅ **Actor 和 Rollout 各自加载完整的学生模型** - 这是设计，用于权重同步
3. ✅ **实际 GPU 分配由 ResourcePoolManager 处理** - 根据 `trainer.n_gpus_per_node` 和 `rollout.n_gpus_per_node` 分配
4. ❓ **卡死原因**: 需要检查 CUDA_VISIBLE_DEVICES 是否正确设置，以及 NCCL 通信是否正常

---

## 快速诊断

在 `recipe/gkd/megatron_workers.py` 的 `init_model()` 中添加:

```python
@register(dispatch_mode=Dispatch.ONE_TO_ALL)
def init_model(self):
    import os
    rank = int(os.environ.get('LOCAL_RANK', 0))
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')
    print(f"Worker {rank}: CUDA_VISIBLE_DEVICES={cuda_devices}, GPU Count={torch.cuda.device_count()}")
    
    # 继续原有逻辑...
```

这样可以立即看到 GPU 分配是否正确。
