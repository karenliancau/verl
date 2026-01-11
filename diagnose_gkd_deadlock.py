#!/usr/bin/env python3
"""
GKD Deadlock Diagnosis Tool
快速诊断 GKD 训练卡死问题
"""

import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path


def print_section(title):
    """打印章节标题"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def run_command(cmd, description=""):
    """运行命令并捕获输出"""
    if description:
        print(f"[*] {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return "[TIMEOUT] Command took too long to execute"
    except Exception as e:
        return f"[ERROR] {str(e)}"


def check_nvidia_gpu():
    """检查 NVIDIA GPU 状态"""
    print_section("1. NVIDIA GPU Status")
    
    # 检查 nvidia-smi 是否可用
    output = run_command("nvidia-smi", "Running nvidia-smi")
    print(output)
    
    # 检查 GPU 内存
    output = run_command(
        "nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader",
        "Checking GPU memory"
    )
    print("GPU Memory Usage:")
    print(output)
    
    # 检查进程占用的 GPU
    output = run_command(
        "nvidia-smi pmon -c 1",
        "Checking GPU processes"
    )
    print("Active GPU Processes:")
    print(output)


def check_ray_cluster():
    """检查 Ray 集群状态"""
    print_section("2. Ray Cluster Status")
    
    code = """
import ray
import json

try:
    if not ray.is_initialized():
        print("Ray is NOT initialized")
        sys.exit(0)
    
    print("Ray is initialized")
    print(f"Dashboard URL: {ray.get_dashboard_url()}")
    
    # 检查集群资源
    resources = ray.cluster_resources()
    print(f"\\nCluster Resources:")
    for key, value in resources.items():
        print(f"  {key}: {value}")
    
    # 检查节点信息
    print(f"\\nCluster Nodes:")
    nodes = ray.nodes()
    for i, node in enumerate(nodes):
        print(f"  Node {i}:")
        print(f"    NodeID: {node['NodeID']}")
        print(f"    RayletIP: {node['RayletIP']}")
        print(f"    Resources: {node['Resources']}")
    
    # 检查 Actor 信息
    print(f"\\nActive Actors:")
    try:
        actors = ray.util.list_named_actors()
        for actor in actors:
            print(f"  {actor['name']}: {actor['state']}")
    except Exception as e:
        print(f"  Cannot list actors: {e}")
        
except Exception as e:
    print(f"Error checking Ray: {e}")
    import traceback
    traceback.print_exc()
"""
    
    output = run_command(
        f"python3 -c \"{code}\"",
        "Checking Ray cluster"
    )
    print(output)


def check_torch_distributed():
    """检查 PyTorch 分布式初始化"""
    print_section("3. PyTorch Distributed Status")
    
    code = """
import torch
import os

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"GPU Names:")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f}GB")

print(f"\\nDistributed Environment Variables:")
dist_vars = [
    'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT',
    'NCCL_DEBUG', 'TORCH_DISTRIBUTED_DEBUG', 'NCCL_TIMEOUT'
]
for var in dist_vars:
    value = os.environ.get(var, "NOT SET")
    print(f"  {var}: {value}")

print(f"\\nNCCL Environment Variables:")
nccl_vars = [k for k in os.environ.keys() if k.startswith('NCCL_')]
for var in nccl_vars:
    print(f"  {var}: {os.environ[var]}")

print(f"\\nDistributed Initialized: {torch.distributed.is_initialized()}")
"""
    
    output = run_command(
        f"python3 -c \"{code}\"",
        "Checking PyTorch distributed"
    )
    print(output)


def check_verl_imports():
    """检查 VERL 模块导入"""
    print_section("4. VERL Module Imports")
    
    code = """
import sys

modules_to_check = [
    'verl',
    'verl.workers',
    'verl.single_controller',
    'recipe.gkd',
    'recipe.gkd.ray_trainer',
    'recipe.gkd.megatron_workers',
    'recipe.gkd.teacher',
]

for module in modules_to_check:
    try:
        __import__(module)
        print(f"✓ {module}")
    except ImportError as e:
        print(f"✗ {module}: {e}")
    except Exception as e:
        print(f"✗ {module}: {type(e).__name__}: {e}")
"""
    
    output = run_command(
        f"python3 -c \"{code}\"",
        "Checking VERL imports"
    )
    print(output)


def check_config():
    """检查配置文件"""
    print_section("5. Configuration Files")
    
    config_file = "recipe/gkd/config/on_policy_distill_trainer.yaml"
    if Path(config_file).exists():
        print(f"✓ Found {config_file}")
        output = run_command(
            f"grep -A 10 'teacher:' {config_file}",
            "Reading teacher configuration"
        )
        print("Teacher Config:")
        print(output)
    else:
        print(f"✗ Config file not found: {config_file}")


def analyze_log(log_file):
    """分析训练日志"""
    print_section(f"6. Log Analysis: {log_file}")
    
    if not Path(log_file).exists():
        print(f"✗ Log file not found: {log_file}")
        return
    
    print("✓ Found log file")
    
    # 查找关键错误
    print("\n[*] Searching for errors...")
    output = run_command(
        f"grep -i 'error\\|failed\\|timeout\\|exception' {log_file} | head -20",
        ""
    )
    if output.strip():
        print("Errors found:")
        print(output)
    else:
        print("No obvious errors found")
    
    # 查找权重同步日志
    print("\n[*] Searching for weight sync logs...")
    output = run_command(
        f"grep 'weight sync\\|collective' {log_file} | tail -20",
        ""
    )
    if output.strip():
        print("Weight sync logs:")
        print(output)
    else:
        print("No weight sync logs found")
    
    # 查找最后的日志
    print("\n[*] Last 30 lines of log:")
    output = run_command(f"tail -30 {log_file}", "")
    print(output)


def generate_recommendations():
    """生成建议"""
    print_section("7. Recommendations")
    
    print("""
如果遇到卡死问题，请按以下步骤排查：

1. 🔍 检查日志中的 "Rollout weight sync" 部分
   - 如果看到 "timeout" → 增加 actor_rollout_ref.nccl_timeout
   - 如果看到 "error" → 检查 GPU 和网络连接

2. 💾 检查 GPU 内存
   - 运行 nvidia-smi，查看是否有 GPU 被卡住或内存溢出
   - 考虑减少 rollout.n_gpus_per_node

3. 🔧 调整配置
   - 增加 nccl_timeout: actor_rollout_ref.nccl_timeout=1200
   - 减少并行: rollout.n_gpus_per_node=2
   - 启用调试: export NCCL_DEBUG=INFO

4. 📊 启用完整诊断
   export NCCL_DEBUG=TRACE
   export TORCH_DISTRIBUTED_DEBUG=INFO
   export VLLM_LOGGING_LEVEL=DEBUG

5. 🚀 重新运行训练
   nohup python3 -m recipe.gkd.main_gkd \\
     --config-path=recipe/gkd/config \\
     --config-name=on_policy_distill_trainer \\
     actor_rollout_ref.model.path=/path/to/model \\
     ... \\
     actor_rollout_ref.nccl_timeout=600 \\
     > train.log 2>&1 &
""")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("  GKD Deadlock Diagnosis Tool")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*60)
    
    # 1. GPU 检查
    check_nvidia_gpu()
    
    # 2. Ray 集群检查
    check_ray_cluster()
    
    # 3. PyTorch 分布式检查
    check_torch_distributed()
    
    # 4. VERL 模块检查
    check_verl_imports()
    
    # 5. 配置文件检查
    check_config()
    
    # 6. 日志分析
    log_file = "train.log"
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    
    if Path(log_file).exists():
        analyze_log(log_file)
    else:
        print_section(f"6. Log Analysis")
        print(f"ℹ️  Log file not found: {log_file}")
        print(f"   Usage: python3 diagnose.py <log_file>")
    
    # 7. 建议
    generate_recommendations()
    
    print("\n" + "="*60)
    print("  Diagnosis Complete")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
