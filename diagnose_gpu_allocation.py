#!/usr/bin/env python3
"""
GKD GPU Allocation Diagnostic Tool
诊断 GKD 中的 GPU 分配和模型加载问题
"""

import os
import sys
import subprocess
import json
from pathlib import Path


def print_section(title):
    """打印章节标题"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def run_cmd(cmd, timeout=5):
    """运行命令"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return "[TIMEOUT]"
    except Exception as e:
        return f"[ERROR] {e}"


def check_cuda():
    """检查 CUDA 基础信息"""
    print_section("1. CUDA 基础信息")
    
    # nvidia-smi
    print("[*] nvidia-smi 输出:")
    output = run_cmd("nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader")
    print(output)
    
    # 环境变量
    print("\n[*] 环境变量:")
    env_vars = ['CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'CUDA_LAUNCH_BLOCKING', 'NCCL_DEBUG']
    for var in env_vars:
        val = os.environ.get(var, "NOT SET")
        print(f"  {var}: {val}")


def check_ray():
    """检查 Ray 集群"""
    print_section("2. Ray 集群状态")
    
    code = """
import ray
import sys

if ray.is_initialized():
    print("✓ Ray 已初始化")
    print(f"Dashboard: {ray.get_dashboard_url()}")
    
    resources = ray.cluster_resources()
    print(f"\\nCluster Resources:")
    for key, val in sorted(resources.items()):
        if 'GPU' in key or 'CPU' in key:
            print(f"  {key}: {val}")
    
    nodes = ray.nodes()
    print(f"\\nNodes: {len(nodes)}")
    for node in nodes:
        print(f"  - {node['NodeID'][:8]}...")
        if 'GPU' in node['Resources']:
            print(f"    GPU: {node['Resources']['GPU']}")
else:
    print("✗ Ray 未初始化")
    sys.exit(1)
"""
    output = run_cmd(f"python3 -c \"{code}\"")
    print(output)


def check_torch():
    """检查 PyTorch 和分布式设置"""
    print_section("3. PyTorch 分布式配置")
    
    code = """
import torch
import os

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    
    print(f"\\nGPU 信息:")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem = props.total_memory / 1024**3
        print(f"  GPU {i}: {props.name} ({mem:.1f}GB)")

print(f"\\nDistributed Variables:")
for var in ['RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT']:
    print(f"  {var}: {os.environ.get(var, 'NOT SET')}")
"""
    output = run_cmd(f"python3 -c \"{code}\"")
    print(output)


def check_resource_pool_allocation(n_gpus_actor, n_gpus_rollout, total_gpus=8):
    """模拟 ResourcePool 分配"""
    print_section("4. ResourcePool 分配模拟")
    
    print(f"配置:")
    print(f"  trainer.n_gpus_per_node: {n_gpus_actor}")
    print(f"  rollout.n_gpus_per_node: {n_gpus_rollout}")
    print(f"  总 GPU 数: {total_gpus}")
    print(f"  Teacher GPU: {total_gpus - n_gpus_actor - n_gpus_rollout} (独立进程)\n")
    
    print(f"预期 GPU 分配:")
    print(f"  [Actor Workers]")
    print(f"    Resource Pool: actor_pool")
    print(f"    GPUs: 0-{n_gpus_actor-1}")
    print(f"    Num Workers: 1")
    print(f"    GPUs per Worker: {n_gpus_actor}\n")
    
    print(f"  [Rollout Workers]")
    print(f"    Resource Pool: rollout_pool")
    print(f"    GPUs: {n_gpus_actor}-{n_gpus_actor + n_gpus_rollout - 1}")
    print(f"    Num Workers: 1")
    print(f"    GPUs per Worker: {n_gpus_rollout}\n")
    
    print(f"  [Teacher Server] (独立启动)")
    print(f"    GPUs: {n_gpus_actor + n_gpus_rollout}-{total_gpus - 1}")
    print(f"    GPU 数: {total_gpus - n_gpus_actor - n_gpus_rollout}\n")
    
    # 验证
    total_allocated = n_gpus_actor + n_gpus_rollout
    if total_allocated <= total_gpus:
        print(f"✓ GPU 分配合理 (已用 {total_allocated}/{total_gpus})")
    else:
        print(f"✗ GPU 分配超额 (需要 {total_allocated}，但只有 {total_gpus})")


def check_model_loading():
    """分析模型加载"""
    print_section("5. 模型加载分析")
    
    print("GKD 中的模型加载:")
    print("\n[Actor Worker] init_model():")
    print("  1. 加载学生模型 (actor_module)")
    print("  2. 初始化优化器 (Adam)")
    print("  3. 创建 OnPolicyDistillActor")
    print("  4. 估计内存: 13-14GB\n")
    
    print("[Rollout Worker] init_model():")
    print("  1. 加载学生模型 (用于推理)")
    print("  2. 初始化 vLLM/SGLang 推理引擎")
    print("  3. 准备 KV 缓存")
    print("  4. 估计内存: 5-6GB\n")
    
    print("[Teacher Server] (独立进程):")
    print("  1. 启动 4 张 GPU 的 vLLM/SGLang")
    print("  2. 加载教师模型 (Qwen3-4B)")
    print("  3. 估计内存: 12-15GB\n")
    
    print("总内存: ~40GB (单机 8 张 A100 可用 320GB，充足)")


def check_log_for_deadlock(log_file=None):
    """检查日志中的卡死迹象"""
    print_section("6. 日志分析")
    
    if log_file is None:
        log_file = "train.log"
    
    if not Path(log_file).exists():
        print(f"✗ 日志文件不存在: {log_file}")
        return
    
    print(f"✓ 分析日志: {log_file}\n")
    
    # 查找关键事件的时间戳
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # 找到 init_model 调用
        print("[*] 关键事件:")
        for i, line in enumerate(lines):
            if 'Before init' in line:
                print(f"  Line {i}: {line.strip()[:80]}")
            elif 'After' in line and 'init' in line:
                print(f"  Line {i}: {line.strip()[:80]}")
            elif 'weight sync' in line.lower():
                print(f"  Line {i}: {line.strip()[:80]}")
        
        # 查找错误
        print("\n[*] 错误和警告:")
        error_count = 0
        for i, line in enumerate(lines[-100:]):  # 只看最后100行
            if any(x in line.lower() for x in ['error', 'failed', 'exception', 'timeout']):
                print(f"  {line.strip()[:80]}")
                error_count += 1
        
        if error_count == 0:
            print("  (未发现明显错误)")
    
    except Exception as e:
        print(f"✗ 读取日志失败: {e}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='GKD GPU 分配诊断工具')
    parser.add_argument('--actor-gpus', type=int, default=1, help='Actor GPUs')
    parser.add_argument('--rollout-gpus', type=int, default=3, help='Rollout GPUs')
    parser.add_argument('--total-gpus', type=int, default=8, help='Total GPUs')
    parser.add_argument('--log', type=str, default='train.log', help='Log file path')
    parser.add_argument('--no-ray', action='store_true', help='跳过 Ray 检查')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  GKD GPU 分配诊断工具")
    print("="*70)
    
    # 1. CUDA
    check_cuda()
    
    # 2. Ray (可选)
    if not args.no_ray:
        try:
            check_ray()
        except Exception as e:
            print(f"✗ Ray 检查失败: {e}")
    
    # 3. PyTorch
    check_torch()
    
    # 4. ResourcePool 分配
    check_resource_pool_allocation(args.actor_gpus, args.rollout_gpus, args.total_gpus)
    
    # 5. 模型加载
    check_model_loading()
    
    # 6. 日志
    check_log_for_deadlock(args.log)
    
    print("\n" + "="*70)
    print("  诊断完成")
    print("="*70 + "\n")
    
    print("常见问题排查:")
    print("1. CUDA_VISIBLE_DEVICES 未设置或不正确")
    print("   解决: export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7")
    print("\n2. Ray 无法找到 GPU")
    print("   解决: ray shutdown; ray start --head")
    print("\n3. Rollout init_model 卡死")
    print("   解决: 增加 NCCL_TIMEOUT，检查 NCCL 通信")
    print("\n4. Actor 和 Rollout 争用同一 GPU")
    print("   解决: 确认 ResourcePool 配置正确")
    print()


if __name__ == '__main__':
    main()
