#!/usr/bin/env python3
"""
GKD 卡死一键诊断脚本
直接运行此脚本，自动执行完整诊断流程
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}\n")
    
    try:
        if isinstance(cmd, str):
            result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
        else:
            result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode == 0:
            print(f"\n✅ {description} 完成")
            return True
        else:
            print(f"\n❌ {description} 失败 (返回码: {result.returncode})")
            return False
    except Exception as e:
        print(f"\n❌ {description} 出错: {e}")
        return False


def check_files_exist():
    """检查必需的文件是否存在"""
    print("\n检查诊断工具...\n")
    
    required_files = [
        "diagnose_gpu_allocation.py",
        "recipe/gkd/config/on_policy_distill_trainer.yaml",
    ]
    
    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)
            print(f"❌ 缺少: {file}")
        else:
            print(f"✅ 找到: {file}")
    
    return len(missing) == 0


def main():
    """主诊断流程"""
    
    print("\n" + "="*70)
    print("  GKD 卡死一键诊断工具")
    print("  自动诊断 GKD 训练卡死问题")
    print("="*70)
    
    # 第 1 步：检查文件
    if not check_files_exist():
        print("\n❌ 缺少必需的诊断工具")
        print("请确保在 verl 根目录运行此脚本")
        sys.exit(1)
    
    # 第 2 步：GPU 诊断
    success = run_command(
        "python diagnose_gpu_allocation.py --actor-gpus 2 --rollout-gpus 2 --total-gpus 8",
        "1. GPU 分配诊断"
    )
    if not success:
        print("\n⚠️ GPU 诊断失败，但继续...")
    
    # 第 3 步：组件测试
    test_script = Path("test_gkd_components.py")
    if test_script.exists():
        success = run_command(
            "python test_gkd_components.py",
            "2. 组件可用性测试"
        )
    else:
        print("\n⚠️ 跳过组件测试（脚本不存在）")
    
    # 第 4 步：显示诊断提示
    print("\n" + "="*70)
    print("  诊断完成")
    print("="*70 + "\n")
    
    print("接下来的步骤：\n")
    
    print("1️⃣  设置调试环境变量:")
    print("   (Windows PowerShell)")
    print("   $env:NCCL_DEBUG = 'TRACE'")
    print("   $env:NCCL_TIMEOUT = '600'\n")
    
    print("2️⃣  运行训练（启用日志）:")
    print("   cd recipe/gkd")
    print("   python main_gkd.py data.output_dir=./output 2>&1 | tee train_debug.log\n")
    
    print("3️⃣  在另一个终端实时监控:")
    print("   tail -f train_debug.log | grep -E '[ACTOR]|[ROLLOUT]|[SYNC]|ERROR'\n")
    
    print("4️⃣  观察日志输出:")
    print("   ✓ 正常: [ACTOR] Model loaded → [ROLLOUT] Rollout built → [SYNC] completed")
    print("   ✗ 卡死: 某一步长时间不输出新日志\n")
    
    print("5️⃣  根据卡死位置参考诊断文档:")
    print("   📄 GKD_QUICK_FIX_CARD.md - 快速参考")
    print("   📄 GKD_REALTIME_DEBUGGING.md - 详细诊断")
    print("   📄 GKD_DEBUGGING_COMPLETE_GUIDE.md - 完整指南\n")
    
    print("常见修复方案：\n")
    print("【方案 1】增加 NCCL 超时（最常见）")
    print("   $env:NCCL_TIMEOUT = '600'\n")
    
    print("【方案 2】减少 GPU 分配（用于测试）")
    print("   python main_gkd.py trainer.n_gpus_per_node=1 rollout.n_gpus_per_node=1\n")
    
    print("【方案 3】禁用 InfiniBand（可能有兼容性问题）")
    print("   $env:NCCL_IB_DISABLE = '1'\n")
    
    print("【方案 4】清理缓存后重试")
    print("   ray shutdown")
    print("   Remove-Item -Path $env:TEMP\\nccl* -Force -Recurse")
    print("   ray start --head\n")
    
    print("="*70)
    print("  👉 立即开始：参考上面的 5 个步骤运行训练")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
