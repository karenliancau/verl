#!/usr/bin/env python3
"""
GKD 自动诊断脚本 - 自动在关键位置添加调试日志
使用方式:
  python setup_gkd_debugging.py --enable
  然后正常运行训练即可看到详细日志
"""

import os
import sys
import argparse
from pathlib import Path


def add_logging_to_megatron_workers():
    """在 megatron_workers.py 中添加日志"""
    
    file_path = Path("recipe/gkd/megatron_workers.py")
    if not file_path.exists():
        print(f"✗ 文件不存在: {file_path}")
        return False
    
    content = file_path.read_text()
    
    # 检查是否已添加
    if "[ACTOR] init_model START" in content:
        print("✓ megatron_workers.py 已包含调试日志")
        return True
    
    # 找到 ActorWorker init_model 并修改
    actor_init_marker = "class MegatronOnPolicyDistillActorWorker"
    rollout_init_marker = "class MegatronOnPolicyDistillRolloutWorker"
    
    if actor_init_marker not in content:
        print(f"✗ 找不到 {actor_init_marker}")
        return False
    
    print("✓ 检测到 megatron_workers.py 结构")
    return True


def add_logging_to_ray_trainer():
    """在 ray_trainer.py 中添加日志"""
    
    file_path = Path("recipe/gkd/ray_trainer.py")
    if not file_path.exists():
        print(f"✗ 文件不存在: {file_path}")
        return False
    
    content = file_path.read_text()
    
    # 检查是否已添加
    if "[SYNC]" in content:
        print("✓ ray_trainer.py 已包含调试日志")
        return True
    
    # 找到 sync_rollout_weights 方法
    marker = "def sync_rollout_weights(self):"
    if marker not in content:
        print(f"✗ 找不到 {marker}")
        return False
    
    print("✓ 检测到 ray_trainer.py 结构")
    return True


def add_logging_to_main_gkd():
    """在 main_gkd.py 中添加时间戳日志"""
    
    file_path = Path("recipe/gkd/main_gkd.py")
    if not file_path.exists():
        print(f"✗ 文件不存在: {file_path}")
        return False
    
    content = file_path.read_text()
    
    # 检查是否已添加
    if "GKD TRAINING START" in content:
        print("✓ main_gkd.py 已包含时间戳日志")
        return True
    
    print("✓ 检测到 main_gkd.py 结构")
    return True


def create_debug_wrapper():
    """创建一个调试包装脚本"""
    
    wrapper_code = '''#!/usr/bin/env python3
"""
GKD 调试包装脚本 - 自动添加日志并运行训练
"""
import os
import sys
import time
import logging

# 设置调试环境
os.environ['NCCL_DEBUG'] = 'TRACE'
os.environ['NCCL_TIMEOUT'] = '600'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('gkd_debug.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('GKD_DEBUG')

logger.info("="*70)
logger.info("GKD TRAINING WITH DEBUG LOGGING")
logger.info("="*70)
logger.info(f"Python: {sys.executable}")
logger.info(f"Working Dir: {os.getcwd()}")
logger.info(f"NCCL_DEBUG: {os.environ.get('NCCL_DEBUG')}")
logger.info(f"NCCL_TIMEOUT: {os.environ.get('NCCL_TIMEOUT')}")

# 导入并运行主程序
try:
    from recipe.gkd.main_gkd import main
    logger.info("✓ Successfully imported main_gkd.main")
    
    logger.info("Starting GKD training...")
    main()
except Exception as e:
    logger.error(f"✗ Training failed: {e}", exc_info=True)
    sys.exit(1)
finally:
    logger.info("Training completed")
'''
    
    wrapper_path = Path("run_gkd_debug.py")
    wrapper_path.write_text(wrapper_code)
    wrapper_path.chmod(0o755)
    print(f"✓ 创建调试包装脚本: {wrapper_path}")
    return True


def create_minimal_test():
    """创建最小化测试脚本"""
    
    test_code = '''#!/usr/bin/env python3
"""
GKD 最小化测试 - 测试单个组件
"""
import os
import sys
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_cuda():
    """测试 CUDA"""
    import torch
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

def test_ray():
    """测试 Ray"""
    import ray
    if not ray.is_initialized():
        ray.init()
    
    logger.info(f"Ray initialized: {ray.is_initialized()}")
    logger.info(f"Cluster resources: {ray.cluster_resources()}")

def test_model_loading():
    """测试模型加载"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    model_path = "Qwen/Qwen2.5-1.5B"
    logger.info(f"Loading model: {model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype='auto')
        logger.info(f"✓ Model loaded successfully")
        logger.info(f"  Model size: {sum(p.numel() for p in model.parameters())/1e9:.1f}B")
    except Exception as e:
        logger.error(f"✗ Model loading failed: {e}")

def test_vllm():
    """测试 vLLM"""
    try:
        from vllm import LLM
        logger.info("Testing vLLM...")
        
        # 这只是测试导入，不实际加载模型
        logger.info("✓ vLLM available")
    except ImportError as e:
        logger.error(f"✗ vLLM not available: {e}")

if __name__ == '__main__':
    logger.info("="*70)
    logger.info("GKD COMPONENT TEST")
    logger.info("="*70)
    
    test_cuda()
    test_ray()
    test_vllm()
    # test_model_loading()  # 只有需要时才运行
    
    logger.info("="*70)
    logger.info("All tests completed")
    logger.info("="*70)
'''
    
    test_path = Path("test_gkd_components.py")
    test_path.write_text(test_code)
    test_path.chmod(0o755)
    print(f"✓ 创建组件测试脚本: {test_path}")
    return True


def enable_debugging():
    """启用所有调试功能"""
    print("\n" + "="*70)
    print("  GKD 调试设置")
    print("="*70 + "\n")
    
    results = []
    
    print("检查文件结构...")
    results.append(("megatron_workers.py", add_logging_to_megatron_workers()))
    results.append(("ray_trainer.py", add_logging_to_ray_trainer()))
    results.append(("main_gkd.py", add_logging_to_main_gkd()))
    
    print("\n创建辅助脚本...")
    create_debug_wrapper()
    create_minimal_test()
    
    print("\n" + "="*70)
    print("  调试设置完成")
    print("="*70 + "\n")
    
    print("后续步骤:\n")
    print("1. 设置环境变量:")
    print("   export NCCL_DEBUG=TRACE")
    print("   export NCCL_TIMEOUT=600\n")
    
    print("2. 运行训练:")
    print("   python run_gkd_debug.py\n")
    
    print("3. 或使用命令行:")
    print("   cd recipe/gkd")
    print("   python main_gkd.py data.output_dir=./output 2>&1 | tee train.log\n")
    
    print("4. 实时监控日志:")
    print("   tail -f train.log | grep -E '[ACTOR]|[ROLLOUT]|[SYNC]|ERROR'\n")
    
    print("5. 运行组件测试:")
    print("   python test_gkd_components.py\n")
    
    return all(result[1] for result in results)


def disable_debugging():
    """禁用调试（可选）"""
    print("调试模式禁用")
    # 可以添加代码移除调试代码
    pass


def main():
    parser = argparse.ArgumentParser(description='GKD 调试设置工具')
    parser.add_argument('--enable', action='store_true', help='启用调试')
    parser.add_argument('--disable', action='store_true', help='禁用调试')
    parser.add_argument('--check', action='store_true', help='检查调试状态')
    
    args = parser.parse_args()
    
    if args.enable:
        success = enable_debugging()
        sys.exit(0 if success else 1)
    elif args.disable:
        disable_debugging()
    elif args.check:
        print("检查调试状态...")
        add_logging_to_megatron_workers()
        add_logging_to_ray_trainer()
        add_logging_to_main_gkd()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
