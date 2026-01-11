
import sys
import os
import inspect
from omegaconf import OmegaConf

# Add project root to sys.path
sys.path.append(os.getcwd())

from verl.single_controller.base.decorator import Dispatch, register, MAGIC_ATTR
from verl.single_controller.ray.base import create_colocated_worker_cls, RayWorkerGroup
import ray

class WeightSyncMixin:
    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def create_weight_sync_group(self, master_address, master_port, rank_offset, world_size):
        pass

class BaseWorker:
    def __init__(self):
        pass

class MyWorker(BaseWorker, WeightSyncMixin):
    def __init__(self, *args, **kwargs):
        pass

def test_inspection():
    print("Inspecting MyWorker...")
    if hasattr(MyWorker.create_weight_sync_group, MAGIC_ATTR):
        print("create_weight_sync_group has MAGIC_ATTR")
    else:
        print("create_weight_sync_group DOES NOT have MAGIC_ATTR")

    print(f"Direct lookup: {getattr(MyWorker, 'create_weight_sync_group', None)}")

def test_worker_group_binding():
    print("\nTesting WorkerGroup binding...")
    # Mock class_dict
    import ray
    try:
        ray.init(ignore_reinit_error=True)
    except:
        pass
    
    RemoteWorker = ray.remote(MyWorker)
    
    # Mock RayClassWithInitArgs
    class MockCIA:
        def __init__(self, cls):
            self.cls = cls
            self.args = ()
            self.kwargs = {}
    
    class_dict = {"actor": MockCIA(RemoteWorker)}
    
    # Create WorkerDict class
    worker_dict_cls_cia = create_colocated_worker_cls(class_dict)
    WorkerDict = worker_dict_cls_cia.cls
    # Unwrap ray remote
    if hasattr(WorkerDict, "__ray_actor_class__"):
        WorkerDict = WorkerDict.__ray_actor_class__
        
    print(f"WorkerDict methods: {dir(WorkerDict)}")
    
    if hasattr(WorkerDict, 'actor_create_weight_sync_group'):
        print("SUCCESS: WorkerDict has actor_create_weight_sync_group")
    else:
        print("FAILURE: WorkerDict missing actor_create_weight_sync_group")

    # Verify RayWorkerGroup binding
    # We can't easily instantiate RayWorkerGroup without actual ray actors and resource pool
    # But we can check _bind_worker_method logic if we can mock it
    
if __name__ == "__main__":
    test_inspection()
    test_worker_group_binding()
