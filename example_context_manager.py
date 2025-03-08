from functools import wraps
import torch
from contextlib import contextmanager, nullcontext
from torch.utils._python_dispatch import TorchDispatchMode
from torch._subclasses.fake_tensor import FakeTensorMode
import torch.distributed as dist
from torch._C._distributed_c10d import Work, FakeWork
from torch.testing._internal.distributed.fake_pg import FakeStore


@contextmanager
def capture_collectives():
    saved_batch_isend_irecv = dist.batch_isend_irecv

    @wraps(dist.batch_isend_irecv)
    def batch_isend_irecv(
        *args,
        **kwargs
    ):
        print("Captured Method")
        return saved_batch_isend_irecv(*args, **kwargs)
    
    try:
        dist.batch_isend_irecv = batch_isend_irecv
        yield
    finally:
        dist.batch_isend_irecv = saved_batch_isend_irecv

if __name__ == "__main__":
    use_nccl = False
    use_fake_mode = False
    world_size = 2
    rank = 0
    torch.set_default_device("cuda")
    if use_nccl:
        dist.init_process_group("nccl", world_size=world_size, rank=rank)
    else:
        store = FakeStore()
        dist.init_process_group(
            "fake", rank=rank, world_size=world_size, store=store
        )
    with capture_collectives():
        with FakeTensorMode() if use_fake_mode else nullcontext():
            send_tensor = torch.arange(2, dtype=torch.float32) + 2 * rank
            recv_tensor = torch.randn(2, dtype=torch.float32)
            send_op = dist.P2POp(dist.isend, send_tensor, (rank + 1)%world_size)
            recv_op = dist.P2POp(dist.irecv, recv_tensor, (rank - 1 + world_size)%world_size)
            reqs = dist.batch_isend_irecv([send_op, recv_op])
            for req in reqs:
                req.wait()
            recv_tensor