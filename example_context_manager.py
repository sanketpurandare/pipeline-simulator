from functools import wraps
import torch
from typing import Optional, Any
from contextlib import contextmanager, nullcontext
from torch.utils._python_dispatch import TorchDispatchMode
from torch._subclasses.fake_tensor import FakeTensorMode
import torch.distributed as dist
from torch._C._distributed_c10d import Work, FakeWork
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.distributed.pipelining.stage import _PipelineStageBase


@contextmanager
def capture_collectives():
    saved_forward_one_chunk = _PipelineStageBase.forward_one_chunk
    
    saved_backward_one_chunk = _PipelineStageBase.backward_one_chunk
    
    saved_backward_weight_one_chunk = _PipelineStageBase.backward_weight_one_chunk

    @wraps(_PipelineStageBase.forward_one_chunk)
    def forward_one_chunk(
        self,
        fwd_chunk_id: int,
        args: tuple[Any, ...],
        kwargs: Optional[dict[str, Any]] = None,
    ):
        print("GPU #", self.stage_index, ": Forwarded mb", fwd_chunk_id)
        return saved_forward_one_chunk(self, fwd_chunk_id, args, kwargs)
    
    @wraps(_PipelineStageBase.backward_one_chunk)
    def backward_one_chunk(
        self,
        bwd_chunk_id: int,
        loss=None,
        full_backward: bool = True,
        last_backward=False,
    ):
        if (full_backward == True):
            print("GPU #", self.stage_index, ": Backwarded mb", bwd_chunk_id)
        else:
            print("GPU #", self.stage_index, ": Inputs Backwarded mb", bwd_chunk_id)

        return saved_backward_one_chunk(self, bwd_chunk_id, loss, full_backward, last_backward)
    
    @wraps(_PipelineStageBase.backward_weight_one_chunk)
    def backward_weight_one_chunk(
        self, 
        bwd_chunk_id: int, 
        last_backward=False,
    ):
        print("GPU #", self.stage_index, ": Weights Backwarded mb", bwd_chunk_id)

        return saved_backward_weight_one_chunk(self, bwd_chunk_id, last_backward)
    
    try:
        _PipelineStageBase.forward_one_chunk = forward_one_chunk
        _PipelineStageBase.backward_one_chunk = backward_one_chunk
        _PipelineStageBase.backward_weight_one_chunk = backward_weight_one_chunk
        yield
    finally:
        _PipelineStageBase.forward_one_chunk = saved_forward_one_chunk
        _PipelineStageBase.backward_one_chunk = saved_backward_one_chunk
        _PipelineStageBase.backward_weight_one_chunk = saved_backward_weight_one_chunk

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