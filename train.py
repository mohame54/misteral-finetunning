import time
import torch
import inspect
from model import (
    masked_cross_entropy_loss,
    set_training
)
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP


mean = lambda x: sum(x) / len(x) if x else 0.0


def train_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    rank: int,
    scaler: torch.cuda.amp.GradScaler,
    max_norm: float = None,
    grad_accum_steps: int = 1,
    compute_dtype: torch.dtype = torch.float16,
):
    set_training(model)
    losses = []
    loader = tqdm(train_loader, desc="Train") if rank == 0 else train_loader
    optimizer.zero_grad()
    accum_loss = 0.0
    last_time = time.time()

    for idx, (inputs, labels, mask) in enumerate(loader):
        inputs, labels, mask = [x.to(rank) for x in (inputs, labels, mask)]

        if isinstance(model, DDP):
            model.require_backward_grad_sync = ((idx + 1) % grad_accum_steps == 0)

        with torch.autocast(device_type="cuda", dtype=compute_dtype):
            preds = model(inputs).logits
            loss = masked_cross_entropy_loss(preds, labels, mask)

        loss = loss / grad_accum_steps
        scaler.scale(loss).backward()
        accum_loss += loss.detach().item()

        if (idx + 1) % grad_accum_steps == 0 or (idx + 1) == len(train_loader):
            if max_norm is not None:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            optimizer.zero_grad()

            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            total_loss = torch.tensor(accum_loss, device=rank)
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            avg_loss = total_loss.item() / (dist.get_world_size() * grad_accum_steps)

            if rank == 0:
                losses.append(avg_loss)
                loader.set_postfix({
                    "Train Loss": f"{mean(losses):.4f}",
                    "Step Time": f"{dt:.2f}s"
                })

            accum_loss = 0.0

    return mean(losses) if rank == 0 else None


@torch.no_grad()
def validate_epoch(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    rank: int,
    compute_dtype: torch.dtype = torch.float16,
):
    set_training(model, False)
    local_sum = 0.0
    local_count = 0
    loader = tqdm(val_loader, desc="Validation") if rank == 0 else val_loader
    last_time = time.time()

    for inputs, labels, mask in loader:
        inputs, labels, mask = [x.to(rank) for x in (inputs, labels, mask)]

        with torch.autocast(device_type="cuda", dtype=compute_dtype):
            preds = model(inputs).logits
            loss = masked_cross_entropy_loss(preds, labels, mask)

        loss_val = loss.detach().item()
        local_sum += loss_val
        local_count += 1

        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        if rank == 0:
            avg_so_far = local_sum / local_count
            loader.set_postfix({
                "Val Loss": f"{avg_so_far:.4f}",
                "Step Time": f"{dt:.2f}s"
            })

    tensor = torch.tensor([local_sum, local_count], device=rank)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    total_sum, total_count = tensor.tolist()

    if total_count == 0:
        return None
    return (total_sum / total_count) if rank == 0 else None

                
def configure_optimizers(model, weight_decay, learning_rate, device_type, rank):
    # start with all of the candidate parameters (that require grad)
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    if rank == 0:
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    if rank == 0:
        print(f"using fused AdamW: {use_fused}")
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
    return optimizer