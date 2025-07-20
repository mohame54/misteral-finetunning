import os
import time
import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from model import (
    load_pretrained,
    make_peft_model,
    load_json,
    hf_permission,
    check_bfloat16_support
)
from train import train_epoch, validate_epoch
from huggingface_hub import HfApi
import warnings
from helpers import get_loader
import bitsandbytes as bnb


warnings.filterwarnings("ignore")


init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
os.environ['TORCH_DISTRIBUTED_DEBUG '] = "INFO"
ddp_local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)


config = load_json("params.json", as_holder=True)
hf_permission(config.hf_token_id)
api = HfApi()


dtype = torch.bfloat16 if check_bfloat16_support(ddp_rank==0) else torch.float16

#load the data

train_loader = get_loader(
    config.tok_path,
    config.train_data_path,
    config.train_batch_size,
    world_size,
    ddp_rank
)

test_loader = get_loader(
    config.tok_path,
    config.val_data_path,
    config.val_batch_size,
    world_size,
    ddp_rank
)

# Load model
model = load_pretrained(config.model_id, logs=ddp_rank==0)
for p in model.parameters():
    if p.requires_grad:
        p.data = p.data.float()

model = make_peft_model(model, ddp_rank==0, **config.peft_kwargs)
model = model.to(device)
Model = DDP(model, device_ids=[ddp_local_rank])
Opt = bnb.optim.AdamW8bit([p for p in model.parameters() if p.requires_grad], config.lr, weight_decay =config.wd)#configure_optimizers(model, 0.1, 6e-4, "cuda")

steps = 0
torch.cuda.empty_cache()
scaler = torch.amp.GradScaler()
master_process = ddp_local_rank == 0
for e in range(config.epochs):
    st = time.time()
    if master_process:
        print(f"Started Training on: {e+1} / {config.epochs}")
    train_loss = train_epoch(
        Model,
        train_loader,
        Opt,
        ddp_rank,
        scaler,
        config.max_norm,
        config.accum_steps,
        dtype,
    )
    torch.cuda.empty_cache()
    if master_process:
        val_loss = validate_epoch(
                    Model,
                    test_loader,
                    rank = ddp_rank,
                    compute_dtype=dtype
                )
        if ((e + 1) % config.get('EPOCHS_LOGS', 1)) == 0 and ddp_rank == 0:
            checkpoint_path = os.path.join(config['save_dir'], f"{config.get('checkpoint_name', 'checkpoint')}_epoch{e+1}")
            Model.model.save_pretrained(checkpoint_path)
            print(f"Saved model checkpoint: {checkpoint_path}")
            if config.get('push_hf', False):
                api.upload_folder(
                    folder_path=checkpoint_path,
                    path_in_repo=config['hf_repo_path'],
                    repo_id=config['repo_id'],
                )

        if world_size > 1:
            dist.barrier()
if config.get('push_hf', False) and ddp_rank == 0:
    api.upload_folder(
        folder_path=checkpoint_path,
        path_in_repo=config['hf_repo_path'],
        repo_id=config['repo_id'],
    )



destroy_process_group()