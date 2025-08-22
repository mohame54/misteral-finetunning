import os
import json
import math
import torch
from unsloth import FastLanguageModel, FastModel
from huggingface_hub.hf_api import HfFolder
from huggingface_hub import create_repo
from huggingface_hub import Repository
import torch.nn.functional as F


class DataHolder:
   def __init__(self, **kwargs):
       for k, v in kwargs.items():
          setattr(self, k, v)
       self.kwargs = kwargs

   def get(self, key, default=None):
       return self.kwargs.get(key, default)
   
   def __getitem__(self, key):
       return self.kwargs[key]
   

def save_model_util(
    model,
    dir_path,
    push_to_hf=False,
    repo_name="ASLM",
    user_name="moha25",
):
    os.makedirs(dir_path, exist_ok=True)
    lora_weights = f"{dir_path}/lora_model"
    model.save_pretrained(lora_weights)
    torch.save({
        "new_embd_weights":model.base_model.new_embd_weights,
        "new_lm_weights":model.base_model.new_lm_weights,
    }, "new_embd_lm_parameters.pt")
    if push_to_hf:
       save_name = dir_path.split("/")[-1]
       create_repo(repo_name, exist_ok=True)
       repo_name = f"{user_name}/{save_name}"
       repo = Repository(dir_path, clone_from=repo_name)
       repo.push_to_hub()


def hf_permission(hf_tok):
    HfFolder.save_token(hf_tok)

def load_json(file_path, as_holder=False):
    with open(file_path, "r") as f:
      data = json.load(f)
    if as_holder:
       data = DataHolder(**data)  
    return data  

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


def check_bfloat16_support(logs=True):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_properties = torch.cuda.get_device_properties(device)

        # Check if the GPU supports bfloat16
        if device_properties.major >= 8:  # Ampere (A100) and newer architectures
            if logs: print(f"GPU {device_properties.name} supports bfloat16.")
            return True
        else:
            if logs: print(f"GPU {device_properties.name} does not support bfloat16.")
    else:
        if logs: print("CUDA is not available on this system.")
    return False


def load_pretrained(model_id, **config):
    model, _ = FastModel.from_pretrained(
        model_name =model_id,
        full_finetuning = False,
        **config
        )   
    return model


def get_lr_util(it, warmup_steps=200, max_steps=500000, max_lr= 6e-4, min_lr=6e-5):
   # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)       


def set_training(model, to_train=True):
    model.train(to_train)
    for name, p in model.module.named_parameters():
        if "lora" in name:
          p.requires_grad = to_train
        else:
          p.requires_grad = not to_train  


def masked_cross_entropy_loss(logits, labels, mask):
    num_tokens = logits.size(-1)
    mask = mask.bool()
    labels_with_ignore = labels.masked_fill(~mask, -100)
    loss = F.cross_entropy(
        logits.view(-1, num_tokens),
        labels_with_ignore.view(-1)
    )
    return loss

      
def make_peft_model(
    model,
    logs=True,
    r=16,
    lora_alpha=32,
    lora_dropout=0,
    target_modules="all-linear",
    bias="none",
    **kwargs
): 
    max_len = kwargs.get("max_length", 2048)
    random_state = kwargs.get("random_state", 3407)
    lora_model = FastLanguageModel.get_peft_model(
        model,
        r = r,
        target_modules = target_modules,
        lora_alpha = lora_alpha,
        lora_dropout = lora_dropout,
        bias = bias,
        use_gradient_checkpointing = "unsloth", 
        random_state = random_state,
        max_seq_length = max_len,
        use_rslora = False, 
        loftq_config = None,
    )
    if logs:
      print_trainable_parameters(lora_model)
    for n, p in lora_model.named_parameters():
        p.requires_grad = "lora_" in n
    return  lora_model  
