import torch
import pandas as pd
from dataclasses import dataclass
from typing import List, Any, Optional, Dict
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.messages import UserMessage, AssistantMessage


@dataclass
class TokenSample:
    tokens: List[int]
    masks: List[bool]


@dataclass
class TrainInstructReq:
    messages: List[Any]
    only_last: Optional[bool] = False
    available_tools: Optional[List[Dict[str, Any]]] = None
    system_prompt: Optional[str] = None


PAD_ID = 0


def collate_fn(batch):
    inputs, outputs, masks = [], [], []
    max_len = 0
    for x, y, mask in batch:
        if len(x) > max_len:
            max_len = len(x)
        inputs.append(x)
        outputs.append(y)
        masks.append(mask)
    
    for i in range(len(inputs)):
        inp, tar, ma = inputs[i], outputs[i], masks[i]
        if max_len > len(inp):
            pad_len = max_len  - len(inp)
            pad_tensor = torch.tensor([PAD_ID] * (pad_len), dtype=torch.long) 
            inp = torch.concat([inp, pad_tensor])
            tar = torch.concat([tar, pad_tensor])
            ma = torch.concat([ma, torch.tensor([False] * pad_len, dtype=torch.bool)])
            inputs[i] = inp
            outputs[i] = tar
            masks[i] = ma

    inputs = torch.stack(inputs)
    outputs = torch.stack(outputs)
    masks = torch.stack(masks)
    return inputs, outputs, masks
    

class TokenDataset(Dataset):
    def __init__(self, tokenizer_pth, data_pth, block_size=2048):
        self.tokenizer = MistralTokenizer.from_file(tokenizer_pth)
        self.block_size = block_size
        self.df = pd.read_json(data_pth, lines=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        messages = row['messages']
        system_prompt = messages[0]['content']
        messages[1]['content'] = f"{system_prompt}\n\n{messages[1]['content']}"
        chat = from_completion2InstructTokens(messages[1:], None)
        sample = tokenize_instruct(chat, self.tokenizer.instruct_tokenizer)
        x = torch.tensor(sample.tokens[:-1], dtype=torch.long)
        y = torch.tensor(sample.tokens[1:], dtype=torch.long)
        mask = torch.tensor(sample.masks[1:], dtype=torch.bool)
        return x, y, mask


def get_loader(tokenizer_pth, data_pth, batch_size, world_size, rank):
    ds = TokenDataset(tokenizer_pth, data_pth)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, drop_last=True, collate_fn=collate_fn)


def from_completion2InstructTokens(
   messages,
   system_prompt
) -> TrainInstructReq:
    req = ChatCompletionRequest(messages=messages)
    messages = req.messages
    return TrainInstructReq(
        messages,
        system_prompt=system_prompt
    )

def tokenize_instruct(
    sample: TrainInstructReq,
    instruct_tokenizer,
) -> TokenSample:
    tokens: List[int] = instruct_tokenizer.start()
    masks: List[bool] = [False]

    mask_all_but_last = sample.only_last
    user_messages = [
        i for i, msg in enumerate(sample.messages) if isinstance(msg, UserMessage)
    ]
    first_user_idx = user_messages[0] if user_messages else -1
    last_user_idx = user_messages[-1] if user_messages else -1

    for msg_idx, message in enumerate(sample.messages):
        if isinstance(message, UserMessage):
            curr_tokens = instruct_tokenizer.encode_user_message(
                message,
                available_tools=sample.available_tools,
                is_last=msg_idx == last_user_idx,
                is_first=msg_idx == first_user_idx,
                system_prompt=sample.system_prompt,
            )
            if isinstance(curr_tokens, tuple):
                curr_tokens = curr_tokens[0]
                
            curr_masks = [False] * len(curr_tokens)  # only predict bot answers

        elif isinstance(message, AssistantMessage):
            is_last_message = msg_idx == (len(sample.messages) - 1)

            curr_tokens = instruct_tokenizer.encode_assistant_message(
                message, is_before_last_user_message=False, continue_message=False
            )
            is_relevant = (not mask_all_but_last) or is_last_message
            if is_relevant:
                curr_masks = [True] * len(curr_tokens)  # only predict bot answers
            else:
                # in function calling we only backprop through last message
                curr_masks = [False] * len(curr_tokens)

        tokens.extend(curr_tokens)
        masks.extend(curr_masks)

    return TokenSample(tokens, masks)