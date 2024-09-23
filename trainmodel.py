from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer,AdamW,get_scheduler,AutoModel
from huggingface_hub import login, Repository
from accelerate import Accelerator
from datasets import load_dataset
import torch
from accelerate import Accelerator
import datasets
import transformers
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
import wandb

import torch
from torch.utils.data import IterableDataset

from argparse import Namespace

org = "transformersbook"
model_ckpt = "codeparrot"

tokenizer_ = AutoTokenizer.from_pretrained(org+"/"+model_ckpt)
config_small = AutoConfig.from_pretrained("gpt2", vocab_size=len(tokenizer_))
model_small_ = AutoModelForCausalLM.from_config(config_small)

config = {"train_batch_size": 2, # 12
          "valid_batch_size": 2, # 12
          "weight_decay": 0.1,
          "shuffle_buffer": 1000,
          "learning_rate": 2e-4, # 5e-4
          "lr_scheduler_type": "cosine",
          "num_warmup_steps": 750, # 2000
          "gradient_accumulation_steps": 16, # 1
          "max_train_steps": 50000, # 150000
          "max_eval_steps": -1,
          "seq_length": 1024,
          "seed": 1,
          "save_checkpoint_steps": 50000} # 15000

args = Namespace(**config)

# 保存模型  
model_small_.save_pretrained("/root/data/bigmodel/models/codeparrot-small")
tokenizer_.save_pretrained("/root/data/bigmodel/models/codeparrot-small")


tokenizer = AutoTokenizer.from_pretrained("/root/data/bigmodel/models/codeparrot-small")
model = AutoModel.from_pretrained("/root/data/bigmodel/models/codeparrot-small")


class ConstantLengthDataset(IterableDataset):    
    def __init__(self, tokenizer, dataset, seq_length=1024,
                 num_of_sequences=1024, chars_per_token=3.6):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.input_characters = seq_length * chars_per_token * num_of_sequences
    
    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.input_characters:
                    m=f"Buffer full: {buffer_len}>={self.input_characters:.0f}"
                    print(m)
                    break
                try:
                    m=f"Fill buffer: {buffer_len}<{self.input_characters:.0f}"
                    print(m)
                    buffer.append(next(iterator)["content"])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    iterator = iter(self.dataset)

            all_token_ids = []
            tokenized_inputs = self.tokenizer(buffer, truncation=False)
            for tokenized_input in tokenized_inputs['input_ids']:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    yield torch.tensor(input_ids)

def get_grouped_params(model, no_decay=["bias", "LayerNorm.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [{'params': params_with_wd, 'weight_decay': args.weight_decay},
            {'params': params_without_wd, 'weight_decay': 0.0}]

def create_dataloaders(dataset_name):
    train_data = load_dataset(dataset_name+'-train', split="train", streaming=True)
    train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
    valid_data = load_dataset(dataset_name+'-valid', split="validation",  streaming=True)
    
    train_dataset = ConstantLengthDataset(tokenizer, train_data, seq_length=args.seq_length)
    valid_dataset = ConstantLengthDataset(tokenizer, valid_data, seq_length=args.seq_length)
    
    train_dataloader=DataLoader(train_dataset, batch_size=args.train_batch_size)
    eval_dataloader=DataLoader(valid_dataset, batch_size=args.valid_batch_size)
    return train_dataloader, eval_dataloader

dataset_name = "transformersbook/codeparrot"
accelerator = Accelerator()
optimizer = AdamW(get_grouped_params(model), lr=args.learning_rate)
train_dataloader, eval_dataloader = create_dataloaders(dataset_name)
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader)

#Token indices sequence length is longer than the specified maximum sequence length for this model (2626 > 1024). Running this sequence through the model will result in indexing errors
#Traceback (most recent call last):
#  File "/root/data/bigmodel/train.py", line 118, in <module>
#    loss = model(batch, labels=batch).loss
#  File "/opt/miniconda/envs/MiniCPMV/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
#    return self._call_impl(*args, **kwargs)
#  File "/opt/miniconda/envs/MiniCPMV/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
#    return forward_call(*args, **kwargs)
#TypeError: GPT2Model.forward() got an unexpected keyword argument 'labels'



completed_steps = 0
for step, batch in enumerate(train_dataloader, start=1):
    loss = model(batch, labels=batch).loss
    log_metrics(step, {'lr': get_lr(), 'samples': step*samples_per_step,
                       'steps': completed_steps, 'loss/train': loss.item()})
    loss = loss / args.gradient_accumulation_steps
    accelerator.backward(loss)
    if step % args.gradient_accumulation_steps == 0:
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        completed_steps += 1
    if step % args.save_checkpoint_steps == 0:
        logger.info('Evaluating and saving model checkpoint')
        eval_loss, perplexity = evaluate()
        log_metrics(step, {'loss/eval': eval_loss, 'perplexity': perplexity})
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        if accelerator.is_main_process:
            unwrapped_model.save_pretrained("./")
            hf_repo.push_to_hub(commit_message=f'step {step}')
        model.train()
    if completed_steps >= args.max_train_steps:
        break
