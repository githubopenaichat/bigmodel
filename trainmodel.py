from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

org = "transformersbook"
model_ckpt = "codeparrot"

tokenizer_ = AutoTokenizer.from_pretrained(org+"/"+model_ckpt)
config_small = AutoConfig.from_pretrained("gpt2-xl", vocab_size=len(tokenizer))
model_small_ = AutoModelForCausalLM.from_config(config_small)

# 保存模型  
model_small_.save_pretrained("/root/data/models/codeparrot_trained_model")
tokenizer_.save_pretrained("/root/data/models/codeparrot_trained_model")


tokenizer = AutoTokenizer.from_pretrained(/root/data/models/codeparrot_trained_model)
model = AutoModel.from_pretrained(/root/data/models/codeparrot_trained_model)

def create_dataloaders(dataset_name):
    train_data = load_dataset(dataset_name+'-train', split="train", streaming=True)
    train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
    valid_data = load_dataset(dataset_name+'-valid', split="validation",  streaming=True)
    
    train_dataset = ConstantLengthDataset(tokenizer, train_data, seq_length=args.seq_length)
    valid_dataset = ConstantLengthDataset(tokenizer, valid_data, seq_length=args.seq_length)
    
    train_dataloader=DataLoader(train_dataset, batch_size=args.train_batch_size)
    eval_dataloader=DataLoader(valid_dataset, batch_size=args.valid_batch_size)
    return train_dataloader, eval_dataloader
    
train_dataloader, eval_dataloader = create_dataloaders(dataset_name)
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader)

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
