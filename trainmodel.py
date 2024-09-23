from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
config_small = AutoConfig.from_pretrained("gpt2", vocab_size=len(tokenizer))
model_small = AutoModelForCausalLM.from_config(config_small)

model_small.save_pretrained("models/" + model_ckpt + "-small", push_to_hub=True, organization=org)

# 保存模型  
model.save_pretrained("./codeparrot_trained_model")  
tokenizer.save_pretrained("./codeparrot_trained_model")
