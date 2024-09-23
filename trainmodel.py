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


