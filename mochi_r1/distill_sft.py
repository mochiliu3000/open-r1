'''
conda env list
conda create -n LMC python=3.11
conda activate LMC

# 添加 conda-forge 渠道
conda config --add channels conda-forge
# 设置 conda-forge 为最高优先级
conda config --set channel_priority strict

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install datasets>=3.2.0
conda install trl>=0.15.1
conda install wandb>=0.19.1
conda install transformers
conda install deepspeed==0.15.4
conda install accelerate>=1.2.1

This script is used to distill the CoT reasoning data to small models using Lora SFT
1. CoT reasoning data: https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k
2. Small model: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

To check GPU memory usage:
watch 'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits'

Also refer to: https://github.com/BinNong/deepseek-r1-llama-8b/blob/main/deepseekr1_8b%E6%9C%AC%E5%9C%B0%E5%BE%AE%E8%B0%83.ipynb


# 1 node of 8 x A100
accelerate launch --config_file=zero3.yaml distill_sft.py

'''

import wandb
from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer

# 1.Initialize wandb
wandb.login(key="e6e4d84f1ccdd4ae465070c962c94c8b24d29703")
run = wandb.init(
    project='CoT reasoning data distill',
    job_type="training",
    anonymous="allow"
)


# 2.Load data
base_dir = '/cuai/LMC'
data_dir = f'{base_dir}/data/open-thoughts/OpenThoughts-114k/metadata/*'
dataset = load_dataset("parquet", data_files=data_dir)


# 3.Preprocess data
# According to question domain, split the data into different domain and train/test datasets
sample_config = {
    'math': [3000, 600], # [train, test]
    'code': [3000, 600],
    'science': {
        'biology': [1000, 200],
        'physics': [1000, 200],
        'chemistry': [1000, 200]
    },
    'puzzle': [1000, 200]
}

train_data = list()
test_data = list()

# Iterate over each category and do sampling
for category, count in sample_config.items():
    if isinstance(count, dict):
        for subcategory, subcount in count.items():
            domain_samples = dataset['train'].filter(lambda x: x['domain'] == subcategory)
            train_samples = domain_samples.select(range(subcount[0]))
            test_samples = domain_samples.select(range(subcount[0], subcount[0] + subcount[1]))
            train_data.extend(train_samples)
            test_data.extend(test_samples)
            print(f"INFO - Domain: {subcategory}, Add train samples: {len(train_samples)}, Add test samples: {len(test_samples)}")
    else:
        domain_samples = dataset['train'].filter(lambda x: x['domain'] == category)
        train_samples = domain_samples.select(range(count[0]))
        test_samples = domain_samples.select(range(count[0], count[0] + count[1]))
        train_data.extend(train_samples)
        test_data.extend(test_samples)
        print(f"INFO - Domain: {category}, Add train samples: {len(train_samples)}, Add test samples: {len(test_samples)}")

train_dataset = Dataset.from_list(train_data)
test_dataset = Dataset.from_list(test_data)
print(f"INFO - Train dataset created with length: {len(train_dataset)}")
print(f"INFO - Test dataset created with length: {len(test_dataset)}")


prompt_template = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and answer are enclosed within <think> </think> and
<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>.\n## User: {}.\n## Assistant: {}"""

# Format the data with prompt template
def formatting_prompt_template(examples):
    problems = examples["problem"]
    cots = examples["deepseek_reasoning"]
    outputs = examples["deepseek_solution"]
    texts = []
    for problem, cot, output in zip(problems, cots, outputs):
        assistant_answer = "<think>" + cot + "</think>\n<answer>" + output + "</answer>"
        text = prompt_template.format(problem, assistant_answer) + EOS_TOKEN
        texts.append(text)
    return {
        "text": texts
    }

model_name_or_path = f"{base_dir}/model/Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
EOS_TOKEN = tokenizer.eos_token
tokenizer.pad_token = tokenizer.eos_token

formatted_train_dataset = train_dataset.map(formatting_prompt_template, batched=True)
formatted_test_dataset = test_dataset.map(formatting_prompt_template, batched=True)
# print(formatted_train_dataset["text"][0])


# 4.Setup SFT trainer
# https://huggingface.co/docs/trl/v0.15.1/en/sft_trainer#trl.SFTConfig
training_args = SFTConfig(
    run_name="Qwen2.5-7B-Instruct-SFT",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs = 1,
    max_steps=-1,
    warmup_ratio=0.05,
    learning_rate=5e-5,
    bf16=True,
    logging_steps=10,
    weight_decay=0.01,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr_rate": 0.1},
    seed=2050,
    output_dir=f"{base_dir}/outputs",
)

# https://huggingface.co/docs/trl/main/en/sft_trainer#trl.SFTTrainer
trainer = SFTTrainer(
    model=model_name_or_path,
    tokenizer=tokenizer,
    train_dataset=formatted_train_dataset,
    eval_dataset=formatted_test_dataset,
    args=training_args
)


# 5.Train model with SFT
checkpoint = None
if checkpoint is not None:
    print(f"INFO - Resume training from checkpoint: {checkpoint}")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
else:
    print("INFO - Start training...")
    train_result = trainer.train()
train_metrics = train_result.metrics
trainer.log_metrics("train", train_metrics)
trainer.save_metrics("train", train_metrics)
trainer.save_state()


# 6.Save model
model_path = f"{base_dir}/checkpoints/Qwen2.5-7B-Instruct-SFT"
trainer.save_model(model_path)
print(f"INFO - Save model to {model_path}")


# 7.Evaluate model
print("INFO - Evaluate model...")
test_metrics = trainer.evaluate()
trainer.log_metrics("eval", test_metrics)
trainer.save_metrics("eval", test_metrics)