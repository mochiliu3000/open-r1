'''
conda env list
conda create -n LMC python=3.11
conda activate LMC

# 添加 conda-forge 渠道
conda config --add channels conda-forge
# 设置 conda-forge 为最高优先级
conda config --set channel_priority strict

pip3 install pytorch pytorch-cuda -c pytorch -c nvidia
pip3 install datasets trl wandb transformers deepspeed accelerate flash_attn

This script is used to distill the CoT reasoning data to small models using Lora SFT
1. CoT reasoning data: https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k
2. Small model: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

To check GPU memory usage:
watch 'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits'

Also refer to: https://github.com/BinNong/deepseek-r1-llama-8b/blob/main/deepseekr1_8b%E6%9C%AC%E5%9C%B0%E5%BE%AE%E8%B0%83.ipynb


accelerate launch --config_file=zero3.yaml distill_sft.py

'''
import os
import json
from functools import partial
import wandb
import torch
from datasets import load_dataset, Dataset, load_from_disk
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer


# Initialize wandb
def init_wandb(key):
    wandb.login(key=key)
    run = wandb.init(
        project='CoT reasoning data distill',
        job_type="training",
        anonymous="allow"
    )


# Load open thoughts data - load processed data if exists, otherwise load from raw data
def load_open_thoughts_data(processed_dir, base_dir, sample_type="all"):   
    train_dataset_dir = f'{processed_dir}/{sample_type}_train_dataset'
    test_dataset_dir = f'{processed_dir}/{sample_type}_test_dataset'
    if os.path.exists(train_dataset_dir) and os.path.exists(test_dataset_dir): 
        train_dataset = load_from_disk(train_dataset_dir)
        test_dataset = load_from_disk(test_dataset_dir)
        print(f"INFO - Load processed dataset from {processed_dir}")
        return train_dataset, test_dataset
    else:
        # Load data
        data_dir = f'{base_dir}/data/open-thoughts/OpenThoughts-114k/metadata/*'
        dataset = load_dataset("parquet", data_files=data_dir)

        # Preprocess data
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
        
        sample_config_math_code = {
            'math': [3000, 0], # [train, test]
            'code': [3000, 0],
            'science': {
                'biology': [0, 200],
                'physics': [0, 200],
                'chemistry': [0, 200]
            },
            'puzzle': [0, 200]
        }
        
        sample_config_puzzle_science = {
            'math': [0, 600], # [train, test]
            'code': [0, 600],
            'science': {
                'biology': [1000, 0],
                'physics': [1000, 0],
                'chemistry': [1000, 0]
            },
            'puzzle': [1000, 0]
        }

        train_data = list()
        test_data = list()

        if sample_type == "math_code":
            sample_config = sample_config_math_code
        elif sample_type == "puzzle_science":
            sample_config = sample_config_puzzle_science
        
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
        train_dataset.save_to_disk(f'{processed_dir}/{sample_type}_train_dataset')
        test_dataset.save_to_disk(f'{processed_dir}/{sample_type}_test_dataset')
        return train_dataset, test_dataset


# Format the data with prompt template - pre-CoT
def formatting_prompt_template_pre_CoT(examples):
    '''
    # NOTE: Format the data in the following style before utilzing the chat_template
    [
        {
            "content": "You are an Assistant good at math, coding, science and puzzling. When user asks a question, you firstly think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>\n<answer> answer here </answer>. The last line within <answer></answer> should be in the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$.'",
            "role": "system"
        },
        {
            "content": "[Problem]",
            "role": "user"
        },
        {
            "content": "<think>[Deepseek Reasoning]</think>\n<answer>[Deepseek Solution]</answer>",
            "role": "assistant"
        }
    ]
    '''

    problems = examples["problem"]
    cots = examples["deepseek_reasoning"]
    outputs = examples["deepseek_solution"]
    messages = []
    for problem, cot, output in zip(problems, cots, outputs):
        prompt_formatted = [
            {
                "content": "You are an Assistant good at math, coding, science and puzzling. When user asks a question, you firstly think about the reasoning process in mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think></think> and <answer></answer> tags, respectively, i.e., <think>reasoning process here</think>\n<answer>answer here</answer>. The last line within <answer></answer> should be in the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$.'",
                "role": "system"
            },
            {
                "content": problem,
                "role": "user"
            },
            {
                "content": f"\n<think>{cot}</think>\n<answer>{output}</answer>",
                "role": "assistant"
            }
        ]

        messages.append(prompt_formatted)
    return {
        "messages": messages
    }


# Format the data with prompt template - post-CoT
def formatting_prompt_template_post_CoT(examples):
    '''
    # NOTE: Format the data in the following style before utilzing the chat_template
    [
        {
            "content": "You are an Assistant good at math, coding, science and puzzling. When user asks a question, you firstly think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <answer> </answer> and <think> </think>tags, respectively, i.e., <answer> answer here </answer>\n<think> reasoning process here </think>.",
            "role": "system"
        },
        {
            "content": "[Problem]",
            "role": "user"
        },
        {
            "content": "<answer>[Deepseek Solution]</answer>\n<think>[Deepseek Reasoning]</think>",
            "role": "assistant"
        }
    ]
    '''

    problems = examples["problem"]
    cots = examples["deepseek_reasoning"]
    outputs = examples["deepseek_solution"]
    messages = []
    for problem, cot, output in zip(problems, cots, outputs):
        prompt_formatted = [
            {
                "content": "You are an Assistant good at math, coding, science and puzzling. When user asks a question, you firstly think about the reasoning process in mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think></think> and <answer></answer> tags, respectively, i.e., <answer>answer here</answer>\n<think>reasoning process here</think>. The last line within <answer></answer> should be in the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$.'",
                "role": "system"
            },
            {
                "content": problem,
                "role": "user"
            },
            {
                "content": f"\n<answer>{output}</answer>\n<think>{cot}</think>",
                "role": "assistant"
            }
        ]

        messages.append(prompt_formatted)
    return {
        "messages": messages
    }


# Setup SFT trainer
def setup_sft_trainer(model_name_or_path, formatted_train_dataset, formatted_test_dataset):
    # https://huggingface.co/docs/trl/v0.15.1/en/sft_trainer#trl.SFTConfig
    training_args = SFTConfig(
        model_init_kwargs={
            "torch_dtype": "bfloat16",
            "attn_implementation": "flash_attention_2"
        },
        run_name="Qwen2.5-7B-Instruct-SFT",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        max_steps=-1,
        warmup_ratio=0.05,
        learning_rate=5e-5,
        bf16=True,
        gradient_checkpointing=True,
        max_seq_length=4096,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=25,
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
        train_dataset=formatted_train_dataset,
        eval_dataset=formatted_test_dataset,
        processing_class=tokenizer,
        args=training_args
    )

    return trainer


if __name__ == "__main__":
    # 1.Initialize wandb
    wandb_key="e6e4d84f1ccdd4ae465070c962c94c8b24d29703"
    #init_wandb(wandb_key)

    # 2.Load open thoughts data
    base_dir = '/home/jovyan/liumochi' # '/cuai/LMC' 
    processed_dir = f'{base_dir}/data/open_thoughts_processed'
    sample_type = 'all' #'puzzle_science' # math_code
    train_dataset, test_dataset = load_open_thoughts_data(processed_dir, base_dir, sample_type)

    # 3.Setup model
    model_name_or_path = f"{base_dir}/model/Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    EOS_TOKEN = tokenizer.eos_token
    tokenizer.pad_token = tokenizer.eos_token

    # 4.Format data with prompt template
    formatted_train_dataset = train_dataset.map(
        formatting_prompt_template_pre_CoT, 
        batched=True).select_columns(["messages"])
    formatted_test_dataset = test_dataset.map(
        formatting_prompt_template_pre_CoT,
        batched=True).select_columns(["messages"])
    formatted_train_dataset.save_to_disk(f'{processed_dir}/{sample_type}_formatted_train_dataset_pre_CoT')
    formatted_test_dataset.save_to_disk(f'{processed_dir}/{sample_type}_formatted_test_dataset_pre_CoT')
    # print(formatted_train_dataset["text"][0])

    """
    # 5.Setup SFT trainer
    print(f'>>>>>>>Device Cuda:{torch.cuda.current_device()} used GPU memory:{torch.cuda.memory_allocated() // 1048576} MB')
    trainer = setup_sft_trainer(model_name_or_path, formatted_train_dataset, formatted_test_dataset)
    print(f'>>>>>>>Device Cuda:{torch.cuda.current_device()} used GPU memory:{torch.cuda.memory_allocated() // 1048576} MB')

    # 6.Train model with SFT
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
    print(f'>>>>>>>Device Cuda:{torch.cuda.current_device()} used GPU memory:{torch.cuda.memory_allocated() // 1048576} MB')

    # 7.Save model
    model_path = f"{base_dir}/checkpoints/Qwen2.5-7B-Instruct-SFT"
    trainer.save_model(model_path)
    print(f"INFO - Save model to {model_path}")

    # 8.Evaluate model
    print("INFO - Evaluate model...")
    test_metrics = trainer.evaluate()
    trainer.log_metrics("eval", test_metrics)
    trainer.save_metrics("eval", test_metrics)
    """