# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Supervised fine-tuning script for decoder language models.

Usage:

# One 1 node of 8 x H100s
accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name HuggingFaceH4/Bespoke-Stratos-17k \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir data/Qwen2.5-1.5B-Open-R1-Distill


ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file mochi_r1/zero3.yaml \
  src/open_r1/sft.py \
  --config mochi_r1/recipes/Qwen2.5-1.5B-Instruct/sft/config_demo_all.yaml
"""

import logging
import os
import sys
import re

from transformers import Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import KwargsForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache
from typing import Optional, Union, List, Tuple, Unpack

import datasets
import torch
import transformers
from datasets import load_dataset, load_from_disk
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import SFTConfig
from open_r1.utils import get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


logger = logging.getLogger(__name__)


class CustomQwenWithReward(Qwen2ForCausalLM):
    def extract_boxed_answer(self, text):
        """
        从文本中提取被 \boxed{} 包围的答案
        
        Args:
            text (str): 包含答案的文本
            
        Returns:
            str: 提取出的答案，如果没有找到则返回空字符串
        """
        # 使用正则表达式匹配 \boxed{...} 格式的内容
        pattern = r'\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}'
        matches = re.findall(pattern, text)
        
        # 如果找到匹配项，返回第一个匹配的内容
        if matches:
            return matches[0].strip()
        
        # 如果没有找到标准格式，尝试查找其他可能的格式
        alt_patterns = [
            r'\[\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}\]',  # [\boxed{...}]
            r'\\boxed\s*\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}',   # \boxed {...}
            r'boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}'         # boxed{...}
        ]
        
        for pattern in alt_patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[0].strip()
                
        return ""
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

        # 适用于Post-CoT策略，我们只生成100个token，让模型先预测正确答案，不输出推理过程
        generate_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'max_new_tokens': 20,  # 限制生成的新token数量
            'num_beams': 1,        # 使用贪婪搜索减少显存
            'use_cache': True,     # 使用缓存加速
            'pad_token_id': self.config.pad_token_id,
            'eos_token_id': self.config.eos_token_id,
        }

        # 在无梯度上下文中生成以避免显存累积
        with torch.no_grad():
            original_mode = self.training
            self.eval()
            generated_ids = self.generate(**generate_kwargs)
            self.train(original_mode)

        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # 提取生成文本中的答案
        predicted_answer = self.extract_boxed_answer(generated_text)  # 自定义解析函数
        true_answer = self.extract_boxed_answer(labels[0])  # 从标签解析真实答案
        
        # 计算正确性奖励（二值或连续值）
        reward = 1.0 if predicted_answer == true_answer else -1.0
        
 # 为生成的序列创建attention mask
        gen_attention_mask = (generated_ids != self.config.pad_token_id).int()

        # 重新前向传播以获取生成序列的logits（保留梯度）
        outputs_rl = super().forward(
            input_ids=generated_ids,
            attention_mask=gen_attention_mask,
        )
        log_probs = torch.log_softmax(outputs_rl.logits, dim=-1)
        selected_log_probs = log_probs[:, :-1].gather(2, generated_ids[:, 1:].unsqueeze(-1)).squeeze()
        rl_loss = -torch.mean(selected_log_probs * reward)
        
        total_loss = outputs.loss + rl_loss
        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    logger.info("*** Loading last checkpoint if exists ***")
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        print(">>>>>>>>>> Found last checkpoint dir")
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        print(">>>>>>>>>> last_checkpoint is None:", last_checkpoint is None)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f">>>>>>>>>> Checkpoint detected, resuming training at {last_checkpoint}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    ################
    # Load datasets
    ################
    #dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    
    if training_args.do_train:
        train_dataset = load_from_disk(script_args.dataset_name)
    if training_args.do_eval:
        test_dataset = load_from_disk(training_args.evalset_name)

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)
    tokenizer.pad_token = tokenizer.eos_token

    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs

    ############################
    # Initialize the SFT Trainer
    ############################
    trainer = SFTTrainer(
        #model=model_args.model_name_or_path,
        model=CustomQwenWithReward.from_pretrained(model_args.model_name_or_path),
        args=training_args,
        #train_dataset=dataset[script_args.dataset_train_split],
        #eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
    )

    ###############
    # Training loop
    ###############
    if training_args.do_train:
        logger.info("*** Train ***")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    if training_args.do_save:
        logger.info("*** Save model ***")
        trainer.save_model(training_args.output_dir)
        logger.info(f"Model saved to {training_args.output_dir}")

        # Save everything else on main process
        kwargs = {
            "dataset_name": script_args.dataset_name,
            "tags": ["open-r1"],
        }
        if trainer.accelerator.is_main_process:
            trainer.create_model_card(**kwargs)
            # Restore k,v cache for fast inference
            trainer.model.config.use_cache = True
            trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(test_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)