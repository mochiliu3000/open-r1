# NOTE: There are 2 ways to customize the loss function:
# 1. Inherit the Trainer/SFTTrainer class and override the compute_loss method
# 2. Inherit the model class and override the forward method, as the compute_loss method in Trainer/SFTTrainer is calling the forward method to get the loss

# NOTE: We need to customize the loss function by adding the answer correctness loss or say answer reward loss. This loss is computed by comparing the answer with the reference answer. Hence, we need to pass the reference answer to the forward method and use the 2nd way to customize the loss function.

# Firstly, we need to load the model and get the model class

import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import KwargsForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache
from typing import Optional, Union, List, Tuple, Unpack


def get_model_class(model_name):
    # 加载预训练模型
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 打印模型类
    print(f"模型类: {type(model)}")  # Qwen2ForCausalLM
    print(f"模型类名: {model.__class__.__name__}")  # transformers.models.qwen2.modeling_qwen2.Qwen2PreTrainedModel
    print(f"模型基类: {model.__class__.__bases__}")  # transformers.generation.utils.GenerationMixin


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
        outputs = super().forward(**kwargs)

        # 生成文本（训练时需启用自回归生成）
        generated_ids = self.generate(input_ids, max_length=10000)
        generated_text = self.tokenizer.decode(generated_ids[0])
        
        # 提取生成文本中的答案
        predicted_answer = self.extract_boxed_answer(generated_text)  # 自定义解析函数
        true_answer = self.extract_boxed_answer(labels[0])  # 从标签解析真实答案
        
        # 计算正确性奖励（二值或连续值）
        reward = 1.0 if predicted_answer == true_answer else -1.0
        
        # 计算策略梯度损失
        log_probs = torch.log_softmax(outputs.logits, dim=-1)
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
    
model_name = "/home/jovyan/liumochi/model/Qwen/Qwen2.5-7B-Instruct"
# get_model_class(model_name)
model = CustomQwenWithReward.from_pretrained(model_name)