from transformers import AutoTokenizer
import json


model_name = "/home/jovyan/liumochi/model/Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
messages = [
    {"role": "system", "content": "你是一个法律助手，需专业回答法律问题。"},
    {"role": "user", "content": "合同违约的赔偿标准是什么？"}
]

ai_mo_messages = [
    {
        "content": "What is the coefficient of $x^2y^6$ in the expansion of $\\left(\\frac{3}{5}x-\\frac{y}{2}\\right)^8$? Express your answer as a common fraction.",
        "role": "user"
    },
    {
        "content": "To determine the coefficient of \\(x^2y^6\\) in the expansion of \\(\\left(\\frac{3}{5}x - \\frac{y}{2}\\right)^8\\), we can use the binomial theorem.\n\nThe binomial theorem states:\n\\[\n(a + b)^n = \\sum_{k=0}^{n} \\binom{n}{k} a^{n-k} b^k\n\\]\n\nIn this case, \\(a = \\frac{3}{5}x\\), \\(b = -\\frac{y}{2}\\), and \\(n = 8\\).\n\nWe are interested in the term that contains \\(x^2y^6\\). In the general term of the binomial expansion:\n\\[\n\\binom{8}{k} \\left(\\frac{3}{5}x\\right)^{8-k} \\left(-\\frac{y}{2}\\right)^k\n\\]\n\nTo get \\(x^2\\), we need \\(8 - k = 2\\), thus \\(k = 6\\).\n\nSubstituting \\(k = 6\\) into the expression:\n\\[\n\\binom{8}{6} \\left(\\frac{3}{5}x\\right)^{8-6} \\left(-\\frac{y}{2}\\right)^6 = \\binom{8}{6} \\left(\\frac{3}{5}x\\right)^2 \\left(-\\frac{y}{2}\\right)^6\n\\]\n\nNow, we will compute each part of this expression.\n\n1. Calculate the binomial coefficient \\(\\binom{8}{6}\\).\n2. Compute \\(\\left(\\frac{3}{5}\\right)^2\\).\n3. Compute \\(\\left(-\\frac{y}{2}\\right)^6\\).\n4. Combine everything together to get the coefficient of \\(x^2y^6\\).\n\nLet's compute these in Python.\n```python\nfrom math import comb\n\n# Given values\nn = 8\nk = 6\n\n# Calculate the binomial coefficient\nbinom_coeff = comb(n, k)\n\n# Compute (3/5)^2\na_term = (3/5)**2\n\n# Compute (-1/2)^6\nb_term = (-1/2)**6\n\n# Combine terms to get the coefficient of x^2y^6\ncoefficient = binom_coeff * a_term * b_term\nprint(coefficient)\n```\n```output\n0.1575\n```\nThe coefficient of \\(x^2y^6\\) in the expansion of \\(\\left(\\frac{3}{5}x - \\frac{y}{2}\\right)^8\\) is \\(0.1575\\). To express this as a common fraction, we recognize that:\n\n\\[ 0.1575 = \\frac{1575}{10000} = \\frac{63}{400} \\]\n\nThus, the coefficient can be expressed as:\n\n\\[\n\\boxed{\\frac{63}{400}}\n\\]",
        "role": "assistant"
    }
]

# 应用模板并添加生成提示
formatted_input = tokenizer.apply_chat_template(
    messages, 
    tokenize=False,
    add_generation_prompt=True
)

ai_mo_formatted_input = tokenizer.apply_chat_template(
    ai_mo_messages, 
    tokenize=False,
    add_generation_prompt=True
)

print('messages:')
print(json.dumps(messages, indent=4, ensure_ascii=False))
print('-' * 100)
print(f'{tokenizer.name_or_path.split("/")[-1]} chat template:')
print(tokenizer.chat_template)
print('-' * 100)
print(f'{tokenizer.name_or_path.split("/")[-1]} template formatted output:')
print(formatted_input)
print('-' * 100)
""" Qwen template format
<|im_start|>system
你是一个法律助手，需专业回答法律问题。<|im_end|>
<|im_start|>user
合同违约的赔偿标准是什么？<|im_end|>
<|im_start|>assistant
"""
print('formatted ai_mo_messages:')
print(ai_mo_formatted_input)
print('=' * 100)

###################################################################################

DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE 

custom_formatted_input = tokenizer.apply_chat_template(
    messages, 
    tokenize=False,
    add_generation_prompt=True
)

ai_mo_formatted_input = tokenizer.apply_chat_template(
    ai_mo_messages, 
    tokenize=False,
    add_generation_prompt=True
)

print('custom chat template:')
print(tokenizer.chat_template)
print('-' * 100)
print('custom template formatted output:')
print(custom_formatted_input)
print('-' * 100)
""" custom template format
<|system|>
你是一个法律助手，需专业回答法律问题。<|im_end|>
<|user|>
合同违约的赔偿标准是什么？<|im_end|>
<|assistant|>
"""
print('custom ai_mo template formatted output:')
print(ai_mo_formatted_input)
print('-' * 100)