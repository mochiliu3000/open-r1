from transformers import AutoTokenizer, PreTrainedTokenizer

from trl import ModelConfig

from ..configs import GRPOConfig, SFTConfig

# NOTE: AI-MO/NuminaMath-TIR 'messages' column has the following format. Need to use the chat_template to format the data. This chat_template can be attached to the tokenizer
""" 
# https://huggingface.co/datasets/AI-MO/NuminaMath-TIR/viewer/default/train?row=0
[
    {
        "content": "What is the coefficient of $x^2y^6$ in the expansion of $\\left(\\frac{3}{5}x-\\frac{y}{2}\\right)^8$? Express your answer as a common fraction.",
        "role": "user"
    },
    {
        "content": "To determine the coefficient of \\(x^2y^6\\) in the expansion of \\(\\left(\\frac{3}{5}x - \\frac{y}{2}\\right)^8\\), we can use the binomial theorem.\n\nThe binomial theorem states:\n\\[\n(a + b)^n = \\sum_{k=0}^{n} \\binom{n}{k} a^{n-k} b^k\n\\]\n\nIn this case, \\(a = \\frac{3}{5}x\\), \\(b = -\\frac{y}{2}\\), and \\(n = 8\\).\n\nWe are interested in the term that contains \\(x^2y^6\\). In the general term of the binomial expansion:\n\\[\n\\binom{8}{k} \\left(\\frac{3}{5}x\\right)^{8-k} \\left(-\\frac{y}{2}\\right)^k\n\\]\n\nTo get \\(x^2\\), we need \\(8 - k = 2\\), thus \\(k = 6\\).\n\nSubstituting \\(k = 6\\) into the expression:\n\\[\n\\binom{8}{6} \\left(\\frac{3}{5}x\\right)^{8-6} \\left(-\\frac{y}{2}\\right)^6 = \\binom{8}{6} \\left(\\frac{3}{5}x\\right)^2 \\left(-\\frac{y}{2}\\right)^6\n\\]\n\nNow, we will compute each part of this expression.\n\n1. Calculate the binomial coefficient \\(\\binom{8}{6}\\).\n2. Compute \\(\\left(\\frac{3}{5}\\right)^2\\).\n3. Compute \\(\\left(-\\frac{y}{2}\\right)^6\\).\n4. Combine everything together to get the coefficient of \\(x^2y^6\\).\n\nLet's compute these in Python.\n```python\nfrom math import comb\n\n# Given values\nn = 8\nk = 6\n\n# Calculate the binomial coefficient\nbinom_coeff = comb(n, k)\n\n# Compute (3/5)^2\na_term = (3/5)**2\n\n# Compute (-1/2)^6\nb_term = (-1/2)**6\n\n# Combine terms to get the coefficient of x^2y^6\ncoefficient = binom_coeff * a_term * b_term\nprint(coefficient)\n```\n```output\n0.1575\n```\nThe coefficient of \\(x^2y^6\\) in the expansion of \\(\\left(\\frac{3}{5}x - \\frac{y}{2}\\right)^8\\) is \\(0.1575\\). To express this as a common fraction, we recognize that:\n\n\\[ 0.1575 = \\frac{1575}{10000} = \\frac{63}{400} \\]\n\nThus, the coefficient can be expressed as:\n\n\\[\n\\boxed{\\frac{63}{400}}\n\\]",
        "role": "assistant"
    }
]
"""

# NOTE: The chat_template is in Jinjia template format, which use {{ }} to denote the variables, {% for %} to iterate over the list, {% if %} to check the condition, {% endif %} to end the condition, {% endfor %} to end the loop.
DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


def get_tokenizer(
    model_args: ModelConfig, training_args: SFTConfig | GRPOConfig, auto_set_chat_template: bool = True
) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    # If user provides a chat_template, use it
    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template
    # If user does not provide a chat_template, and the model does not have a default chat_template, use the Given DEFAULT_CHAT_TEMPLATE
    elif auto_set_chat_template and tokenizer.get_chat_template() is None:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    # Otherwise, use the model's built-in chat_template
    print('----------tokenizer template----------')
    print(tokenizer.chat_template)
    print('----------tokenizer template----------')
    return tokenizer
