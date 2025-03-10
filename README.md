## Env Setup
```
# Source uv env
source OPENR1/bin/activate

# Login wandb
wandb login

# Check GPU usage
watch 'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits'
```

## GRPO training
```
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file mochi_r1/recipes/accelerate_configs/zero3.yaml src/open_r1/grpo.py --config mochi_r1/recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo.yaml
```

## SFT training
```
# Train + Eval
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file mochi_r1/zero3.yaml src/open_r1/sft.py --config mochi_r1/recipes/Qwen2.5-1.5B-Instruct/sft/config_demo_all.yaml

# Eval
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file mochi_r1/zero3.yaml src/open_r1/sft.py --config mochi_r1/recipes/Qwen2.5-1.5B-Instruct/sft/config_demo_all_eval.yaml
```

## Generation
```
python mochi_r1/generate_vllm.py \
  --hf-dataset "/home/jovyan/liumochi/data/AI-MO/NuminaMath-TIR/data" \
  --hf-dataset-split "train" \
  --prompt-column "problem" \
  --prompt-template "You will be given a problem. Please reason step by step, and put your final answer within \\boxed{}: {{ instruction }}" \
  --model "/home/jovyan/liumochi/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --temperature 0.6 \
  --top-p 0.9 \
  --max-new-tokens 8192 \
  --max-model-len 8192 \
  --num-generations 2 \
  --input-batch-size 64 \
  --cuda-devices "1" \
  --tensor-parallel-size 1 \
  --hf-save-dir "/home/jovyan/liumochi/open-r1/data_distilled"
```

## Inference
```
# Use "model_infer.py" in another repo "llm-utils"
```

## Evaluation
```
# https://github.com/huggingface/lighteval/blob/main/src/lighteval/models/vllm/vllm_model.py

# MODEL_NAME=Qwen2.5-7B-Open-R1-Distill
# MODEL_NAME=Distill-Qwen2.5-7B-Math-Code
# MODEL_NAME=Distill-Qwen2.5-7B-Puzzle-Science
# MODEL=/home/jovyan/liumochi/open-r1/data_out/$MODEL_NAME

MODEL_NAME=DeepSeek-R1-Distill-Qwen-7B
MODEL=/home/jovyan/liumochi/model/deepseek-ai/$MODEL_NAME

MODEL_NAME=Qwen2.5-7B-Instruct
MODEL=/home/jovyan/liumochi/model/Qwen/$MODEL_NAME

MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilisation=0.9"
OUTPUT_DIR=/home/jovyan/liumochi/open-r1/evals/$MODEL_NAME
# TASK=math_500
TASK=gpqa:diamond
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
  --custom-tasks mochi_r1/evaluate.py \
  --use-chat-template \
  --output-dir $OUTPUT_DIR
```