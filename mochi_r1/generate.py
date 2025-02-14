from datasets import load_dataset
from distilabel.models import vLLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration


prompt_template = """\
You will be given a problem. Please reason step by step, and put your final answer within \boxed{}:
{{ instruction }}"""

data_files = {
    "train": "/home/jovyan/liumochi/open-r1/data/AI-MO/NuminaMath-TIR/data/train-00000-of-00001.parquet",
    "test": "/home/jovyan/liumochi/open-r1/data/AI-MO/NuminaMath-TIR/data/test-00000-of-00001.parquet"
}
dataset = load_dataset("parquet", data_files=data_files)
train_dataset = dataset["train"].select(range(2))
# dataset = load_dataset("AI-MO/NuminaMath-TIR", split="train").select(range(10))

model_id = "/home/jovyan/liumochi/open-r1/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

cuda_devices = [1]


with Pipeline(
    name="distill-qwen-7b-r1",
    description="A pipeline to generate data from a distilled r1 model",
) as pipeline:

    llm = vLLM(
        model=model_id,
        tokenizer=model_id,
        cuda_devices=cuda_devices,
        extra_kwargs={
            "tensor_parallel_size": 1,
            "max_model_len": 8192,
        },
        generation_kwargs={
            "temperature": 0.6,
            "max_new_tokens": 8192,
        },
    )
    prompt_column = "problem"
    text_generation = TextGeneration(
        llm=llm, 
        template=prompt_template,
        num_generations=2,
        input_mappings={"instruction": prompt_column} if prompt_column is not None else {}
    )


if __name__ == "__main__":
    # Refer to: https://distilabel.argilla.io/1.5.2/api/distiset/#distilabel.distiset.Distiset
    dir_to_save = '/home/jovyan/liumochi/open-r1/data_distilled'
    distiset = pipeline.run(dataset=train_dataset)
    # distiset.push_to_hub(repo_id="username/numina-deepseek-r1-qwen-7b")
    distiset.save_to_disk(dir_to_save)
    '''
    Distiset({
        default: DatasetDict({
            train: Dataset({
                features: ['problem', 'solution', 'messages', 'generation', 'distilabel_metadata', 'model_name'],
                num_rows: 4
            })
        })
    })
    '''
    # Refer to: https://huggingface.co/docs/datasets/v2.21.0/en/process#select-and-filter
    print(distiset['default']['train']['generation'][2])