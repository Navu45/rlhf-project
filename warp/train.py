from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline

from trl import (
    AutoModelForCausalLMWithValueHead,
    AutoModelForSeq2SeqLMWithValueHead,
    PPOConfig,
    set_seed,
)
from trl.core import LengthSampler
from trl.import_utils import is_npu_available, is_xpu_available
from warp_model import WarpModel
from warp_trainer import WarpTrainer

import os
import shutil


tqdm.pandas()


@dataclass
class ScriptArguments:
    use_seq2seq: bool = field(
        default=False, metadata={"help": "whether to use seq2seq"}
    )
    trust_remote_code: bool = field(
        default=False, metadata={"help": "Enable `trust_remote_code`"}
    )

    # LoraConfig
    use_peft: bool = field(default=False, metadata={"help": "whether to use peft"})
    lora_alpha: Optional[float] = field(
        default=16, metadata={"help": "the lora alpha parameter"}
    )
    ema_coef: Optional[int] = field(
        default=16, metadata={"help": "the rate coefficient for EMA"}
    )
    slerp_coef: Optional[int] = field(
        default=16, metadata={"help": "the rate coefficient for SLERP"}
    )
    iterations: Optional[int] = field(
        default=2, metadata={"help": "the num of iterations for WARP training"}
    )
    lora_r: Optional[int] = field(default=16, metadata={"help": "the lora r parameter"})
    output_dir: Optional[str] = field(
        default="data/warp", metadata={"help": "directory to save model"}
    )


parser = HfArgumentParser((ScriptArguments, PPOConfig))
args, config = parser.parse_args_into_dataclasses()

trl_model_class = (
    AutoModelForCausalLMWithValueHead
    if not args.use_seq2seq
    else AutoModelForSeq2SeqLMWithValueHead
)


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(
    config, query_dataset, input_min_text_length=8, input_max_text_length=15
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        query_dataset (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    ds = load_dataset(query_dataset, split="train")
    ds = ds.rename_columns({"text": "review"})

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(config, config.query_dataset)


def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# Now let's build the model, the reference model, and the tokenizer.
if not args.use_peft:
    ref_model = trl_model_class.from_pretrained(
        config.model_name, trust_remote_code=args.trust_remote_code
    )
    device_map = None
    peft_config = None
else:
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        bias="none",
        task_type="CAUSAL_LM",
    )
    ref_model = None
    # Copy the model to each device
    device_map = {"": Accelerator().local_process_index}

model = trl_model_class.from_pretrained(
    config.model_name,
    trust_remote_code=args.trust_remote_code,
    device_map=device_map,
    peft_config=peft_config,
)

tokenizer = AutoTokenizer.from_pretrained(config.model_name)

# Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
tokenizer.pad_token_id = tokenizer.eos_token_id


def prepare_trainer(model):
    peft_model = WarpModel(
        model, peft_config, "merge", anchor_adapter="ema", train_adapters=["rl1", "rl2"]
    )
    # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
    trainer = WarpTrainer(
        config,
        model,
        peft_model=peft_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
    )
    return peft_model, trainer


peft_model, trainer = prepare_trainer(model)

# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
device = trainer.accelerator.device
if trainer.accelerator.num_processes == 1:
    if is_xpu_available():
        device = "xpu:0"
    elif is_npu_available():
        device = "npu:0"
    else:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
ds_plugin = trainer.accelerator.state.deepspeed_plugin
task, model_name = config.reward_model.split(":")
if ds_plugin is not None and ds_plugin.is_zero3_init_enabled():
    with ds_plugin.zero3_init_context_manager(enable=False):
        sentiment_pipe = pipeline(task, model=model_name, device=device)
else:
    sentiment_pipe = pipeline(task, model=model_name, device=device)

# Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
if sentiment_pipe.tokenizer.pad_token_id is None:
    sentiment_pipe.tokenizer.pad_token_id = tokenizer.pad_token_id

if sentiment_pipe.model.config.pad_token_id is None:
    sentiment_pipe.model.config.pad_token_id = tokenizer.pad_token_id

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 80,
}
# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {
    "top_k": None,
    "function_to_apply": "none",
    "batch_size": config.batch_size,
}


for i in range(args.iterations):
    for current_model in peft_model.train_adapters:
        peft_model.current_adapter = current_model
        for step, batch in tqdm(enumerate(trainer.dataloader), total=config.steps):
            if step == config.steps:
                break

            query_tensors = batch["input_ids"]

            # Get response from gpt2
            response_tensors, ref_response_tensors = trainer.generate(
                query_tensors,
                return_prompt=False,
                generate_ref_response=True,
                **generation_kwargs,
            )
            batch["response"] = tokenizer.batch_decode(response_tensors)
            batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)

            # Compute sentiment score
            texts = [q + r for q, r in zip(batch["query"], batch["response"])]
            pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
            rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
            ref_texts = [q + r for q, r in zip(batch["query"], batch["ref_response"])]
            ref_pipe_outputs = sentiment_pipe(ref_texts, **sent_kwargs)
            ref_rewards = [
                torch.tensor(output[1]["score"]) for output in ref_pipe_outputs
            ]
            batch["ref_rewards"] = ref_rewards

            # Run PPO step
            stats = trainer.step(query_tensors, response_tensors, rewards)
            trainer.log_stats(
                stats,
                batch,
                rewards,
                columns_to_log=["query", "response", "ref_response", "ref_rewards"],
            )
            peft_model.weight_averaging_step(
                "ema", [current_model], peft_model.anchor_adapter, args.ema_coef
            )

    peft_model.set_adapter(peft_model.base_adapter)
    peft_model.weight_averaging_step(
        "slerp", peft_model.train_adapters, peft_model.base_adapter, args.slerp_coef
    )
    merged_model = peft_model.merge_and_unload(
        progressbar=True, safe_merge=True, adapter_names=[peft_model.base_adapter]
    )
    iter_output_dir = f"{args.output_dir}/iter-{i}/"

    if not os.path.isdir(iter_output_dir):
        os.makedirs(iter_output_dir)

    merged_model.save_pretrained(iter_output_dir)
    del model, trainer, peft_model
    peft_model, trainer = prepare_trainer(merged_model)
