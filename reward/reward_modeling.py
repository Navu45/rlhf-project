import torch
from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    DefaultDataCollator,
)

from trl import (
    ModelConfig,
    RewardConfig,
    RewardTrainer,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from torch.utils.data import Dataset

from datasets import load_dataset


class IMDBPairsDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        split="train",
        num_proc=4,
        max_length=512,
        shuffle=False,
        seed=42,
    ):

        def tokenization(example):
            return tokenizer(example["text"], max_length=max_length)

        dataset = load_dataset("imdb", split=split).map(tokenization, batched=True, num_proc=num_proc)
        self.positive_reviews = dataset.filter(
            lambda x: x["label"] == 1, num_proc=num_proc
        )
        self.negative_reviews = dataset.filter(
            lambda x: x["label"] == 0, num_proc=num_proc
        )
        if shuffle:
            self.positive_reviews = self.positive_reviews.shuffle(seed=seed)
            self.negative_reviews = self.negative_reviews.shuffle(seed=seed)

    def __len__(self):
        return len(self.positive_reviews) * len(self.negative_reviews)

    def __getitem__(self, idx):
        pos, neg = (
            self.positive_reviews[idx // len(self.positive_reviews)],
            self.negative_reviews[idx % len(self.negative_reviews)],
        )
        return {
            "input_ids_chosen": pos["input_ids"],
            "attention_mask_chosen": pos["attention_mask"],
            "input_ids_rejected": neg["input_ids"],
            "attention_mask_rejected": neg["attention_mask"],
        }


tqdm.pandas()


if __name__ == "__main__":
    parser = HfArgumentParser((RewardConfig, ModelConfig))
    config, model_config = parser.parse_args_into_dataclasses()
    config.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        use_fast=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path,
        num_labels=1,
        trust_remote_code=model_config.trust_remote_code,
        **model_kwargs
    )

    ################
    # Dataset
    ################
    train_dataset = IMDBPairsDataset(
        tokenizer, split="train", max_length=config.max_length, shuffle=True
    )
    eval_dataset = IMDBPairsDataset(
        tokenizer, split="test[:80]+test[-80:]", max_length=config.max_length
    )

    ################
    # Training
    ################
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_config),
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    print(metrics)
