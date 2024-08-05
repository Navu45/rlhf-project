import inspect
import math
import os
import time
import typing
import warnings
from contextlib import nullcontext
from typing import Callable, List, Optional, Union

import datasets
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, gather_object, is_deepspeed_available
from datasets import Dataset
from huggingface_hub import whoami
from packaging import version
from torch.optim import AdamW
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)

from trl.core import (
    WANDB_PADDING,
    PPODecorators,
    clip_by_value,
    convert_to_scalar,
    entropy_from_logits,
    flatten_dict,
    logprobs_from_logits,
    masked_mean,
    masked_var,
    masked_whiten,
    set_seed,
    stack_dicts,
    stats_to_np,
)
from trl.import_utils import is_npu_available, is_torch_greater_2_0, is_xpu_available
from trl.models import (
    SUPPORTED_ARCHITECTURES,
    PreTrainedModelWrapper,
    create_reference_model,
    unwrap_model_for_generation,
)
from trl.trainer import (
    AdaptiveKLController,
    BaseTrainer,
    FixedKLController,
    PPOConfig,
    RunningMoments,
)

from peft import PeftModel

if is_deepspeed_available():
    import deepspeed


MODEL_CARD_TEMPLATE = """---
license: apache-2.0
tags:
- trl
- ppo
- transformers
- reinforcement-learning
---

# {model_name}

This is a [TRL language model](https://github.com/huggingface/trl) that has been fine-tuned with reinforcement learning to
 guide the model outputs according to a value, function, or human feedback. The model can be used for text generation.

## Usage

To use this model for inference, first install the TRL library:

```bash
python -m pip install trl
```

You can then generate text as follows:

```python
from transformers import pipeline

generator = pipeline("text-generation", model="{model_id}")
outputs = generator("Hello, my llama is cute")
```

If you want to use the model for training or to obtain the outputs from the value head, load the model as follows:

```python
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

tokenizer = AutoTokenizer.from_pretrained("{model_id}")
model = AutoModelForCausalLMWithValueHead.from_pretrained("{model_id}")

inputs = tokenizer("Hello, my llama is cute", return_tensors="pt")
outputs = model(**inputs, labels=inputs["input_ids"])
```
"""


class REINFORCETrainer(BaseTrainer):
    _tag_names = ["trl", "ppo"]

    def __init__(
        self,
        config: Optional[PPOConfig] = None,
        model: Optional[PreTrainedModelWrapper] = None,
        peft_model: Optional[PeftModel] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        dataset: Optional[Union[torch.utils.data.Dataset, Dataset]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        data_collator: Optional[typing.Callable] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        training_data_collator: Optional[typing.Callable] = None,
        adapters: Optional[List[str]] = ["merge", "ema"],
    ):
        """
        Initialize REINFORCETrainer.

        Args:
            config (`PPOConfig`):
                Configuration object for PPOTrainer. Check the documentation of `PPOConfig` for more details.
            model (`PreTrainedModelWrapper`):
                Hugging Face transformer model with a value head.
            tokenizer (`transformers.PreTrainedTokenizerBase`):
                Hugging Face tokenizer
            dataset (Optional[Union[`torch.utils.data.Dataset`, `datasets.Dataset`]]):
                PyTorch dataset or Hugging Face dataset. If a Hugging Face dataset is passed, the dataset
                will be preprocessed by removing the columns that are not used by the model. If none is passed,
                a warning will be raised in a multi-GPU setting.
            optimizer (`Optional[torch.optim.Optimizer]`):
                Optimizer used for training. If `None`, the `AdamW` is used as default.
            data_collator (Optional[function]):
                Data collator function that is going to be used for `prepare_dataloader` method. Note this collator
                is different from the one we use for training. Pass a valid `training_data_collator` instead.
            num_shared_layers (Optional[int]):
                Number of shared layers between the model and the reference model. If `None`, all layers are shared.
                used only if `ref_model` is `None`.
            lr_scheduler (`Optional[torch.optim.lr_scheduler]`):
                Learning rate scheduler used for training.
            training_data_collator (Optional[function]):
                Custom data collator used for training.
        """
        super().__init__(config)
        # initial seed for reproducible experiments
        set_seed(config.seed)

        # Step 0: check positional arguments validity
        if not isinstance(config, PPOConfig):
            raise ValueError(f"config must be a PPOConfig, got {type(config)}")
        if not isinstance(tokenizer, (PreTrainedTokenizerBase)):
            raise ValueError(
                f"tokenizer must be a PreTrainedTokenizerBase like a PreTrainedTokenizer or a PreTrainedTokenizerFast, got {type(tokenizer)}"
            )

        # Step 1: Initialize Accelerator
        self.accelerator = Accelerator(
            log_with=config.log_with,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            project_config=ProjectConfiguration(**config.project_kwargs),
            **config.accelerator_kwargs,
        )

        # Step 1.1 Runtime variables filled by the accelerator
        config.world_size = self.accelerator.num_processes
        config.global_backward_batch_size = (
            config.backward_batch_size * config.world_size
        )
        config.global_batch_size = config.batch_size * config.world_size

        self.model = model
        self.peft_model = peft_model
        self.model_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.is_encoder_decoder = hasattr(self.model, "is_encoder_decoder")
        self.is_peft_model = getattr(self.model, "is_peft_model", False)
        config.is_encoder_decoder = self.is_encoder_decoder
        config.is_peft_model = self.is_peft_model

        is_using_tensorboard = (
            config.log_with is not None and config.log_with == "tensorboard"
        )
        self.accelerator.init_trackers(
            config.tracker_project_name,
            config=(
                dict(trl_ppo_trainer_config=config.to_dict())
                if not is_using_tensorboard
                else config.to_dict()
            ),
            init_kwargs=config.tracker_kwargs,
        )
        self.is_using_text_environment = getattr(config, "use_text_environment", False)

        self.optional_peft_ctx = (
            self.accelerator.unwrap_model(self.model).pretrained_model.disable_adapter
            if self.is_peft_model
            else nullcontext
        )

        if not (
            isinstance(tokenizer, PreTrainedTokenizer)
            or isinstance(tokenizer, PreTrainedTokenizerFast)
        ):
            raise ValueError(
                "tokenizer must be a transformers.PreTrainedTokenizer or transformers.PreTrainedTokenizerFast"
            )
        self.tokenizer = tokenizer

        if dataset is not None and not (
            isinstance(dataset, torch.utils.data.Dataset)
            or isinstance(dataset, Dataset)
        ):
            raise ValueError(
                "dataset must be a torch.utils.data.Dataset or datasets.Dataset"
            )
        elif dataset is None:
            warnings.warn(
                "No dataset is provided. Make sure to set config.batch_size to the correct value before training.",
                UserWarning,
            )
        self.dataset = dataset
        self._signature_columns = None
        if self.dataset is not None:
            self.dataloader = self.prepare_dataloader(self.dataset, data_collator)
        elif self.dataset is None and self.accelerator.num_processes > 1:
            warnings.warn(
                "No dataset is provided. In a multi-GPU setting, this will lead to an error. You should"
                " prepare your dataloader yourself with `dataloader = ppo_trainer.accelerator.prepare(dataloader)`"
                " and using `torch.utils.data.DataLoader`, or pass a dataset to the `PPOTrainer`. Please "
                " refer to the documentation for more details.",
                UserWarning,
            )
            self.dataloader = None
        else:
            self.dataloader = None

        # Step 3: Initialize optimizer and data collator
        if training_data_collator is None:
            self.data_collator = DataCollatorForLanguageModeling(
                self.tokenizer, mlm=False
            )
        else:
            self.data_collator = training_data_collator
        if optimizer is None:
            self.optimizer = AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config.learning_rate,
            )
        else:
            self.optimizer = optimizer

        self.lr_scheduler = lr_scheduler
        if self.lr_scheduler is not None:
            lr_scheduler_class = (
                torch.optim.lr_scheduler._LRScheduler
                if not is_torch_greater_2_0()
                else torch.optim.lr_scheduler.LRScheduler
            )

            if not isinstance(self.lr_scheduler, lr_scheduler_class):
                raise ValueError(
                    "lr_scheduler must be a torch.optim.lr_scheduler._LRScheduler or torch.optim.lr_scheduler.LRScheduler (for torch >= 2.0)"
                )

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
            else:
                # For backward compatibility with older versions of transformers
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                self.model.pretrained_model.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad
                )

        self.base_adapter, self.anchor_adapter = adapters
        self.current_adapter = None

        (
            self.model,
            self.optimizer,
            self.data_collator,
            self.dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.data_collator,
            self.dataloader,
            self.lr_scheduler,
        )

        # In a distributed setup, only logging needs to be performed on the main process
        # check: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
        # or: https://discuss.pytorch.org/t/use-distributed-data-parallel-correctly/82500/11
        self.is_distributed = self.accelerator.num_processes > 1

        # init the current step
        self.current_step = 0

        # init variables for pushing model to hub
        if config.push_to_hub_if_best_kwargs:
            if "repo_id" not in config.push_to_hub_if_best_kwargs:
                raise ValueError(
                    "You have to specify repo_id in order to push the model to the hub!"
                )
            self.push_to_hub_kwargs = config.push_to_hub_if_best_kwargs
            self.compare_step = 0
            self.highest_reward = torch.tensor(-float("inf"))

        # post process for PP
        if not getattr(self.model, "is_sequential_parallel", False):
            self.current_device = self.accelerator.device
        else:
            if is_xpu_available():
                self.current_device = torch.device("xpu:0")
            elif is_npu_available():
                self.current_device = torch.device("npu:0")
            else:
                self.current_device = torch.device("cuda:0")

        PPODecorators.optimize_device_cache = self.config.optimize_device_cache

        self.running = RunningMoments(self.accelerator)

    def _filter_kwargs(self, kwargs, target_func):
        """
        filter the keyword arguments that are supported by the target function.

        Args:
            kwargs (dict):
                Keyword arguments
            target_func (function):
                Target function
        """
        return {
            k: v
            for k, v in kwargs.items()
            if k in inspect.signature(target_func).parameters.keys()
        }

    def prepare_dataloader(
        self, dataset: Union[torch.utils.data.Dataset, Dataset], data_collator=None
    ):
        """
        Prepare the dataloader for training.

        Args:
            dataset (Union[`torch.utils.data.Dataset`, `datasets.Dataset`]):
                PyTorch dataset or Hugging Face dataset. If a Hugging Face dataset is passed, the dataset
                will be preprocessed by removing the columns that are not used by the model.
            data_collator (Optional[function]):
                Data collator function.

        Returns:
            `torch.utils.data.DataLoader`: PyTorch dataloader
        """
        if isinstance(dataset, Dataset):
            dataset = self._remove_unused_columns(dataset)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            collate_fn=data_collator,
            shuffle=True,
            drop_last=True,
        )
        return dataloader

    # Adapted from transformers.Trainer._set_signature_columns_if_needed
    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # label => sentiment | we need query and response for logging purpose
            self._signature_columns += ["label", "query", "response"]

    # Adapted from transformers.Trainer._remove_unused_columns
    def _remove_unused_columns(self, dataset: "Dataset"):
        if not self.config.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        ignored_columns = list(set(dataset.column_names) - set(signature_columns))

        columns = [k for k in signature_columns if k in dataset.column_names]

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"],
                columns=columns,
                format_kwargs=dataset.format["format_kwargs"],
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)

    def generate(
        self,
        query_tensor: Union[torch.Tensor, List[torch.Tensor]],
        length_sampler: Optional[Callable] = None,
        batch_size: int = 4,
        return_prompt: bool = True,
        generate_ref_response: bool = False,
        **generation_kwargs,
    ):
        """
        Generate response with the model given the query tensor.
        call the `generate` method of the model.

        Args:
            query_tensor (`torch.LongTensor`):
                A tensor of shape (`seq_len`) containing query tokens or a list of tensors of shape (`seq_len`).
            length_sampler (`Callable`, *optional*):
                Callable that returns the number of newly generated tokens.
            batch_size (`int`, *optional):
                Batch size used for generation, defaults to `4`.
            return_prompt (`bool`, *optional*):
                If set to `False` the prompt is not returned but only the newly generated tokens, defaults to `True`.
            generate_ref_response (`bool`, *optional*):
                If set to `True` the reference response is also generated, defaults to `False`.
            generation_kwargs (dict[str, Any]):
                Keyword arguments for generation.

        Returns:
            `torch.LongTensor`: A tensor of shape (`batch_size`, `gen_len`) containing response tokens.
        """

        if isinstance(query_tensor, List):
            self.peft_model.set_adapter(self.peft_model.current_adapter)
            response = self._generate_batched(
                self.model,
                query_tensor,
                length_sampler=length_sampler,
                batch_size=batch_size,
                return_prompt=return_prompt,
                **generation_kwargs,
            )
            if generate_ref_response:
                self.peft_model.set_adapter(self.peft_model.base_adapter)
                ref_response = self._generate_batched(
                    self.model,
                    query_tensor,
                    length_sampler=length_sampler,
                    batch_size=batch_size,
                    return_prompt=return_prompt,
                    **generation_kwargs,
                )
        else:
            if len(query_tensor.shape) == 2:
                raise ValueError(
                    "query_tensor must be a tensor of shape (`seq_len`) or a list of tensors of shape (`seq_len`)"
                )

            if length_sampler is not None:
                generation_kwargs["max_new_tokens"] = length_sampler()

            self.peft_model.set_adapter(self.peft_model.current_adapter)
            with unwrap_model_for_generation(
                self.model, self.accelerator
            ) as unwrapped_model:
                response = unwrapped_model.generate(
                    input_ids=query_tensor.unsqueeze(dim=0), **generation_kwargs
                )

            if generate_ref_response:
                self.peft_model.set_adapter(self.peft_model.base_adapter)
                with unwrap_model_for_generation(
                    self.peft_model, self.accelerator, is_peft_model=self.is_peft_model
                ) as unwrapped_model:
                    ref_response = unwrapped_model.generate(
                        input_ids=query_tensor.unsqueeze(dim=0), **generation_kwargs
                    )

            if not return_prompt and not self.is_encoder_decoder:
                response = response[:, query_tensor.shape[0] :]
                if generate_ref_response:
                    ref_response = ref_response[:, query_tensor.shape[0] :]

        if generate_ref_response:
            return response, ref_response
        return response

    def _generate_batched(
        self,
        model: PreTrainedModelWrapper,
        query_tensors: List[torch.Tensor],
        length_sampler: Optional[Callable] = None,
        batch_size: int = 4,
        return_prompt: bool = True,
        pad_to_multiple_of: Optional[int] = None,
        remove_padding: bool = True,
        **generation_kwargs,
    ):
        outputs = []

        padding_side_default = self.tokenizer.padding_side
        if not self.is_encoder_decoder:
            self.tokenizer.padding_side = "left"

        # in case we have fewer examples than bs
        batch_size = min(len(query_tensors), batch_size)

        for i in range(0, len(query_tensors), batch_size):
            if length_sampler is not None:
                generation_kwargs["max_new_tokens"] = length_sampler()

            # prevent overflow if query tensors are not even multiple of bs
            end_index = min(len(query_tensors), i + batch_size)

            batch = query_tensors[i:end_index]
            batch_mask = [torch.ones_like(element) for element in batch]
            inputs = {"input_ids": batch, "attention_mask": batch_mask}

            padded_inputs = self.tokenizer.pad(
                inputs,
                padding=True,
                max_length=None,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors="pt",
            ).to(self.current_device)

            with unwrap_model_for_generation(
                model,
                self.accelerator,
            ) as unwrapped_model:
                generations = unwrapped_model.generate(
                    **padded_inputs, **generation_kwargs
                )

            for generation, mask in zip(generations, padded_inputs["attention_mask"]):
                if not self.is_encoder_decoder:
                    output = generation[(1 - mask).sum() :]  # remove padding
                else:
                    output = generation

                if not return_prompt and not self.is_encoder_decoder:
                    output = output[(mask).sum() :]  # remove prompt

                if remove_padding and self.tokenizer.eos_token_id in output:
                    pad_mask = output == self.tokenizer.eos_token_id
                    pad_start = torch.nonzero(pad_mask, as_tuple=False)[0, 0].item()
                    output = output[: pad_start + 1]  # keep the eos token at the end

                outputs.append(output)

        self.tokenizer.padding_side = padding_side_default
        return outputs

    def _step_safety_checker(
        self,
        batch_size: int,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        rewards: List[torch.FloatTensor],
        masks: Optional[List[torch.LongTensor]] = None,
    ):
        """
        Check if the input data is valid for training.

        Args:
            batch_size (int):
                Batch size from the config file.
            queries (List[`torch.LongTensor`]):
                List of tensors containing the encoded queries of shape (`query_length`)
            responses (List[`torch.LongTensor`]):
                List of tensors containing the encoded responses of shape (`response_length`)
            rewards (List[`torch.FloatTensor`]):
                List of tensors containing the rewards.
            masks (List[`torch.LongTensor`], *optional*):
                list of optional tensors containing the masks of shape (`response_length`)
        Returns:
            `tuple`: The input processed data.
        """
        for name, tensor_list in zip(
            ["queries", "responses", "rewards"], [queries, responses, rewards]
        ):
            if not isinstance(tensor_list, list):
                raise ValueError(
                    f"{name} must be a list of tensors - got {type(tensor_list)}"
                )
            if not isinstance(tensor_list[0], torch.Tensor):
                raise ValueError(
                    f"Elements in {name} must be tensors - got {type(tensor_list[0])}"
                )
            if batch_size is not None and len(tensor_list) != batch_size:
                raise ValueError(
                    f"Batch size ({batch_size}) does not match number of examples - but got {len(tensor_list)} for: {name}"
                )

        # add queries, rewards and responses on the correct device
        queries = [tensor.to(self.current_device) for tensor in queries]
        responses = [tensor.to(self.current_device) for tensor in responses]
        rewards = [tensor.to(self.current_device) for tensor in rewards]
        masks = (
            [tensor.to(self.current_device) for tensor in masks]
            if masks is not None
            else None
        )

        # squeeze rewards if needed
        for i, reward in enumerate(rewards):
            if reward.dim() > 1:
                raise ValueError(
                    f"rewards must be 1-dimensional - got {reward.dim()} for {reward}"
                )
            elif reward.dim() == 1:
                rewards[i] = reward.squeeze()

        return queries, responses, rewards, masks

    @PPODecorators.empty_device_cache()
    def step(
        self,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        rewards: List[torch.FloatTensor],
        response_masks: Optional[List[torch.LongTensor]] = None,
    ):
        """
        Run a PPO optimisation step given a list of queries, model responses, and rewards.

        Args:
            queries (List[`torch.LongTensor`]):
                List of tensors containing the encoded queries of shape (`query_length`)
            responses (List[`torch.LongTensor`]):
                List of tensors containing the encoded responses of shape (`response_length`)
            rewards (List[`torch.FloatTensor`]):
                List of tensors containing the rewards.
            response_masks (List[`torch.FloatTensor`], *optional*)):
                List of tensors containing masks of the response tokens.

        Returns:
            `dict[str, Any]`: A summary of the training statistics
        """
        bs = self.config.batch_size

        queries, responses, rewards, response_masks = self._step_safety_checker(
            bs, queries, responses, rewards, response_masks
        )
        rewards = torch.tensor(rewards, device=self.current_device)

        # if we want to push best model to the hub
        if hasattr(self, "highest_reward"):
            if self.compare_step % self.config.compare_steps == 0:
                curr_mean_reward = rewards.mean()
                # if the best reward ever seen
                if curr_mean_reward > self.highest_reward:
                    self.highest_reward = curr_mean_reward
                    # push model to hub
                    self.push_to_hub(**self.push_to_hub_kwargs)
            self.compare_step += 1

        timing = dict()
        t0 = time.time()

        t = time.time()

        model_inputs = self.prepare_model_inputs(queries, responses)

        if self.is_distributed:
            pad_first = self.tokenizer.padding_side == "left"

            model_inputs["input_ids"] = self.accelerator.pad_across_processes(
                model_inputs["input_ids"],
                dim=1,
                pad_index=self.tokenizer.pad_token_id,
                pad_first=pad_first,
            )
            model_inputs["attention_mask"] = self.accelerator.pad_across_processes(
                model_inputs["attention_mask"], dim=1, pad_index=0, pad_first=pad_first
            )
            if self.is_encoder_decoder:
                model_inputs["decoder_input_ids"] = (
                    self.accelerator.pad_across_processes(
                        model_inputs["decoder_input_ids"],
                        dim=1,
                        pad_index=self.tokenizer.pad_token_id,
                        pad_first=pad_first,
                    )
                )
                model_inputs["decoder_attention_mask"] = (
                    self.accelerator.pad_across_processes(
                        model_inputs["decoder_attention_mask"],
                        dim=1,
                        pad_index=0,
                        pad_first=pad_first,
                    )
                )

        model_inputs_names = list(model_inputs.keys())

        with torch.no_grad():
            all_logprobs, _, masks = self.batched_forward_pass(
                self.model,
                queries,
                responses,
                model_inputs,
                response_masks=response_masks,
                return_logits=False,
            )

        timing["time/ppo/forward_pass"] = time.time() - t

        # upcast to float32 to avoid dataset issues
        batch_dict = {
            "queries": queries,
            "responses": responses,
            "logprobs": all_logprobs.to(torch.float32),
            "masks": masks,
            "rewards": rewards,
        }
        batch_dict.update(model_inputs)

        t = time.time()
        all_stats = []
        early_stop = False
        for _ in range(self.config.ppo_epochs):
            if early_stop:
                break
            b_inds = np.random.permutation(bs)
            for backward_batch_start in range(0, bs, self.config.backward_batch_size):
                backward_batch_end = (
                    backward_batch_start + self.config.backward_batch_size
                )
                backward_batch_inds = b_inds[backward_batch_start:backward_batch_end]

                for mini_batch_start in range(
                    0, self.config.backward_batch_size, self.config.mini_batch_size
                ):
                    mini_batch_end = mini_batch_start + self.config.mini_batch_size
                    mini_batch_inds = backward_batch_inds[
                        mini_batch_start:mini_batch_end
                    ]
                    mini_batch_dict = {
                        "logprobs": batch_dict["logprobs"][mini_batch_inds],
                        "masks": batch_dict["masks"][mini_batch_inds],
                        # hacks: the queries and responses are ragged.
                        "queries": [batch_dict["queries"][i] for i in mini_batch_inds],
                        "responses": [
                            batch_dict["responses"][i] for i in mini_batch_inds
                        ],
                        "rewards": batch_dict["rewards"][mini_batch_inds],
                    }
                    for k in model_inputs_names:
                        mini_batch_dict[k] = batch_dict[k][mini_batch_inds]
                    with self.accelerator.accumulate(self.model):
                        model_inputs = {
                            k: mini_batch_dict[k] for k in model_inputs_names
                        }

                        logprobs, logits, _ = self.batched_forward_pass(
                            self.model,
                            mini_batch_dict["queries"],
                            mini_batch_dict["responses"],
                            model_inputs,
                            return_logits=True,
                        )
                        train_stats = self.train_minibatch(
                            logprobs,
                            mini_batch_dict["rewards"],
                            mini_batch_dict["masks"],
                        )
                        all_stats.append(train_stats)

        timing["time/ppo/optimize_step"] = time.time() - t

        t = time.time()
        train_stats = stack_dicts(all_stats)

        stats = self.record_step_stats(
            rewards=rewards,
            logprobs=all_logprobs,
            train_stats=train_stats,
            masks=masks,
            queries=queries,
            responses=responses,
        )
        # Gather/Reduce stats from all processes
        if self.is_distributed:
            stats = self.gather_stats(stats)
        stats = stats_to_np(stats)
        timing["time/ppo/calc_stats"] = time.time() - t
        stats["ppo/learning_rate"] = self.optimizer.param_groups[0]["lr"]

        # Log the total ppo time
        timing["time/ppo/total"] = time.time() - t0
        stats.update(timing)

        # post-process stats for tensorboard and other loggers
        if self.config.log_with != "wandb":
            stats = convert_to_scalar(stats)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return stats

    def _early_stop(self, policykl):
        r"""
        Handles the early stopping logic. If the policy KL is greater than the target KL, then the gradient is zeroed and
        the optimization step is skipped.
        This also handles the multi-gpu case where the policy KL is averaged across all processes.

        Args:
            policy_kl (torch.Tensor):
                the policy KL

        Returns:
            `bool`: whether to early stop or not
        """
        early_stop = False
        if not self.config.early_stopping:
            return early_stop

        if not self.is_distributed and policykl > 1.5 * self.config.target_kl:
            self.optimizer.zero_grad()
            early_stop = True
        elif self.is_distributed:
            import torch.distributed as dist

            # Wait for all processes to finish
            dist.barrier()

            # all gather the policykl
            dist.all_reduce(policykl, dist.ReduceOp.SUM)
            policykl /= self.accelerator.num_processes

            if policykl > 1.5 * self.config.target_kl:
                self.optimizer.zero_grad()
                early_stop = True
        return early_stop

    def gather_stats(self, stats):
        """
        Gather stats from all processes. Useful in the context of distributed training.

        Args:
            stats (dict[str, Any]):
            a dictionary of stats to be gathered. The stats should contain torch tensors.

        Returns:
            `dict[str, Any]`: A dictionary of stats with the tensors gathered.
        """
        import torch.distributed as dist

        # Wait for all processes to finish
        dist.barrier()

        for k, v in stats.items():
            if isinstance(v, torch.Tensor):
                dist.all_reduce(v.to(self.accelerator.device), dist.ReduceOp.SUM)
                v /= self.accelerator.num_processes
            stats[k] = v
        return stats

    def prepare_model_inputs(self, queries: torch.Tensor, responses: torch.Tensor):
        if self.is_encoder_decoder:
            input_data = self.data_collator(
                [
                    {"input_ids": q, "attention_mask": torch.ones_like(q)}
                    for q in queries
                ]
            ).to(self.current_device)

            decoder_inputs = self.data_collator(
                [
                    {"input_ids": r, "attention_mask": torch.ones_like(r)}
                    for r in responses
                ]
            ).to(self.current_device)

            input_data["decoder_input_ids"] = decoder_inputs["input_ids"]
            input_data["decoder_attention_mask"] = decoder_inputs["attention_mask"]
        else:
            input_ids = [torch.cat([q, r]) for q, r in zip(queries, responses)]
            input_data = self.data_collator(
                [
                    {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
                    for ids in input_ids
                ]
            ).to(self.current_device)

        input_data.pop("labels", None)  # we don't want to compute LM losses
        return input_data

    @PPODecorators.empty_device_cache()
    def batched_forward_pass(
        self,
        model: PreTrainedModelWrapper,
        queries: torch.Tensor,
        responses: torch.Tensor,
        model_inputs: dict,
        return_logits: bool = False,
        response_masks: Optional[torch.Tensor] = None,
    ):
        """
        Calculate model outputs in multiple batches.

        Args:
            queries (`torch.LongTensor`):
                List of tensors containing the encoded queries, shape (`batch_size`, `query_length`)
            responses (`torch.LongTensor`):
                List of tensors containing the encoded responses, shape (`batch_size`, `response_length`)
            return_logits (`bool`, *optional*, defaults to `False`):
                Whether to return all_logits. Set to `False` if logits are not needed to reduce memory consumption.
        Returns:
            (tuple):
                - all_logprobs (`torch.FloatTensor`): Log probabilities of the responses,
                    shape (`batch_size`, `response_length`)
                - all_ref_logprobs (`torch.FloatTensor`): Log probabilities of the responses,
                    shape (`batch_size`, `response_length`)
        """
        bs = len(queries)
        fbs = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []

        model.eval()

        for i in range(math.ceil(bs / fbs)):
            input_kwargs = {
                key: value[i * fbs : (i + 1) * fbs]
                for key, value in model_inputs.items()
            }
            query_batch = queries[i * fbs : (i + 1) * fbs]
            response_batch = responses[i * fbs : (i + 1) * fbs]
            if response_masks is not None:
                response_masks_batch = response_masks[i * fbs : (i + 1) * fbs]
            logits, _, _ = model(**input_kwargs)

            if self.is_encoder_decoder:
                input_ids = input_kwargs["decoder_input_ids"]
                attention_mask = input_kwargs["decoder_attention_mask"]
            else:
                input_ids = input_kwargs["input_ids"]
                attention_mask = input_kwargs["attention_mask"]

            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]

            for j in range(len(query_batch)):
                if self.is_encoder_decoder:
                    # Decoder sentence starts always in the index 1 after padding in the Enc-Dec Models
                    start = 1
                    end = attention_mask[j, :].sum() - 1
                else:
                    start = (
                        len(query_batch[j]) - 1
                    )  # logprobs starts from the second query token
                    if attention_mask[j, 0] == 0:  # offset left padding
                        start += attention_mask[j, :].nonzero()[0]
                    end = start + len(response_batch[j])

                masks[j, :start] = 0
                masks[j, end:] = 0
                if response_masks is not None:
                    masks[j, start:end] = masks[j, start:end] * response_masks_batch[j]

            if return_logits:
                all_logits.append(logits)
            else:
                del logits
            all_logprobs.append(logprobs)
            all_masks.append(masks)

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1] if return_logits else None,
            torch.cat(all_masks)[:, :-1],
        )

    @PPODecorators.empty_device_cache()
    def train_minibatch(
        self,
        logprobs: torch.FloatTensor,
        rewards: torch.FloatTensor,
        mask: torch.LongTensor,
    ):
        """
        Train one PPO minibatch

        Args:
            logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape [mini_batch_size, response_length]
            values (`torch.FloatTensor`):
                Values of the value head, shape [mini_batch_size, response_length]
            query (`torch.LongTensor`):
                Encoded queries, shape [mini_batch_size, query_length]
            response (`torch.LongTensor`):
                Encoded responses, shape [mini_batch_size, response_length]
            model_input (`torch.LongTensor`):
                Concatenated queries and responses, shape [mini_batch_size, query_length+response_length]

        Returns:
            train_stats (dict[str, `torch.Tensor`]):
                Dictionary of training statistics
        """
        self.model.train()
        loss, train_stats = self.loss(logprobs, rewards, mask)
        self.accelerator.backward(loss)
        if self.config.max_grad_norm is not None:
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(
                    self.model_params, self.config.max_grad_norm
                )
        self.optimizer.step()
        # we call optimizer.zero_grad() every time and let `accelerator` handle accumulation
        # see https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation#the-finished-code
        self.optimizer.zero_grad()
        return train_stats

    def loss(
        self,
        logprobs: torch.FloatTensor,
        rewards: torch.FloatTensor,
        mask: torch.LongTensor,
    ):
        """
        Calculate policy and value losses for REINFORCE.

        Args:
            logprobs (torch.FloatTensor): Log probabilities of the model, shape (batch_size, response_length)
            rewards (torch.FloatTensor): Rewards from the reward model, shape (batch_size, response_length)
        """
        # Compute the policy gradient loss
        policy_loss = masked_mean(-logprobs * rewards, mask)

        # Calculate additional stats for logging
        entropy = -(logprobs * torch.exp(logprobs)).sum(dim=-1).mean()

        stats = {
            "loss/policy": policy_loss.detach(),
            "policy/entropy": entropy.detach(),
        }

        return policy_loss, stats

    def record_step_stats(self, **data):
        """
        Record training step statistics.


        Args:
            data (`dict`):
                Dictionary of training step data

        Returns:
            stats (`dict`):
                Dictionary of training step statistics
        """
        mask = data.pop("masks")

        mean_entropy = (-data["logprobs"] * mask).sum(axis=-1).mean()

        stats = {
            "objective/logprobs": data["logprobs"],
            # "objective/ref_logprobs": data["ref_logprobs"],
            "objective/entropy": mean_entropy,
        }

        # Log text properties
        query_lens = torch.tensor(
            [len(query) for query in data["queries"]], dtype=torch.float
        )
        response_lens = torch.tensor(
            [len(response) for response in data["responses"]], dtype=torch.float
        )

        stats["tokens/queries_len_mean"] = torch.mean(query_lens).cpu().numpy().item()
        stats["tokens/queries_len_std"] = torch.std(query_lens).cpu().numpy().item()
        stats["tokens/queries_dist"] = query_lens.cpu().numpy()
        stats["tokens/responses_len_mean"] = (
            torch.mean(response_lens).cpu().numpy().item()
        )
        stats["tokens/responses_len_std"] = (
            torch.std(response_lens).cpu().numpy().item()
        )
        stats["tokens/responses_dist"] = response_lens.cpu().numpy()

        for k, v in data["train_stats"].items():
            stats[f"ppo/{k}"] = torch.mean(v, axis=0)
        return stats

    def log_stats(
        self,
        stats: dict,
        batch: dict,
        rewards: List[torch.FloatTensor],
        columns_to_log: typing.Iterable[str] = ("query", "response"),
    ):
        """
        A function that logs all the training stats. Call it at the end of each epoch.

        Args:
            stats (dict[str, Any]):
                A dictionary of training stats.
            batch (dict[str, Any]):
                A dictionary of batch data, this contains the queries and responses.
            rewards (`List[torch.FloatTensor]`):
                A tensor of rewards.
        """

        # all gather stats
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards).to(self.current_device)
        rewards = self.accelerator.gather(rewards).flatten()

        if self.config.log_with == "wandb":
            import wandb

            if any(
                column_to_log not in batch.keys() for column_to_log in columns_to_log
            ):
                raise ValueError(
                    f"Columns to log {columns_to_log} are not present in the batch {batch.keys()}."
                )

            batch_list = [batch[column_to_log] for column_to_log in columns_to_log]
            if self.is_distributed:
                gathered_batch_list = []
                for b in batch_list:
                    flattened = gather_object(b)
                    gathered_batch_list.append(flattened)
                batch_list = gathered_batch_list

        # Log only if we are in the main process
        if self.accelerator.is_main_process:
            logs = {}

            # Log stats
            if "query" not in batch.keys() and "response" not in batch.keys():
                # warn the user that the game logs will not be logged
                warnings.warn(
                    "The game logs will not be logged because the batch does not contain the keys 'query' and "
                    "'response'. "
                )
            elif self.config.log_with == "wandb":
                table_rows = [list(r) for r in zip(*batch_list, rewards.cpu().tolist())]
                logs.update(
                    {
                        "game_log": wandb.Table(
                            columns=[*columns_to_log, "reward"], rows=table_rows
                        )
                    }
                )

            logs.update(stats)

            # manually cast in fp32 for bf16 torch tensors
            for k, v in logs.items():
                if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16:
                    logs[k] = v.float()

            logs["env/reward_mean"] = torch.mean(rewards).cpu().numpy().item()
            logs["env/reward_std"] = torch.std(rewards).cpu().numpy().item()
            logs["env/reward_dist"] = rewards.cpu().numpy()

            if self.config.log_with == "tensorboard":
                # update the current step
                self.current_step += 1

            self.accelerator.log(
                logs,
                step=(
                    self.current_step if self.config.log_with == "tensorboard" else None
                ),
            )

    def create_model_card(
        self, path: str, model_name: Optional[str] = "TRL Model"
    ) -> None:
        """Creates and saves a model card for a TRL model.

        Args:
            path (`str`): The path to save the model card to.
            model_name (`str`, *optional*): The name of the model, defaults to `TRL Model`.
        """
        try:
            user = whoami()["name"]
        # handle the offline case
        except Exception:
            warnings.warn(
                "Cannot retrieve user information assuming you are running in offline mode."
            )
            return

        if not os.path.exists(path):
            os.makedirs(path)

        model_card_content = MODEL_CARD_TEMPLATE.format(
            model_name=model_name, model_id=f"{user}/{path}"
        )
        with open(os.path.join(path, "README.md"), "w", encoding="utf-8") as f:
            f.write(model_card_content)

    def _save_pretrained(self, save_directory: str) -> None:
        self.accelerator.unwrap_model(self.model).save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        self.create_model_card(save_directory)

    def _show_tokens(self, tokens, masks):
        from rich import print
        from rich.text import Text

        text = Text()

        for _i, (token, mask) in enumerate(zip(tokens, masks)):
            if mask == 1:
                text.append(
                    self.tokenizer.decode(token.item()), style="black on deep_sky_blue1"
                )
                text.append(" ")
            else:
                text.append(self.tokenizer.decode(token.item()), style="black on cyan3")
                text.append(" ")
        print(text)

    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepspeed_plugin.deepspeed_config
        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if (
                    hidden_size is not None
                    and config_kwargs["zero_optimization"]["stage"] == 3
                ):
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size
                            * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10
                            * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9
                            * hidden_size
                            * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model
