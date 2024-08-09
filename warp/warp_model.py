import torch

from typing import List
from peft import PeftModelForCausalLM, PeftConfig
from dataclasses import replace
from peft.tuners.lora import LoraLayer
from peft.utils import _freeze_adapter, _get_submodules


def slerp(
    task_tensors: List[torch.Tensor],
    base_tensor: torch.Tensor,
    weights: List[float],
) -> torch.Tensor:
    """
    Merge the task tensors using SLERP.

    Args:
        task_tensors (`List[torch.Tensor]`): The task tensors to merge.
        init_tensor (`torch.Tensor`): The initial tensor.
        weight (`float`): The interpolation coefficient (lambda).

    Returns:
        `torch.Tensor`: The merged tensor.
    """
    assert (
        len(task_tensors) == 2
    ), "This implementation of SLERP takes only 2 adapters to merge!"

    delta_1, delta_2 = [task_tensor - base_tensor for task_tensor in task_tensors]
    
    assert not torch.isnan(delta_1).any(), 'No Nan!!!'
    assert not torch.isnan(delta_2).any(), 'No Nan!!!'

    weight_1, weight_2 = weights

    cos_omega = torch.dot(delta_1.flatten(), delta_2.flatten()) / (
        torch.norm(delta_1) * torch.norm(delta_2)
    )
    omega = torch.acos(cos_omega)
    delta_1 *= torch.sin((1 - weight_1) * omega) / torch.sin(omega)
    delta_2 *= torch.sin(weight_2 * omega) / torch.sin(omega)
    mixed_task_tensors = base_tensor + delta_1 + delta_2
    
    assert not torch.isnan(mixed_task_tensors).any(), 'No Nan!!!'
    return mixed_task_tensors


def ema(task_tensors: torch.Tensor, anchor_tensor: torch.Tensor, weight: float):
    task_tensors = torch.stack(task_tensors, dim=0)
    mixed_task_tensor = (1 - weight) * anchor_tensor + weight * task_tensors

    assert not torch.isnan(mixed_task_tensor).any(), 'No Nan!!!'
    return mixed_task_tensor


class WarpModel(PeftModelForCausalLM):
    def __init__(
        self,
        model: torch.nn.Module,
        peft_config: PeftConfig,
        adapter_name: str = "default",
        anchor_adapter: str = 'ema',
        train_adapters: List[str] = None,
        **kwargs
    ):
        super().__init__(model, peft_config, adapter_name, **kwargs)
        self.anchor_adapter = anchor_adapter
        self.base_adapter = adapter_name
        self.train_adapters = train_adapters
        
        self.current_adapter = None

        for adapter_name in [anchor_adapter, *train_adapters]:
            if adapter_name not in self.peft_config:
                self.add_adapter(adapter_name, peft_config)

    def weight_averaging_step(self, wa_type, adapters, adapter_name, weight):

        if wa_type not in ["slerp", "ema"]:
            raise NotImplementedError("Only (\'slerp\', \'ema\') for weight averaging")

        if adapter_name not in list(self.peft_config.keys()):
            _, new_rank, new_target_modules = self._check_add_weighted_adapter(
                adapters=adapters,
                combination_type="linear",
                svd_rank=None,
            )

            self.peft_config[adapter_name] = replace(
                self.peft_config[adapters[0]],
                r=new_rank,
                lora_alpha=new_rank,
                target_modules=new_target_modules,
            )
            self.inject_adapter(self.model, adapter_name)

        key_list = [
            key for key, _ in self.model.named_modules() if self.prefix not in key
        ]
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)

            if isinstance(target, LoraLayer):
                if adapter_name in target.lora_A:
                    target_lora_A = target.lora_A[adapter_name].weight
                    target_lora_B = target.lora_B[adapter_name].weight
                elif adapter_name in target.lora_embedding_A:
                    target_lora_A = target.lora_embedding_A[adapter_name]
                    target_lora_B = target.lora_embedding_B[adapter_name]
                else:
                    continue

                target_lora_A.data, target_lora_B.data = (
                    self.custom_add_weighted_adapter(
                        wa_type,
                        adapters,
                        [target_lora_A, target_lora_B],
                        weight,
                        target,
                    )
                )

    def custom_add_weighted_adapter(
        self,
        wa_type,
        adapters,
        base_tensors,
        weight,
        target,
    ):
        valid_weights = []
        lora_A_deltas = []
        lora_B_deltas = []
        for adapter in adapters:
            if adapter in target.lora_A:
                current_adapter_lora_A = target.lora_A[adapter].weight
                current_adapter_lora_B = target.lora_B[adapter].weight
            elif adapter in target.lora_embedding_A:
                current_adapter_lora_A = target.lora_embedding_A[adapter]
                current_adapter_lora_B = target.lora_embedding_B[adapter]
            else:
                continue
            valid_weights.append(weight * target.scaling[adapter])
            lora_A_deltas.append(current_adapter_lora_A.data)
            lora_B_deltas.append(current_adapter_lora_B.data)

        valid_weights = torch.tensor(valid_weights).to(lora_A_deltas[0].device)
        lora_deltas = [lora_A_deltas, lora_B_deltas]
        dtype = lora_A_deltas[0].dtype

        for i, (task_tensors, base_tensor) in enumerate(zip(lora_deltas, base_tensors)):
            if wa_type == "slerp":
                lora_deltas[i] = slerp(task_tensors, base_tensor, valid_weights)
            elif wa_type == "ema":
                lora_deltas[i] = ema(task_tensors, base_tensor, valid_weights)

        lora_deltas = [delta.to(dtype) for delta in lora_deltas]
        return lora_deltas
