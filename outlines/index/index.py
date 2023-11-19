from dataclasses import dataclass
from typing import NewType, Protocol, Union

import torch

State = NewType("State", int)


@dataclass(frozen=True)
class GenerateInstruction:
    logits_mask: str
    temperature: float
    top_k: int
    top_p: int


@dataclass(frozen=True)
class FillInstruction:
    token_ids: int


FSMInstruction = Union[GenerateInstruction, FillInstruction]


class Index(Protocol):
    def next_instruction(self, state: State) -> FSMInstruction:
        ...

    def next_state(self, state: State, token_id: torch.Tensor) -> State:
        ...

    def is_final_state(self, state: State) -> bool:
        ...
