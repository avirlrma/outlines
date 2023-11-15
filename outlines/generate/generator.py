import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional, Union

import torch

if TYPE_CHECKING:
    from outlines.generate.samplers import Sampler
    from outlines.index.index import Index

"""
--------------------
Functions vs classes
--------------------

def text_generator(model, tokenizer, sampler, fsm):
    generate_token = token_generator(model, sampler)
    generator = generator(generate_token, fsm, tokenizer)
    return generator

generator = text_generator(model, tokenizer, sampler, fsm)
generator("prompt")

def call_text_generator(prompt):
    _*, sequence = generator(prompt)
    return sequence

--> We need to chain the generator with a generator that yields a `Sequence` state.
--> Then use this generator at the end when calling the calling function directly

def json(model, tokenizer, sampler, fsm):
    generate_token = token_generator(model, sampler)
    generator = sequence_generator(generate_token, fsm, tokenizer)
    return generator

"""


@dataclass(frozen=True)
class GenerationState:
    token_ids: torch.Tensor
    attention_masks: torch.Tensor
    kv_cache: Optional[torch.Tensor] = None
    rng: Optional[torch.Generator] = None


class SequenceGenerator:
    def __init__(self, fsm, model, sampler):
        self.generate_token = token_generator(model, sampler)
        self.fsm = fsm
        self.model = model
        self.device = self.model.device

    def init_generation_state(
        self, prompt: Union[str, List[str]], rng: Optional[torch.Generator] = None
    ):
        """Initialize the generation state.

        This method is responsible for encoding the prompt, moving token ids
        to the device and initializing the random number generator.

        Parameters
        ----------
        prompt
            The prompt on which the generation is conditioned.
        rng
            The state of the random number generator.

        Returns
        -------
        A `GenerationState` object.

        """
        token_ids, attention_masks = self.model.tokenizer.encode(prompt)
        token_ids = token_ids.squeeze(0).to(self.device)
        attention_masks = token_ids.squeeze(0).to(self.device)

        if rng is None:
            rng = torch.Generator(device=self.device)
            rng.seed()

        return GenerationState(token_ids, attention_masks, None, rng)

    def __call__(self, prompt, rng: Optional[torch.Generator]):
        self.state
        generator = self.stream(prompt, rng)
        *_, last = generator
        return last

    def stream(self, prompt, rng):
        self.state = self.init_generation_state(prompt, rng)
        self.fsm_states = [0 for _ in range(self.num_sequences)]
        return self

    def __iter__(self):
        """Generates new tokens based on the model and the FSM.

        Parameter
        ---------
        token_generator
            A generator that yields new tokens from a `GenerationState` and a list
            of the ids of the tokens we need to mask.
        index
            The index that drives the generation.

        """


def process(generator: Callable, index: "Index", state: GenerationState):
    fsm_states = [0 for _ in range(state.token_ids.shape[0])]
    while True:
        logits_masks = get_next_instructions(index, fsm_states)

        next_token_ids, kv_cache = generator(
            state.token_ids,
            state.attention_masks,
            state.kv_cache,
            logits_masks,
            1,
            state.rng,
        )

        token_ids = update_token_ids(state.token_ids, next_token_ids)
        attention_masks = update_attention_masks(state.attention_masks)
        state = GenerationState(token_ids, attention_masks, kv_cache)

        fsm_states = get_next_fsm_states(index, fsm_states, next_token_ids)
        is_finished = is_generation_finished(index, fsm_states)
        if is_finished:
            yield token_ids, next_token_ids
            return

        yield state


def token_generator(model, sampler: "Sampler") -> Callable:
    """Generate one token at a time.

    This process is designed to be steered by another supervising
    process that supplies the current sequence and the indices
    of the tokens to mask before sampling.

    Parameters
    ----------
    model
        A model that takes a sequence of tokens as an input and
        returns a probability distribution over the next tokens.
    sampler
        A function that samples tokens from a probability
        distribution over the next tokens.

    Returns
    -------
    A tensor with the sampled tokens.

    """

    def generate(
        token_ids,
        attention_masks,
        kv_cache,
        logits_masks,
        rng: torch.Generator,
    ):
        try:
            logits, new_kv_cache = model(token_ids, attention_masks, kv_cache)
        except IndexError:  # Exceeding the context length
            return

        biased_logits = bias_logits(logits, logits_masks)
        next_token_ids = sampler(biased_logits, 1, rng)

        yield next_token_ids, new_kv_cache

    return generate


def get_next_fsm_states(
    index, fsm_states: List[int], next_token_ids: torch.Tensor
) -> List[int]:
    return [
        index.next_state(fsm_state, token_id)
        for fsm_state, token_id in zip(fsm_states, next_token_ids)
    ]


def get_next_instructions(index, fsm_states: List[int]) -> torch.Tensor:
    return [index.next_instruction(state) for state in fsm_states]


def is_generation_finished(index, fsm_states: List[int]) -> bool:
    return all([index.is_finished(state) for state in fsm_states])


def update_token_ids(
    token_ids: torch.Tensor, next_token_ids: torch.Tensor
) -> torch.Tensor:
    return torch.concatenate([token_ids, next_token_ids], dim=1 - 1)


def update_attention_masks(attention_masks: torch.Tensor) -> torch.Tensor:
    return torch.concatenate(
        [
            attention_masks,
            torch.ones(
                attention_masks.shape[:-1] + (1,), device=attention_masks.device
            ),
        ],
        axis=-1,
    )


def bias_logits(
    logits: torch.Tensor,
    ids_to_mask: List,
) -> torch.Tensor:
    """Mask the logits.

    The function iterates over a nested list where each list corresponds to the
    indices that need to be masked for each row in the array.

    Parameters
    ----------
    logits
        Two dimensional tensor that contains the next-token probability
        distribution.
    ids_to_mask
        The ids to mask in each dimension.

    Returns
    -------
    A view of the original logits tensor where some values are masked.

    """
    for i, ids in enumerate(ids_to_mask):
        logits[i, ids] = -math.inf
    return logits
