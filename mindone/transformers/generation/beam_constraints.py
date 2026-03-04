# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Beam search constraints for constrained generation.

Note: This module was removed in transformers v5.0.0. This is a stub implementation
for backward compatibility with mindone.
"""

from abc import ABC, abstractmethod
from typing import Optional

from transformers.utils import logging


logger = logging.get_logger(__name__)


class Constraint(ABC):
    """Abstract base class for all constraints that can be applied during generation."""

    def __init__(self):
        pass

    @abstractmethod
    def advance(self):
        """Returns the token(s) that would take this constraint one step closer to being fulfilled."""
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def does_advance(self, token_id: int):
        """Reads in a token and returns whether it creates progress."""
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def update(self, token_id: int):
        """Reads in a token and returns booleans that indicate the progress made by it."""
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def reset(self):
        """Resets the state of this constraint to its initialization."""
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def remaining(self):
        """Returns the number of remaining steps of `advance()` in order to complete this constraint."""
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def copy(self, stateful=False):
        """Creates a new instance of this constraint."""
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class ConstraintListState:
    """
    Records the progress of fulfilling constraints in a list of constraints.

    Args:
        constraints (`list[Constraint]`):
            A list of constraints that need to be fulfilled.
    """

    def __init__(self, constraints: list[Constraint]):
        if constraints is None or len(constraints) == 0:
            raise ValueError("`constraints` has to be a non-empty list.")

        self.constraints = constraints
        self.constraint_states = [constraint.copy(stateful=True) for constraint in self.constraints]

    def get_bank(self):
        """
        Retrieves the "bank" of tokens that can be used to advance the constraints.

        Return:
            `list[Optional[int]]`: A list of token ids or None values representing the bank.
        """
        tokens = []
        for constraint in self.constraint_states:
            tokens.append(constraint.advance())
        return tokens

    def advance(self, token_id: int):
        """
        Advances the state of constraints by reading in a token.

        Args:
            token_id (`int`):
                The id of a newly generated token in the beam search.

        Return:
            `bool`: Whether the token has advanced the constraints.
        """
        any_advanced = False

        for constraint in self.constraint_states:
            stepped, completed, reset = constraint.update(token_id)
            if stepped:
                any_advanced = True

        return any_advanced

    def reset(self, token_id: int):
        """
        Resets the state of constraints by reading in a token.

        Args:
            token_id (`int`):
                The id of a newly generated token in the beam search that causes a reset.

        Return:
            `bool`: Whether the token has caused a reset.
        """
        any_reset = False

        for constraint in self.constraint_states:
            if constraint.does_advance(token_id):
                continue
            else:
                if hasattr(constraint, 'reset'):
                    constraint.reset()
                any_reset = True

        return any_reset

    def completed(self):
        """
        Checks whether all constraints have been completed.

        Return:
            `bool`: Whether all constraints have been completed.
        """
        return all(constraint.remaining() == 0 for constraint in self.constraint_states)

    def copy(self):
        """
        Creates a copy of this ConstraintListState.

        Return:
            `ConstraintListState`: A copy of this ConstraintListState.
        """
        new_state = ConstraintListState.__new__(ConstraintListState)
        new_state.constraints = self.constraints
        new_state.constraint_states = [constraint.copy(stateful=True) for constraint in self.constraints]
        return new_state
