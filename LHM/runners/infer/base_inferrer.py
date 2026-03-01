# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from abc import abstractmethod

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger

from LHM.runners.abstract import Runner
import safetensors

logger = get_logger(__name__)


class Inferrer(Runner):

    EXP_TYPE: str = None

    def __init__(self):
        super().__init__()

        torch._dynamo.config.disable = True
        self.accelerator = Accelerator()

        self.model: torch.nn.Module = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @property
    def device(self):
        return self.accelerator.device

    @abstractmethod
    def _build_model(self, cfg):
        pass

    @abstractmethod
    def infer_single(self, *args, **kwargs):
        pass

    @abstractmethod
    def infer(self):
        pass

    def load_model_(self, load_model):
        logger.info(f"======== Loading model from {load_model} ========")
        safetensors.torch.load_model(
            self.accelerator.unwrap_model(self.model),
            load_model,
            strict=False, #qw00n; except for Sapiens
        )
        logger.info(f"======== Model loaded ========")

    def run(self):
        self.infer()
