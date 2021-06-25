#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .psenet import PSENET_IC15
from .psenet import PSENET_IC17
from .psenet import PSENET_TT
from .psenet import PSENET_Synth
from .builder import build_data_loader

__all__ = ['PSENET_IC15','PSENET_IC17','PSENET_TT','PSENET_Synth']
