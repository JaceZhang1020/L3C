#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from gym.envs.registration import register
from l3c.anymdp.anymdp_env import AnyMDPEnv, AnyMDPEnvD2C
from l3c.anymdp.anymdp_solver_opt import AnyMDPSolverOpt
from l3c.anymdp.anymdp_solver_ots import AnyMDPSolverOTS
from l3c.anymdp.anymdp_solver_q import AnyMDPSolverQ
from l3c.anymdp.task_sampler import AnyMDPTaskSampler
from l3c.anymdp.visualizer import task_visualizer

register(
    id='anymdp-v0',
    entry_point='l3c.anymdp:AnyMDPEnv',
    kwargs={"max_steps": 5000},
)

register(
    id='anymdp-d2c-v0',
    entry_point='l3c.anymdp:AnyMDPEnvD2C',
    kwargs={"max_steps": 5000,
            "state_dim": 16},
)