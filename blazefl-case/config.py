from dataclasses import dataclass
from enum import StrEnum

from blazefl.core import IPCMode

from models.selector import FedAvgModelName


class ExecutionMode(StrEnum):
    SINGLE_THREADED = "SINGLE_THREADED"
    MULTI_PROCESS = "MULTI_PROCESS"
    MULTI_THREADED = "MULTI_THREADED"


@dataclass
class MyConfig:
    model_name: FedAvgModelName = FedAvgModelName.CNN
    num_clients: int = 100
    global_round: int = 10
    sample_ratio: float = 1.0
    partition: str = "shards"
    num_shards: int = 200
    dir_alpha: float = 1.0
    seed: int = 42
    epochs: int = 5
    lr: float = 0.1
    batch_size: int = 50
    num_parallels: int = 10
    dataset_root_dir: str = "/tmp/blazefl-case/dataset"
    dataset_split_dir: str = "/tmp/blazefl-case/split"
    share_dir: str = "/tmp/blazefl-case/share"
    state_dir: str = "/tmp/blazefl-case/state"
    execution_mode: ExecutionMode = ExecutionMode(ExecutionMode.MULTI_THREADED)
    ipc_mode: IPCMode = IPCMode(IPCMode.SHARED_MEMORY)
