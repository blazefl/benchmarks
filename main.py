import logging
import math
import subprocess
import os
from typing import NamedTuple
import torch
import argparse
import re
import statistics

from rich.logging import RichHandler
from rich.progress import track
from rich.console import Console
from rich.table import Table


def run_benchmark(command: str, num_runs: int) -> list[float]:
    execution_times: list[float] = []
    for _ in track(range(num_runs), description="Benchmark Progress"):
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        output = result.stdout + result.stderr
        match = re.search(r"BENCHMARK_RESULT_TIME: ([\d.]+)", output)
        if match:
            execution_times.append(float(match.group(1)))
        else:
            logging.error(f"Could not find benchmark result in output:\n{output}")
    return execution_times


class Result(NamedTuple):
    method: str
    avg_time: float
    std_time: float


def display_results(title: str, results: list[Result]) -> None:
    console = Console()

    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Method", style="dim")
    table.add_column("Execution Time (s)", justify="right")
    for result in results:
        table.add_row(result.method, f"{result.avg_time:.4f} ± {result.std_time:.4f}")

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Benchmark BlazeFL and Flower.")
    parser.add_argument(
        "--min-parallel",
        type=int,
        default=2,
        help="Minimum number of parallel workers.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=10,
        help="Maximum number of parallel workers.",
    )
    parser.add_argument(
        "--num-runs", type=int, default=2, help="Number of runs for each setting."
    )
    args = parser.parse_args()

    cpu_count = os.cpu_count() or 1
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    num_parallels: list[int] = [2**i for i in range(int(math.log2(cpu_count) + 1))]
    num_parallels = [8]

    for num_parallel in num_parallels:
        results: list[Result] = []

        blazefl_command = f"cd blazefl-case && uv run python main.py execution_mode=MULTI_THREADED num_parallels={num_parallel} && cd .."
        blazefl_times = run_benchmark(blazefl_command, args.num_runs)
        if blazefl_times:
            result = Result(
                method="BlazeFL",
                avg_time=statistics.mean(blazefl_times),
                std_time=statistics.stdev(blazefl_times),
            )
            results.append(result)

        client_cpus = cpu_count // num_parallel
        client_gpus = gpu_count / num_parallel

        flower_command = (
            f"cd flower-case && uv run flwr run . local-simulation "
            f'--federation-config "options.num-supernodes=100 '
            f"options.backend.client-resources.num-cpus={client_cpus} "
            f'options.backend.client-resources.num-gpus={client_gpus}" '
            "&& cd .."
        )
        print(flower_command)
        flower_times = run_benchmark(flower_command, args.num_runs)
        if flower_times:
            result = Result(
                method="Flower",
                avg_time=statistics.mean(flower_times),
                std_time=statistics.stdev(flower_times),
            )
            results.append(result)

        display_results(f"FedAvg Benchmark ({num_parallel=})", results)


if __name__ == "__main__":
    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(markup=True, rich_tracebacks=True)],
    )
    main()
