import logging
import math
from pathlib import Path
import subprocess
import os
from typing import NamedTuple
import torch
import re
import statistics
from datetime import datetime

from rich.logging import RichHandler
from rich.progress import track
from rich.console import Console
from rich.table import Table
import typer

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"{datetime.now():%Y%m%d_%H%M%S}.log"


def setup_logging() -> None:
    console_handler = RichHandler(markup=True, rich_tracebacks=True)
    console_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="[%X]")
    )

    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[console_handler, file_handler],
    )


def run_benchmark(command: str, num_runs: int) -> list[float]:
    execution_times: list[float] = []
    for i in track(range(num_runs), description="Benchmark Progress"):
        logging.info(f"Running benchmark (run {i + 1}/{num_runs}): {command}")
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, check=True
            )

            if result.stdout:
                logging.debug(f"Run {i + 1} stdout:\n{result.stdout.strip()}")
            if result.stderr:
                logging.debug(f"Run {i + 1} stderr:\n{result.stderr.strip()}")

            output = result.stdout + result.stderr
            match = re.search(r"BENCHMARK_RESULT_TIME: ([\d.]+)", output)
            if match:
                time_taken = float(match.group(1))
                execution_times.append(time_taken)
                logging.info(
                    f"Run {i + 1}/{num_runs} finished in {time_taken:.4f} seconds."
                )
            else:
                logging.warning(
                    f"Could not find benchmark result in output for run {i + 1}."
                )
        except subprocess.CalledProcessError as e:
            logging.error(f"Command failed for run {i + 1}/{num_runs}")
            logging.error(f"Return code: {e.returncode}")

            if e.stdout:
                logging.error(f"stdout:\n{e.stdout.strip()}")
            if e.stderr:
                logging.error(f"stderr:\n{e.stderr.strip()}")
            continue

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
        row = f"{result.avg_time:.4f} ± {result.std_time:.4f}"
        table.add_row(result.method, row)
        logging.info(f"Result - {result.method}: {row}")

    console.print(table)


def main(num_runs: int = 3, model_name: str = "cnn") -> None:
    logging.info("Starting benchmark...")
    cpu_count = os.cpu_count() or 1
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    num_parallels: list[int] = [2**i for i in range(int(math.log2(cpu_count) + 1))]

    for num_parallel in num_parallels:
        results: list[Result] = []

        execution_modes = ["MULTI_THREADED", "MULTI_PROCESS"]
        for mode in execution_modes:
            blazefl_command = (
                f"cd blazefl-case && "
                f"uv run python {'-Xgil=0' if mode == 'MULTI_THREADED' else ''} main.py "
                f"model_name={model_name.upper()} execution_mode={mode} num_parallels={num_parallel} "
                "&& cd .."
            )
            blazefl_times = run_benchmark(blazefl_command, num_runs)
            if blazefl_times:
                result = Result(
                    method=f"BlazeFL ({mode})",
                    avg_time=statistics.mean(blazefl_times),
                    std_time=statistics.stdev(blazefl_times)
                    if len(blazefl_times) > 1
                    else 0.0,
                )
                results.append(result)

        client_cpus = cpu_count // num_parallel
        client_gpus_float = gpu_count / num_parallel
        if client_gpus_float >= 1:
            client_gpus = int(client_gpus_float)
        else:
            client_gpus = client_gpus_float

        flower_command = (
            "cd flower-case && "
            "uv run flwr run . "
            "local-simulation "
            '--federation-config "options.num-supernodes=100 '
            f"options.backend.client-resources.num-cpus={client_cpus} "
            f'options.backend.client-resources.num-gpus={client_gpus}" '
            f"""--run-config "model-name='{model_name}'" """
            "&& cd .."
        )
        flower_times = run_benchmark(flower_command, num_runs)
        if flower_times:
            result = Result(
                method="Flower",
                avg_time=statistics.mean(flower_times),
                std_time=statistics.stdev(flower_times)
                if len(flower_times) > 1
                else 0.0,
            )
            results.append(result)

        display_results(f"FedAvg Benchmark ({num_parallel=})", results)
    logging.info("Benchmark finished.")


if __name__ == "__main__":
    setup_logging()
    typer.run(main)
