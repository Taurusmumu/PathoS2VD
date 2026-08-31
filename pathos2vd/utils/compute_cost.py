from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

import torch


LOGGER = logging.getLogger(__name__)
T = TypeVar("T")
_GIB = float(1024**3)


@dataclass(frozen=True)
class ComputeCostSummary:
    hardware: str
    num_processes: int
    optimization_steps: int
    measured_steps: int
    optimization_time_seconds: float
    measured_time_seconds: float
    total_loop_time_seconds: float
    auxiliary_time_seconds: float
    peak_allocated_gib: list[float]
    peak_reserved_gib: list[float]

    @property
    def mean_iteration_seconds(self) -> float:
        return self.measured_time_seconds / self.measured_steps if self.measured_steps else 0.0

    @property
    def iterations_per_second(self) -> float:
        return self.measured_steps / self.measured_time_seconds if self.measured_time_seconds else 0.0


class TrainingComputeProfiler:
    """Low-overhead wall-clock and CUDA-memory measurement for Accelerate loops.

    Timing is synchronized only at warm-up/interval boundaries, around explicitly
    marked auxiliary work (for example checkpoint saving), and at the end.
    """

    def __init__(self, accelerator, training_config: dict, logging_config: dict | None = None) -> None:
        self.accelerator = accelerator
        self.enabled = bool(training_config.get("profile_compute_cost", False))
        self.warmup_steps = int(training_config.get("profile_warmup_steps", 10))
        self.log_interval = int(training_config.get("profile_log_interval", 100))
        if self.warmup_steps < 0:
            raise ValueError("profile_warmup_steps must be non-negative")
        if self.log_interval <= 0:
            raise ValueError("profile_log_interval must be positive")

        self.logging_config = logging_config or {}
        self._started = False
        self._finished = False
        self._optimization_steps = 0
        self._optimization_time = 0.0
        self._measured_time = 0.0
        self._auxiliary_time = 0.0
        self._loop_start = 0.0
        self._segment_start = 0.0
        self._measurement_active = self.warmup_steps == 0

    @property
    def _uses_cuda(self) -> bool:
        return torch.cuda.is_available() and self.accelerator.device.type == "cuda"

    def _timestamp(self) -> float:
        if self._uses_cuda:
            torch.cuda.synchronize(self.accelerator.device)
        return time.perf_counter()

    def _wandb_enabled(self) -> bool:
        report_to = self.logging_config.get("report_to")
        targets = report_to if isinstance(report_to, (list, tuple)) else [report_to]
        return "wandb" in targets

    def _initialize_wandb(self) -> None:
        if not self._wandb_enabled() or self.accelerator.trackers:
            return
        project_name = self.logging_config.get("project_name", "pathos2vd")
        init_kwargs = {}
        if self.logging_config.get("run_name"):
            init_kwargs["wandb"] = {"name": self.logging_config["run_name"]}
        self.accelerator.init_trackers(project_name, init_kwargs=init_kwargs)

    def start(self) -> None:
        if not self.enabled:
            return
        if self._started:
            raise RuntimeError("TrainingComputeProfiler.start() called more than once")

        # Tracker initialization is deliberately outside the measured region.
        self._initialize_wandb()
        self.accelerator.wait_for_everyone()
        if self._uses_cuda:
            torch.cuda.synchronize(self.accelerator.device)
            torch.cuda.reset_peak_memory_stats(self.accelerator.device)
        now = time.perf_counter()
        self._loop_start = now
        self._segment_start = now
        self._started = True

    def _close_optimization_segment(self, now: float) -> None:
        elapsed = now - self._segment_start
        self._optimization_time += elapsed
        if self._measurement_active:
            self._measured_time += elapsed

    def record_optimization_step(self, *, completed: bool = True) -> None:
        """Record one real optimizer update, not a gradient-accumulation micro-step."""
        if not self.enabled or not completed:
            return
        if not self._started or self._finished:
            raise RuntimeError("Profiler must be active when recording an optimization step")

        self._optimization_steps += 1
        measured_steps = max(self._optimization_steps - self.warmup_steps, 0)
        warmup_boundary = self._optimization_steps == self.warmup_steps and self.warmup_steps > 0
        interval_boundary = measured_steps > 0 and measured_steps % self.log_interval == 0
        if warmup_boundary or interval_boundary:
            now = self._timestamp()
            self._close_optimization_segment(now)
            self._segment_start = now
            if warmup_boundary:
                self._measurement_active = True

    def time_auxiliary(self, callback: Callable[[], T]) -> T:
        """Run checkpoint/validation work outside optimization-time accounting."""
        if not self.enabled:
            return callback()
        if not self._started or self._finished:
            raise RuntimeError("Profiler must be active when timing auxiliary work")

        auxiliary_start = self._timestamp()
        self._close_optimization_segment(auxiliary_start)
        try:
            return callback()
        finally:
            self.accelerator.wait_for_everyone()
            auxiliary_end = self._timestamp()
            self._auxiliary_time += auxiliary_end - auxiliary_start
            self._segment_start = auxiliary_end

    def finish(self) -> ComputeCostSummary | None:
        if not self.enabled:
            return None
        if not self._started or self._finished:
            raise RuntimeError("Profiler must be started exactly once before finish()")

        self.accelerator.wait_for_everyone()
        end = self._timestamp()
        self._close_optimization_segment(end)
        self._finished = True

        allocated = torch.cuda.max_memory_allocated(self.accelerator.device) / _GIB if self._uses_cuda else 0.0
        reserved = torch.cuda.max_memory_reserved(self.accelerator.device) / _GIB if self._uses_cuda else 0.0
        local = torch.tensor(
            [
                self._optimization_time,
                self._measured_time,
                end - self._loop_start,
                self._auxiliary_time,
                float(self._optimization_steps),
                allocated,
                reserved,
            ],
            dtype=torch.float64,
            device=self.accelerator.device,
        )
        gathered = self.accelerator.gather(local).reshape(-1, local.numel()).cpu()

        step_counts = gathered[:, 4].to(torch.int64)
        if self.accelerator.is_main_process and not torch.equal(step_counts, step_counts[:1].expand_as(step_counts)):
            LOGGER.warning("Optimization-step counts differ across processes: %s", step_counts.tolist())
        optimization_steps = int(step_counts.min().item())
        measured_steps = max(optimization_steps - self.warmup_steps, 0)
        hardware = torch.cuda.get_device_name(self.accelerator.device) if self._uses_cuda else "CPU"
        summary = ComputeCostSummary(
            hardware=hardware,
            num_processes=int(self.accelerator.num_processes),
            optimization_steps=optimization_steps,
            measured_steps=measured_steps,
            optimization_time_seconds=float(gathered[:, 0].max().item()),
            measured_time_seconds=float(gathered[:, 1].max().item()),
            total_loop_time_seconds=float(gathered[:, 2].max().item()),
            auxiliary_time_seconds=float(gathered[:, 3].max().item()),
            peak_allocated_gib=gathered[:, 5].tolist(),
            peak_reserved_gib=gathered[:, 6].tolist(),
        )
        self._report(summary)
        return summary

    def _report(self, summary: ComputeCostSummary) -> None:
        metrics = {
            "compute_cost/optimization_steps": summary.optimization_steps,
            "compute_cost/measured_steps": summary.measured_steps,
            "compute_cost/optimization_time_hours": summary.optimization_time_seconds / 3600.0,
            "compute_cost/total_loop_time_hours": summary.total_loop_time_seconds / 3600.0,
            "compute_cost/auxiliary_time_hours": summary.auxiliary_time_seconds / 3600.0,
            "compute_cost/mean_iteration_seconds": summary.mean_iteration_seconds,
            "compute_cost/iterations_per_second": summary.iterations_per_second,
            "compute_cost/max_peak_allocated_gib": max(summary.peak_allocated_gib, default=0.0),
            "compute_cost/max_peak_reserved_gib": max(summary.peak_reserved_gib, default=0.0),
        }
        for index, value in enumerate(summary.peak_allocated_gib):
            metrics[f"compute_cost/gpu_{index}_peak_allocated_gib"] = value
        for index, value in enumerate(summary.peak_reserved_gib):
            metrics[f"compute_cost/gpu_{index}_peak_reserved_gib"] = value

        if self._wandb_enabled():
            self.accelerator.log(metrics, step=summary.optimization_steps)
        if not self.accelerator.is_main_process:
            return

        memory_lines = [
            f"GPU {index}: peak allocated {allocated:.2f} GiB, peak reserved {reserved:.2f} GiB"
            for index, (allocated, reserved) in enumerate(
                zip(summary.peak_allocated_gib, summary.peak_reserved_gib)
            )
        ]
        LOGGER.info(
            "\n===== Training Computational Cost =====\n"
            "Hardware: %d x %s\n"
            "Number of optimization steps: %d\n"
            "Measured steps excluding warm-up: %d\n"
            "Optimization time: %.2f hours\n"
            "Checkpoint/auxiliary time: %.2f hours\n"
            "Total training-loop wall time: %.2f hours\n"
            "Mean time / iteration: %.3f s\n"
            "Throughput: %.3f iterations/s\n"
            "%s\n"
            "Peak allocated memory / GPU: %s\n"
            "Peak reserved memory / GPU: %s\n"
            "Maximum peak allocated memory: %.2f GiB\n"
            "Maximum peak reserved memory: %.2f GiB\n"
            "=======================================",
            summary.num_processes,
            summary.hardware,
            summary.optimization_steps,
            summary.measured_steps,
            summary.optimization_time_seconds / 3600.0,
            summary.auxiliary_time_seconds / 3600.0,
            summary.total_loop_time_seconds / 3600.0,
            summary.mean_iteration_seconds,
            summary.iterations_per_second,
            "\n".join(memory_lines),
            [round(value, 3) for value in summary.peak_allocated_gib],
            [round(value, 3) for value in summary.peak_reserved_gib],
            max(summary.peak_allocated_gib, default=0.0),
            max(summary.peak_reserved_gib, default=0.0),
        )
