"""
优先级映射：模拟退火优化作业顺序，使 G = (满足 SLO 的作业数) / (总延迟) 尽可能大。

由 C++ `priority_mapper.cpp` / `priority_mapper.h` 迁移；对外以 dict 列表为接口。
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple
from vllm.v1.request import Request, RequestStatus, StreamingUpdate


class JobType(str, Enum):
    """与 C++ `enum class JobType` 对应。"""

    E2E_LATENCY = "E2E_LATENCY"
    INTERACTIVE = "INTERACTIVE"


@dataclass
class SLO:
    e2e: float = 30000.0
    ttft: float = 10000.0
    tpot: float = 50.0


@dataclass
class Job:
    id: int
    type: JobType
    input_length: str
    output_length: int
    ttft: float
    tpot: float
    slo: SLO
    e2e: float = field(init=False)
    prio: int = -1
    batch_id: int = -1
    wait_time: float = 0.0

    def __post_init__(self) -> None:
        self.e2e = self.ttft + self.tpot * self.output_length

    def get_actual_e2e(self) -> float:
        return self.wait_time + self.e2e

    def get_actual_ttft(self) -> float:
        return self.wait_time + self.ttft

    def meets_slo(self) -> bool:
        if self.type == JobType.E2E_LATENCY:
            return self.get_actual_e2e() <= self.slo.e2e
        return self.get_actual_ttft() <= self.slo.ttft and self.tpot <= self.slo.tpot


def _parse_job_type(v: Any) -> JobType:
    if isinstance(v, JobType):
        return v
    s = str(v).strip()
    if s in ("E2E_LATENCY", "0"):
        return JobType.E2E_LATENCY
    if s in ("INTERACTIVE", "1"):
        return JobType.INTERACTIVE
    return JobType(s)


def _parse_slo(d: Optional[Dict[str, Any]]) -> SLO:
    if not d:
        return SLO()
    return SLO(
        e2e=float(d.get("e2e", 30000.0)),
        ttft=float(d.get("ttft", 10000.0)),
        tpot=float(d.get("tpot", 50.0)),
    )


def job_from_dict(d: Dict[str, Any]) -> Job:
    """从输入 dict 构造 `Job`。必填：id, type, input_length, output_length, ttft, tpot；可选 slo、prio、batch_id、wait_time。"""
    slo = _parse_slo(d.get("slo"))
    j = Job(
        id=int(d["id"]),
        type=_parse_job_type(d["type"]),
        input_length=str(d["input_length"]),
        output_length=int(d["output_length"]),
        ttft=float(d["ttft"]),
        tpot=float(d["tpot"]),
        slo=slo,
    )
    if "prio" in d:
        j.prio = int(d["prio"])
    if "batch_id" in d:
        j.batch_id = int(d["batch_id"])
    if "wait_time" in d:
        j.wait_time = float(d["wait_time"])
    return j


def _ensure_job(d: Dict[str, Any] | Job) -> Job:
    """optimize_job_priority 的输入可为作业 dict 或已构造的 Job。"""
    if isinstance(d, Job):
        return d
    return job_from_dict(d)


def job_to_dict(j: Job) -> Dict[str, Any]:
    """将 `Job` 转为 dict，便于序列化或与业务代码对接。"""
    return {
        "id": j.id,
        "type": j.type.value,
        "input_length": j.input_length,
        "output_length": j.output_length,
        "ttft": j.ttft,
        "tpot": j.tpot,
        "e2e": j.e2e,
        "slo": {"e2e": j.slo.e2e, "ttft": j.slo.ttft, "tpot": j.slo.tpot},
        "prio": j.prio,
        "batch_id": j.batch_id,
        "wait_time": j.wait_time,
    }


class SimulatedAnnealingPriorityMapper:
    """模拟退火优先级映射器。"""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def set_seed(self, seed: int) -> None:
        self._rng.seed(seed)

    @staticmethod
    def count_meets_slo(jobs: Sequence[Job]) -> int:
        return sum(1 for j in jobs if j.meets_slo())

    @staticmethod
    def calculate_total_latency(jobs: Sequence[Job]) -> float:
        return sum(j.get_actual_e2e() for j in jobs)

    def calculate_g(self, jobs: Sequence[Job]) -> float:
        n = self.count_meets_slo(jobs)
        t = self.calculate_total_latency(jobs)
        if t <= 0:
            return 0.0
        return float(n) / t

    def compute_wait_times_and_batches(self, jobs: List[Job], max_batch_size: int) -> None:
        if not jobs:
            return
        for j in jobs:
            j.wait_time = 0.0
            j.batch_id = -1

        current_batch = 0
        batch_count = 0
        batch_execution_time = 0.0
        cumulative_time = 0.0

        for i in range(len(jobs)):
            if batch_count >= max_batch_size:
                cumulative_time += batch_execution_time
                current_batch += 1
                batch_count = 0
                batch_execution_time = 0.0

            jobs[i].batch_id = current_batch
            jobs[i].wait_time = cumulative_time
            batch_execution_time = max(batch_execution_time, jobs[i].e2e)
            batch_count += 1

    def _squeeze_last_iter(self, jobs: List[Job]) -> List[Job]:
        if len(jobs) <= 1:
            return copy.deepcopy(jobs)
        new_jobs = copy.deepcopy(jobs)
        pos = self._rng.randint(0, len(jobs) - 2)
        last = new_jobs.pop()
        new_jobs.insert(pos, last)
        return new_jobs

    def _delay_next_iter(self, jobs: List[Job]) -> List[Job]:
        if len(jobs) <= 1:
            return copy.deepcopy(jobs)
        new_jobs = copy.deepcopy(jobs)
        pos = self._rng.randint(1, len(jobs) - 1)
        first = new_jobs.pop(0)
        new_jobs.insert(pos - 1, first)
        return new_jobs

    def _rand_swapping(self, jobs: List[Job]) -> List[Job]:
        if len(jobs) <= 1:
            return copy.deepcopy(jobs)
        new_jobs = copy.deepcopy(jobs)
        i = self._rng.randrange(len(jobs))
        j = self._rng.randrange(len(jobs))
        while j == i:
            j = self._rng.randrange(len(jobs))
        new_jobs[i], new_jobs[j] = new_jobs[j], new_jobs[i]
        return new_jobs

    def _adjust_batch_distribution(self, jobs: List[Job], max_batch_size: int) -> List[Job]:
        if len(jobs) <= 1:
            return copy.deepcopy(jobs)
        new_jobs = copy.deepcopy(jobs)
        n = len(new_jobs)
        batch_ids: List[int] = []
        current_batch = 0
        batch_count = 0
        for _ in range(n):
            if batch_count >= max_batch_size:
                current_batch += 1
                batch_count = 0
            batch_ids.append(current_batch)
            batch_count += 1

        candidates: List[Tuple[int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                if batch_ids[i] != batch_ids[j]:
                    candidates.append((i, j))

        if candidates:
            i, j = self._rng.choice(candidates)
            new_jobs[i], new_jobs[j] = new_jobs[j], new_jobs[i]
        return new_jobs

    def priority_mapping(
        self,
        jobs_in: List[Job],
        max_batch_size: int,
        t0: float = 500.0,
        t_thres: float = 20.0,
        inner_iter: int = 100,
        tau: float = 0.95,
        verbose: bool = False,
    ) -> List[Job]:
        if not jobs_in:
            return []

        jobs_in = copy.deepcopy(jobs_in)
        t = t0

        j_prio = copy.deepcopy(jobs_in)
        j_prio.sort(key=lambda x: x.e2e)
        self.compute_wait_times_and_batches(j_prio, max_batch_size)
        f = self.calculate_g(j_prio)

        if self.count_meets_slo(j_prio) == len(j_prio):
            for i, job in enumerate(j_prio):
                job.prio = i
            return j_prio

        j_temp = copy.deepcopy(jobs_in)
        self.compute_wait_times_and_batches(j_temp, max_batch_size)
        f_init = self.calculate_g(j_temp)

        if f < f_init:
            j_prio = j_temp
            f = f_init

        j_best = copy.deepcopy(j_prio)
        f_best = f

        iteration = 0
        # 与 C++ `do { ... T *= tau; } while (T >= T_thres);` 一致：至少做一轮内层迭代
        while True:
            k = 0
            while k < inner_iter:
                k += 1
                iteration += 1
                op = self._rng.randint(0, 3)
                if op == 0:
                    j_cur = self._squeeze_last_iter(j_prio)
                elif op == 1:
                    j_cur = self._delay_next_iter(j_prio)
                elif op == 2:
                    j_cur = self._rand_swapping(j_prio)
                else:
                    j_cur = self._adjust_batch_distribution(j_prio, max_batch_size)

                self.compute_wait_times_and_batches(j_cur, max_batch_size)
                f_new = self.calculate_g(j_cur)

                if f_new > f:
                    j_prio = j_cur
                    f = f_new
                    if f > f_best:
                        j_best = copy.deepcopy(j_prio)
                        f_best = f
                else:
                    acceptance = math.exp((f_new - f) / t) if t > 0 else 0.0
                    if acceptance > self._rng.random():
                        j_prio = j_cur
                        f = f_new

            t *= tau
            if t < t_thres:
                break

        j_prio = copy.deepcopy(j_best)
        self.compute_wait_times_and_batches(j_prio, max_batch_size)
        for i, job in enumerate(j_prio):
            job.prio = i

        if verbose:
            print(f"模拟退火完成，总迭代次数: {iteration}")

        return j_prio

    def evaluate_schedule(self, jobs: Sequence[Job]) -> None:
        total = len(jobs)
        if total == 0:
            print("\n===== 调度结果评估 =====\n总作业数: 0")
            return
        met = self.count_meets_slo(jobs)
        total_latency = self.calculate_total_latency(jobs)
        g = self.calculate_g(jobs)
        print("\n===== 调度结果评估 =====")
        print(f"总作业数: {total}")
        print(f"满足SLO的作业数: {met}")
        print(f"SLO达成率: {100.0 * met / total:.2f}%")
        print(f"总延迟: {total_latency:.2f} ms")
        print(f"平均延迟: {total_latency / total:.2f} ms")
        print(f"适应度 G: {g:.4e} req/ms")


def optimize_job_priority(
    jobs: List[Dict[str, Any] | Job],
    max_batch_size: int,
    seed: int = 42,
    *,
    t0: float = 80.0,
    t_thres: float = 80.0,
    inner_iter: int = 8,
    tau: float = 0.82,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    简化入口：输入作业 dict 列表或 Job 实例列表，返回重排并写好 prio/batch_id/wait_time 的 dict 列表。

    每个作业 dict 需包含：id, type, input_length, output_length, ttft, tpot；
    可选 slo: {e2e, ttft, tpot}。若传入 Job，则不再做 dict 解析。
    """
    mapper = SimulatedAnnealingPriorityMapper(seed=seed)
    parsed = [_ensure_job(d) for d in jobs]
    out = mapper.priority_mapping(
        parsed, max_batch_size, t0=t0, t_thres=t_thres, inner_iter=inner_iter, tau=tau, verbose=verbose
    )
    return [job_to_dict(j) for j in out]


'''
inner_iter	每个温度下内层尝试次数，最吃 CPU	从默认 12 再降到 4～8
t_thres	温度低于此值就结束	略增大（如 55→80）→ 更早停，外层轮数更少
t0	初始温度	略减小（如 120→80）→ 一般更少外层轮数
tau	每轮外层降温乘子	略减小（如 0.88→0.82）→ 降温更快、轮数更少
'''

