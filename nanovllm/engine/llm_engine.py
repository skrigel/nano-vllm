import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp
import numpy as np

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        seq.t_submitted = perf_counter()
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
        return_metrics: bool = False,
        max_concurrency: int | None = None,
    ) -> tuple[list[dict], dict] | list[dict]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        sequences = {}

        # Closed-loop workload: maintain fixed concurrency
        if max_concurrency is not None:
            pending_requests = list(zip(prompts, sampling_params))
            # Submit first N requests
            for prompt, sp in pending_requests[:max_concurrency]:
                self.add_request(prompt, sp)
            pending_requests = pending_requests[max_concurrency:]
            # Track sequences by ID for metrics
            for seq in self.scheduler.waiting:
                sequences[seq.seq_id] = seq
        else:
            # Burst mode: submit all requests at once
            for prompt, sp in zip(prompts, sampling_params):
                self.add_request(prompt, sp)
            # Track sequences by ID for metrics
            for seq in self.scheduler.waiting:
                sequences[seq.seq_id] = seq
            pending_requests = []

        outputs = {}
        prefill_throughput = decode_throughput = 0.
        total_prefill_tokens = total_prefill_time = 0.
        total_decode_tokens = total_decode_time = 0.
        ttfts = []
        latencies = []
        tpots = []
        overall_start = perf_counter()
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            step_time = perf_counter() - t
            if num_tokens > 0:
                prefill_throughput = num_tokens / step_time
                total_prefill_tokens += num_tokens
                total_prefill_time += step_time
            else:
                decode_throughput = -num_tokens / step_time
                total_decode_tokens += -num_tokens
                total_decode_time += step_time
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                seq = sequences.pop(seq_id, None)
                if seq:
                    if seq.t_first_token is not None:
                        ttfts.append(seq.t_first_token - seq.t_submitted)
                    if seq.t_completed is not None:
                        latencies.append(seq.t_completed - seq.t_submitted)
                    # Calculate per-request TPOT
                    if seq.t_first_token is not None and seq.t_completed is not None and seq.num_completion_tokens > 1:
                        tpot = (seq.t_completed - seq.t_first_token) / (seq.num_completion_tokens - 1)
                        tpots.append(tpot)

                # Closed-loop: submit next request when one finishes
                if max_concurrency is not None and pending_requests:
                    prompt, sp = pending_requests.pop(0)
                    self.add_request(prompt, sp)
                    # Track new sequence
                    for new_seq in self.scheduler.waiting:
                        if new_seq.seq_id not in sequences:
                            sequences[new_seq.seq_id] = new_seq
                            break

                pbar.update(1)

        overall_end = perf_counter()
        overall_time = overall_end - overall_start
        pbar.close()

        # Build metrics dict
        total_tokens = total_prefill_tokens + total_decode_tokens
        metrics = {
            "overall_time_s": overall_time,
            "total_tokens": total_tokens,
            "overall_throughput_tok_per_s": total_tokens / overall_time if overall_time > 0 else 0,
            "prefill_tokens": total_prefill_tokens,
            "prefill_time_s": total_prefill_time,
            "prefill_throughput_tok_per_s": total_prefill_tokens / total_prefill_time if total_prefill_time > 0 else 0,
            "decode_tokens": total_decode_tokens,
            "decode_time_s": total_decode_time,
            "decode_throughput_tok_per_s": total_decode_tokens / total_decode_time if total_decode_time > 0 else 0,
            "tpot_ms": (total_decode_time / total_decode_tokens) * 1000 if total_decode_tokens > 0 else 0,
        }

        # Add TTFT statistics
        if ttfts:
            ttfts_ms = [t * 1000 for t in ttfts]
            metrics.update({
                "ttft_avg_ms": np.mean(ttfts_ms),
                "ttft_min_ms": np.min(ttfts_ms),
                "ttft_max_ms": np.max(ttfts_ms),
                "ttft_p50_ms": np.percentile(ttfts_ms, 50),
                "ttft_p90_ms": np.percentile(ttfts_ms, 90),
                "ttft_p99_ms": np.percentile(ttfts_ms, 99),
                "ttft_values_s": ttfts,
            })

        # Add latency statistics
        if latencies:
            latencies_ms = [l * 1000 for l in latencies]
            metrics.update({
                "latency_avg_ms": np.mean(latencies_ms),
                "latency_min_ms": np.min(latencies_ms),
                "latency_max_ms": np.max(latencies_ms),
                "latency_p50_ms": np.percentile(latencies_ms, 50),
                "latency_p90_ms": np.percentile(latencies_ms, 90),
                "latency_p99_ms": np.percentile(latencies_ms, 99),
                "latency_values_s": latencies,
            })

        # Add per-request TPOT statistics
        if tpots:
            tpots_ms = [t * 1000 for t in tpots]
            metrics.update({
                "tpot_per_request_avg_ms": np.mean(tpots_ms),
                "tpot_per_request_min_ms": np.min(tpots_ms),
                "tpot_per_request_max_ms": np.max(tpots_ms),
            })

        # Print metrics summary
        if use_tqdm:
            print("\n=== Generation Metrics ===")
            if overall_time > 0 and total_tokens > 0:
                print(f"Overall Throughput: {metrics['overall_throughput_tok_per_s']:.2f} tok/s ({total_tokens} tokens in {overall_time:.2f}s)")
            if total_prefill_time > 0:
                print(f"Prefill Throughput: {metrics['prefill_throughput_tok_per_s']:.2f} tok/s")
            if total_decode_time > 0:
                print(f"Decode Throughput: {metrics['decode_throughput_tok_per_s']:.2f} tok/s")
                print(f"TPOT (aggregate): {metrics['tpot_ms']:.2f} ms")
            if tpots:
                print(f"TPOT (per-request): {metrics['tpot_per_request_avg_ms']:.2f} ms (avg), {metrics['tpot_per_request_min_ms']:.2f} ms (min), {metrics['tpot_per_request_max_ms']:.2f} ms (max)")
            if ttfts:
                print(f"TTFT (Time To First Token): {metrics['ttft_avg_ms']:.2f} ms (avg), {metrics['ttft_min_ms']:.2f} ms (min), {metrics['ttft_max_ms']:.2f} ms (max)")
            if latencies:
                print(f"Per-Request Latency: {metrics['latency_avg_ms']:.2f} ms (avg), {metrics['latency_min_ms']:.2f} ms (min), {metrics['latency_max_ms']:.2f} ms (max)")

        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]

        if return_metrics:
            return outputs, metrics
        return outputs
