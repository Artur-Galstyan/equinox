import time

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def benchmark_fn(fn, args, warmup=5, repeats=20):
    for _ in range(warmup):
        out = fn(*args)
        jax.tree.map(lambda x: x.block_until_ready(), out)

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn(*args)
        jax.tree.map(lambda x: x.block_until_ready(), out)
        end = time.perf_counter()
        times.append(end - start)
    return np.array(times)


def make_forward_fn(model):
    @eqx.filter_jit
    def forward(q, k, v):
        return model(q, k, v)

    return forward


def make_forward_backward_fn(model):
    @eqx.filter_jit
    @eqx.filter_value_and_grad
    def forward_backward(model, q, k, v):
        return jnp.sum(model(q, k, v))

    def run(q, k, v):
        return forward_backward(model, q, k, v)

    return run


def run_benchmark():
    rng = jax.random.key(42)

    num_heads = 16
    query_size = 1024
    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768]

    mha_existing = eqx.nn.MultiheadAttention(
        num_heads=num_heads, query_size=query_size, key=rng
    )
    mha_cudnn = eqx.nn.MultiheadAttention(
        num_heads=num_heads,
        query_size=query_size,
        key=rng,
        use_flash_attn=True,
        dtype=jnp.bfloat16,
        implementation="cudnn",
    )

    fwd_existing = make_forward_fn(mha_existing)
    fwd_cudnn = make_forward_fn(mha_cudnn)
    fwd_bwd_existing = make_forward_backward_fn(mha_existing)
    fwd_bwd_cudnn = make_forward_backward_fn(mha_cudnn)

    results = {
        "fwd_existing": [],
        "fwd_cudnn": [],
        "fwd_bwd_existing": [],
        "fwd_bwd_cudnn": [],
    }
    completed_seq_lengths = []

    for seq_len in seq_lengths:
        print(f"\nBenchmarking seq_len={seq_len}...")

        q_f32 = jax.random.normal(rng, (seq_len, query_size), dtype=jnp.float32)
        k_f32 = jax.random.normal(rng, (seq_len, query_size), dtype=jnp.float32)
        v_f32 = jax.random.normal(rng, (seq_len, query_size), dtype=jnp.float32)

        q_bf16 = q_f32.astype(jnp.bfloat16)
        k_bf16 = k_f32.astype(jnp.bfloat16)
        v_bf16 = v_f32.astype(jnp.bfloat16)

        try:
            times = benchmark_fn(fwd_existing, (q_f32, k_f32, v_f32))
            results["fwd_existing"].append(np.median(times) * 1000)
            print(f"  fwd existing: {np.median(times) * 1000:.2f}ms")
        except Exception as e:
            print(f"  fwd existing: OOM or error ({e})")
            results["fwd_existing"].append(np.nan)

        try:
            times = benchmark_fn(fwd_cudnn, (q_bf16, k_bf16, v_bf16))
            results["fwd_cudnn"].append(np.median(times) * 1000)
            print(f"  fwd cudnn:    {np.median(times) * 1000:.2f}ms")
        except Exception as e:
            print(f"  fwd cudnn:    OOM or error ({e})")
            results["fwd_cudnn"].append(np.nan)

        try:
            times = benchmark_fn(fwd_bwd_existing, (q_f32, k_f32, v_f32))
            results["fwd_bwd_existing"].append(np.median(times) * 1000)
            print(f"  fwd+bwd existing: {np.median(times) * 1000:.2f}ms")
        except Exception as e:
            print(f"  fwd+bwd existing: OOM or error ({e})")
            results["fwd_bwd_existing"].append(np.nan)

        try:
            times = benchmark_fn(fwd_bwd_cudnn, (q_bf16, k_bf16, v_bf16))
            results["fwd_bwd_cudnn"].append(np.median(times) * 1000)
            print(f"  fwd+bwd cudnn:    {np.median(times) * 1000:.2f}ms")
        except Exception as e:
            print(f"  fwd+bwd cudnn:    OOM or error ({e})")
            results["fwd_bwd_cudnn"].append(np.nan)

        completed_seq_lengths.append(seq_len)

    return completed_seq_lengths, results


def plot_results(seq_lengths, results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    c_existing = "#e74c3c"
    c_cudnn = "#2ecc71"

    ax = axes[0, 0]
    ax.plot(
        seq_lengths,
        results["fwd_existing"],
        "o-",
        color=c_existing,
        label="Existing (vmap)",
        linewidth=2,
        markersize=6,
    )
    ax.plot(
        seq_lengths,
        results["fwd_cudnn"],
        "o-",
        color=c_cudnn,
        label="cuDNN Flash",
        linewidth=2,
        markersize=6,
    )
    ax.set_xlabel("Sequence Length", fontsize=11)
    ax.set_ylabel("Median Time (ms)", fontsize=11)
    ax.set_title("Forward Pass", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(
        seq_lengths,
        results["fwd_bwd_existing"],
        "o-",
        color=c_existing,
        label="Existing (vmap)",
        linewidth=2,
        markersize=6,
    )
    ax.plot(
        seq_lengths,
        results["fwd_bwd_cudnn"],
        "o-",
        color=c_cudnn,
        label="cuDNN Flash",
        linewidth=2,
        markersize=6,
    )
    ax.set_xlabel("Sequence Length", fontsize=11)
    ax.set_ylabel("Median Time (ms)", fontsize=11)
    ax.set_title("Forward + Backward Pass", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, alpha=0.3)

    existing_fwd = np.array(results["fwd_existing"])
    cudnn_fwd = np.array(results["fwd_cudnn"])
    existing_fwd_bwd = np.array(results["fwd_bwd_existing"])
    cudnn_fwd_bwd = np.array(results["fwd_bwd_cudnn"])

    ax = axes[1, 0]
    fwd_speedup = existing_fwd / cudnn_fwd
    ax.plot(seq_lengths, fwd_speedup, "o-", color=c_cudnn, linewidth=2, markersize=6)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Sequence Length", fontsize=11)
    ax.set_ylabel("Speedup (×)", fontsize=11)
    ax.set_title("Forward Speedup (cuDNN / Existing)", fontsize=13, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    fwd_bwd_speedup = existing_fwd_bwd / cudnn_fwd_bwd
    ax.plot(
        seq_lengths, fwd_bwd_speedup, "o-", color=c_cudnn, linewidth=2, markersize=6
    )
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Sequence Length", fontsize=11)
    ax.set_ylabel("Speedup (×)", fontsize=11)
    ax.set_title(
        "Forward+Backward Speedup (cuDNN / Existing)", fontsize=13, fontweight="bold"
    )
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("mha_benchmark.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved to mha_benchmark.png")


if __name__ == "__main__":
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")

    seq_lengths, results = run_benchmark()
    plot_results(seq_lengths, results)
