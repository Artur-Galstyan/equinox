import math
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from equinox._custom_types import PRNGKey
from equinox.nn import Dropout


def dot_product_attention_weights(
    query: Float[Array, "q_seq qk_size"],
    key: Float[Array, "kv_seq qk_size"],
    mask: Optional[Bool[Array, "q_seq kv_seq"]] = None,
) -> Float[Array, "q_seq kv_seq"]:
    query = query / math.sqrt(query.shape[-1])
    print(f"{query.shape=}, {key.shape=}")
    logits = jnp.einsum("sd,Sd->sS", query, key)
    if mask is not None:
        if mask.shape != logits.shape:
            raise ValueError(
                f"mask must have shape (query_seq_length, "
                f"kv_seq_length)=({query.shape[0]}, "
                f"{key.shape[0]}). Got {mask.shape}."
            )
        logits = jnp.where(mask, logits, jnp.finfo(logits.dtype).min)

    return jax.nn.softmax(logits, axis=-1)


def dot_product_attention(
    query: Float[Array, "q_seq qk_size"],
    key_: Float[Array, "kv_seq qk_size"],
    value: Float[Array, "kv_seq v_size"],
    mask: Optional[Bool[Array, "q_seq kv_seq"]] = None,
    dropout: Optional[Dropout] = None,
    *,
    key: Optional[PRNGKey] = None,
    inference: Optional[bool] = None,
) -> Float[Array, "q_seq v_size"]:
    weights = dot_product_attention_weights(query, key_, mask)
    print(f"{weights.shape=}")
    if dropout is not None:
        weights = dropout(weights, key=key, inference=inference)
    attn = jnp.einsum("sS,Sd->sd", weights, value)
    return attn


# Define the function to be applied at each step of the scan
def step_fn(carry, i):
    # we step from 0...d, and i is the current iteration
    query_heads, key_heads, value_heads, result = carry
    # query_heads.shape =       L n p   (in_axis=1)
    # key_heads.shape =         L d p
    # value_heads.shape =       L d p
    print(f"{result.shape=}")
    # plug out the i-th dimension to add to the
    # cumulative attention
    key_slice = key_heads[:, i, :]  # shape = L p (in_axis=None)
    value_slice = value_heads[:, i, :]  # shape = L p (in_axis=None)

    dpa = jax.vmap(
        lambda q, k, v: dot_product_attention(q, k, v),
        in_axes=(1, None, None),
        out_axes=1,
    )(query_heads, key_slice, value_slice)
    print(f"{dpa.shape=}")  # = L n p (same as input query_heads)
    # Accumulate the result
    new_result = result + dpa

    return (query_heads, key_heads, value_heads, new_result), dpa


def vmapped_fn(query_heads, key_slice, value_slice):
    dpa = jax.vmap(
        lambda q, k, v: dot_product_attention(q, k, v),
        in_axes=(1, None, None),
        out_axes=1,
    )(query_heads, key_slice, value_slice)
    return dpa


# Initialize matrices A and B
L = 10
n = 2
p = 4
d = 5

key = jax.random.PRNGKey(22)
key, subkey, subkey1, subkey2 = jax.random.split(key, 4)

A = jax.random.uniform(shape=(L, n, p), key=key)
B = jax.random.uniform(shape=(L, d, p), key=subkey)
C = jax.random.uniform(shape=(L, d, p), key=subkey1)

# Initialize carry state
init_carry = (A, jnp.zeros((L, n, p)))
print(f"{A.shape=}")
print(f"{B.shape=}")
print(f"{C.shape=}")

# Perform the scan
# _, result = jax.lax.scan(step_fn, init_carry, B.swapaxes(0, 1))
init_carry = (A, B, C, jnp.zeros((L, n, p)))
_, result = jax.lax.scan(step_fn, init_carry, jnp.arange(d))
print(f"result after scan {result.shape=}")
result = jnp.einsum("ijkl->jkl", result)
print(f"result after einsum {result.shape=}")
# Take the mean over the d dimension
result_mean = result / d

print(f"result mean {result_mean.shape=}")
print(result_mean.shape)

print("=====================")
result = jax.vmap(vmapped_fn, in_axes=(None, 1, 1), out_axes=1)(A, B, C)

print(f"result after vmap {result.shape=}")
result_sum = jnp.sum(result, axis=1)

# Taking the mean over the d dimension
result_mean = result_sum / d

print(f"{result_mean.shape=}")
