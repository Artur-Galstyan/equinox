import jax
import jax.numpy as jnp
from typing import Literal, Optional, Tuple, Union, overload
from jaxtyping import Array, Bool, Float
import math
from equinox.nn import Dropout
from equinox._custom_types import PRNGKey
import functools as ft
from icecream import ic
import numpy as np


dropout = None
inference = True
key = None
num_heads = 2

query_multihead = True
key_multihead = False
value_multihead = False
mask = None

key_multihead_dim = 1
value_multihead_dim = 1
query_multihead_dim = 2

p = 3
L = 5


def dot_product_attention_weights(
    query: Float[Array, "q_seq qk_size"],
    key: Float[Array, "kv_seq qk_size"],
    mask: Optional[Bool[Array, "q_seq kv_seq"]] = None,
) -> Float[Array, "q_seq kv_seq"]:
    query = query / math.sqrt(query.shape[-1])
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
    print(f"{query.shape=}, {key_.shape=}, {value.shape=}")
    weights = dot_product_attention_weights(query, key_, mask)
    if dropout is not None:
        weights = dropout(weights, key=key, inference=inference)
    attn = jnp.einsum("sS,Sd->sd", weights, value)
    return attn


def vmapped_fn(query_heads, key_heads, value_heads, dropout, inference, mask, keys):
    attn_fn = ft.partial(
        dot_product_attention, dropout=dropout, inference=inference, key=keys, mask=mask
    )
    print(f"{query_heads.shape=}, {key_heads.shape=}, {value_heads.shape=}")
    # Batch `keys` down its first axis as it is passed as a keyword argument.

    dpa = jax.vmap(
        lambda q, k, v: attn_fn(q, k, v),
        in_axes=(1, None, None),
        out_axes=1,
    )(query_heads, key_heads, value_heads)
    return dpa


key = jax.random.PRNGKey(42)
qkey, kkey, vkey = jax.random.split(key, 3)
Q = jax.random.uniform(shape=(L, num_heads, p), key=qkey)
K = jax.random.uniform(shape=(L, p), key=kkey)
V = jax.random.uniform(shape=(L, p), key=vkey)

print("========================= START =========================")
key = None

attn_fn = ft.partial(dot_product_attention, dropout=dropout, inference=inference)
keys = None if key is None else jax.random.split(key, num_heads)
in_axes = (
    1 if query_multihead else None,
    1 if key_multihead else None,
    1 if value_multihead else None,
    0 if mask is not None and mask.ndim == 3 else None,
)
ic(in_axes)
attn = jax.vmap(attn_fn, in_axes=in_axes, out_axes=1, axis_size=num_heads)(
    Q, K, V, mask, key=keys
)

print(f"{attn.shape=}")


print("========================= VMAPPED =========================")
key = jax.random.PRNGKey(42)
qkey, kkey, vkey = jax.random.split(key, 3)
Q = jax.random.uniform(shape=(L, num_heads, p), key=qkey)
K = jax.random.uniform(shape=(L, key_multihead_dim, p), key=kkey)
V = jax.random.uniform(shape=(L, value_multihead_dim, p), key=vkey)
pt_vmapped_fn = ft.partial(
    vmapped_fn,
    dropout=dropout,
    inference=inference,
    mask=mask,
    keys=keys,
)

# Outer VMAP
result = jax.vmap(
    pt_vmapped_fn,
    in_axes=(None, 1, 1),
)(Q, K, V)

ic(attn, attn.shape)
ic(result, result.shape)

t1 = np.array(result.ravel())
t1 = np.round(t1, 5)
ic(len(t1))
t2 = np.array(attn.ravel())
t2 = np.round(t2, 5)
ic(len(t2))
ic(t1)
ic(t2)
t1_set = set(t1)
t2_set = set(t2)

ic(t2_set.issubset(t1_set))


"""
print(f"{result.shape=}")
result_sum = jnp.sum(result, axis=1)
print(f"{result_sum.shape=}")

result = result_sum / num_heads


print(f"{result.shape=}")

print(f"{attn=}")
print(f"{result=}")

"""
print(f"{jnp.allclose(attn, result)=}")

"""
print(f"============ ISOLATED TESTING ============")


isolated_result = vmapped_fn(Q, K[:, 0, :], V[:, 0, :], dropout, inference, None, None)
isolated_result2 = vmapped_fn(Q, K[:, 1, :], V[:, 1, :], dropout, inference, None, None)
isolated_result_stacked = jnp.stack([isolated_result, isolated_result2], axis=1)
isolated_result_vmap = jax.vmap(
    lambda q, k, v: vmapped_fn(q, k, v, dropout, inference, None, None),
    in_axes=(None, 1, 1),
    out_axes=1,
)(Q, K, V)

print(f"{jnp.allclose(isolated_result_stacked, isolated_result_vmap)=}")

print(f"Isolated result: {isolated_result.shape=}")
isolated_attn1 = dot_product_attention(
    Q[:, 0, :], K[:, 0, :], V[:, 0, :], mask=mask, dropout=dropout, inference=inference
)
isolated_attn2 = dot_product_attention(
    Q[:, 1, :], K[:, 0, :], V[:, 0, :], mask=mask, dropout=dropout, inference=inference
)
print(f"Isolated attn: {isolated_attn1.shape=}")
combined_attn = jnp.stack([isolated_attn1, isolated_attn2], axis=1)
try:
    print(f"{jnp.allclose(isolated_result, combined_attn)=}")
except Exception as e:
    print("Error")
"""
