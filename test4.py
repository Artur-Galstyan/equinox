import jax
import jax.numpy as jnp
from typing import Literal, Optional, Tuple, Union, overload
from jaxtyping import Array, Bool, Float
import math
from equinox.nn import Dropout
from equinox._custom_types import PRNGKey
import functools as ft
from icecream import ic

dropout = None
inference = True
key = None
num_heads = 2

query_multihead = True
key_multihead = True
value_multihead = True
mask = None

key_multihead_dim = 2
value_multihead_dim = 2
query_multihead_dim = 2

p = 3
L = 5

key = jax.random.PRNGKey(42)
qkey, kkey, vkey = jax.random.split(key, 3)
Q = jax.random.uniform(shape=(L, num_heads, p), key=qkey)
K = jax.random.uniform(shape=(L, num_heads, p), key=kkey)
V = jax.random.uniform(shape=(L, num_heads, p), key=vkey)

ic(Q.shape)
ic(K.shape)
ic(V.shape)


def fn(q, k, v):
    return jnp.sum(q) + jnp.sum(k) + jnp.sum(v)


def fn_vmapped(q, k, v):
    return jax.vmap(fn, in_axes=(1, None, None))(q, k, v)


test = fn(Q, K, V)

ic(test)

test2 = jax.vmap(fn, in_axes=(1, 1, 1))(Q, K, V)
ic(test2)


test3 = jax.vmap(fn_vmapped, in_axes=(None, 1, 1))(Q, K, V)
ic(test3)
