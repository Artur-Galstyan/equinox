import jax
import jax.numpy as jnp

import equinox as eqx
from equinox.nn import MultiheadAttention

key = jax.random.PRNGKey(22)
key, *subkeys = jax.random.split(key, 12)


state_len = 10
num_heads = 2


query_size = 8
k_size = 8
v_size = 8
d_queries = 4

query = jax.random.uniform(key=subkeys[0], shape=(state_len, query_size))
key_ = jax.random.uniform(key=subkeys[1], shape=(state_len, k_size))
value = jax.random.uniform(key=subkeys[2], shape=(state_len, v_size))


print("====================")
mha = MultiheadAttention(
    num_heads=num_heads,
    query_size=query_size,
    key=key,
    inference=True
    # key_multihead_dim=1,
    # value_multihead_dim=1,
    # key_multihead=False,
    # value_multihead=False,
)


output = (mha(query, key_, value),)  # mask=jnp.ones((state_len, state_len)))
# output, state = mha(output, key_, value, mask="causal", state=state)
# output, state = mha(output, key_, value, mask="causal", state=state)

# print(output)
