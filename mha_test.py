from typing import List, Optional
import equinox as eqx
import jax
import jax.numpy as jnp
from icecream import ic
from jaxtyping import Array, Float, Bool, PRNGKeyArray, PyTree
import math
import functools as ft
import time
import sys 
from tinyshakespeareloader.hamlet import get_data
import optax
# ic.disable()

query_input_dim = 16
query_embedding_dim = 32
key_input_dim = 16
key_embedding_dim = 32
value_input_dim = 16
value_embedding_dim = 32
num_heads = 32
max_seq_len = 8
batch_size = 64
output_dim = 32
kv_multihead_dim = 32
query_multihead_dim = 32

dropout_rate = 0.2

key = jax.random.PRNGKey(42)


def get_positional_encoding(
    n_tokens: int, n_vocab: int
) -> Float[Array, "n_tokens n_vocab"]:
    pos = jnp.arange(n_tokens)[:, jnp.newaxis]
    div_term = jnp.exp(jnp.arange(0, n_vocab, 2) * -(jnp.log(10000.0) / n_vocab))
    # or alternatively: 
    # div_term = 1 / 10000 ** (jnp.arange(0, D, 2) / D)
    # that's closer to the actual notation the authors in the original
    # "Attention is all you need" paper used.
    pos_enc = jnp.zeros((n_tokens, n_vocab))
    pos_enc = pos_enc.at[:, 0::2].set(jnp.sin(pos * div_term))
    pos_enc = pos_enc.at[:, 1::2].set(jnp.cos(pos * div_term))
    return pos_enc


class RMSNorm(eqx.Module):
    weight: Array
    eps: float

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = jnp.ones(dim)

    def _norm(self, x: Array):
        return x * jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)

    def __call__(self, x: Array) -> Array:
        output = self._norm(x)
        return output * self.weight


class Transformer(eqx.Module):
    input_embedding: eqx.nn.Embedding
    
    masked_mhas: List[eqx.nn.MultiheadAttention]

    feedforward: eqx.nn.MLP 
    ff_dropout: eqx.nn.Dropout
    rms_norm: RMSNorm

    output: eqx.nn.Linear
    positional_encoding: Array

    def __init__(
        self,
        n_dims: int,
        n_embd: int,
        n_heads: int,
        key: PRNGKeyArray,
        width_size: int = 32,
        depth: int = 2,
        max_token_size: int = 8,
        n_layers:int = 6,
    ) -> None:
        key, *subkeys = jax.random.split(
            key, 20
        )  # let's just split 20 for now, we'll probably need them later
        self.input_embedding = eqx.nn.Embedding(n_dims, n_embd, key=subkeys[0])

        self.masked_mhas = [
                eqx.nn.MultiheadAttention(
                    num_heads=n_heads,
                    query_size = n_embd,
                    output_size=n_embd,
                    query_multihead_dim=n_heads,
                    kv_multihead_dim=n_heads - 2,
                    key=subkeys[1],
                ) 
                for _ in range(n_layers)
            ] 
        

        self.feedforward = eqx.nn.MLP(
                in_size=n_embd,
                out_size=n_embd,
                width_size=width_size,
                key=subkeys[2],
                depth=depth,
            )
        self.ff_dropout = eqx.nn.Dropout(dropout_rate)
        self.positional_encoding = get_positional_encoding(max_token_size, n_embd)

        self.rms_norm = RMSNorm(dim=n_embd)

        self.output = eqx.nn.Linear(
            in_features=n_embd, out_features=n_dims, key=subkeys[4], use_bias=False
        )

    def __call__(self, x, key: Optional[PRNGKeyArray | None] = None, inference: bool = False):
        mha_key = None
        ff_key = None
        if key is not None:
            key, mha_key, ff_key = jax.random.split(key, 3)
        x = jax.vmap(self.input_embedding)(x)
        x += self.positional_encoding
        masked_mha_output = x
        
        T = x.shape[0]
        mask = jnp.tril(jnp.ones(shape=(T, T))) == 1
        for mha in self.masked_mhas:
            masked_mha_output = mha(query=masked_mha_output, key_=masked_mha_output, value=masked_mha_output, mask=mask, inference=inference, key=mha_key)
        x = self.rms_norm(masked_mha_output + x)  # residual connection
        ff = jax.vmap(self.feedforward)(x)
        if ff_key is not None:
            partial_dropout = ft.partial(self.ff_dropout, key=ff_key, inference=False)
        else:
            partial_dropout = ft.partial(self.ff_dropout, inference=True)
        ff = partial_dropout(ff)
        x = self.rms_norm(ff + x)  # residual connection
        x = jax.vmap(self.output)(x)
        # x = jax.nn.softmax(x) # we don't softmax here, because we want the raw logits for our loss function
        # but you can totally softmax here and inverse that later;
        return x


def generate_text(
    transformer: Transformer,
    decode,
    max_len: int = 100,
    max_seq_len: int = 8,
):
    key = jax.random.PRNGKey(222)
    start_tokens = jnp.array([0] * max_seq_len) # we start with a sequence of zeros

    for i in range(max_len):
        key, subkey = jax.random.split(key)
        logits = transformer(start_tokens, inference=True) # [max_seq_len, vocab_size]; this generates a distribution over the vocabulary
        #ic(logits.shape)
        # we can sample from this distribution to get the next token
        next_token = jax.random.categorical(logits=logits, axis=-1, key=subkey)[-1] # we take the last token
        # next token needs to have the same dimensions as start tokens
        next_token = jnp.expand_dims(next_token, axis=0)
        #ic(next_token)
        start_tokens = jnp.concatenate([start_tokens[1:], next_token]) # we shift the tokens to the left and append the next token
        #ic(int(next_token))
        print(decode([int(next_token)]), end="", flush=True)

    print()    

def main():

    MAX_T = 128
    data = get_data(batch_size=batch_size, block_size=MAX_T)

    train_dataloader, test_dataloader, vocabulary_size, decode = data.train_dataloader, data.test_dataloader, data.vocab_size, data.decode
    key = jax.random.PRNGKey(420)
    INPUT_DIMS = vocabulary_size
    if not INPUT_DIMS:
        raise ValueError("vocabulary size is 0")
    N_EMBD = 384
    N_HEADS = 16
    N_LAYERS = 6

    def loss_fn(transformer: Transformer, x: Array, y: Array, key: Optional[PRNGKeyArray | None] = None, inference: bool = False):
        partial_transformer = ft.partial(transformer, key=key, inference=inference)
        logits = eqx.filter_vmap(partial_transformer)(x)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)

        return jnp.mean(loss)

    def evaluate(transformer: Transformer, test_dataloader):
        loss = 0
        jitted_loss_fn = eqx.filter_jit(loss_fn)
        for x, y in test_dataloader:
            x = jnp.array(x.numpy())
            y = jnp.array(y.numpy())
            loss += jitted_loss_fn(transformer, x, y, inference=True)

        return loss / len(test_dataloader)

    @eqx.filter_jit
    def step(
        transformer: PyTree,
        opt_state: optax.OptState,
        optimiser: optax.GradientTransformation,
        x: Array,
        y: Array,
        key: PRNGKeyArray,
    ):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(transformer, x, y, key=key)
        updates, opt_state = optimiser.update(grads, opt_state, transformer)
        transformer = eqx.apply_updates(transformer, updates)
        return transformer, opt_state, loss

    transformer = Transformer(
        n_dims=INPUT_DIMS, n_embd=N_EMBD, n_heads=N_HEADS, key=key, max_token_size=MAX_T, n_layers=N_LAYERS
    )
    # start_loss = evaluate(transformer, test_dataloader)
    # print(f"{start_loss=}")
    optimiser = optax.adamw(learning_rate=3e-4)
    opt_state = optimiser.init(eqx.filter(transformer, eqx.is_inexact_array))
    generate_text(transformer, decode, max_seq_len=MAX_T)
    ic("starting training")
    start_time = time.time()
    for i, (x, y) in enumerate(train_dataloader):
        x = jnp.array(x.numpy())
        y = jnp.array(y.numpy())
        key, subkey = jax.random.split(key)
        transformer, opt_state, loss = step(transformer, opt_state, optimiser, x, y, key=subkey)
        if i % 100 == 0:
            eval_loss = evaluate(transformer, test_dataloader)
            ic(i, loss, eval_loss)
        if i % 1000 == 0:
            generate_text(transformer, decode, max_seq_len=MAX_T)
        if i == 5000:
            ic("early stopping")
    end_time = time.time()
    ic("done training")
    ic("training time:", end_time - start_time)
    generate_text(transformer, decode)
    ic(evaluate(transformer, test_dataloader))


if __name__ == "__main__":
    ic.enable()
    main()
