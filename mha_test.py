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
import wandb


# ic.disable()


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
    feedforwards: List[eqx.nn.MLP]

    ff_dropout: eqx.nn.Dropout
    rms_norm: RMSNorm

    output: eqx.nn.Linear
    positional_encoding: Array

    dropout_rate: float = eqx.static_field()
    n_layers: int = eqx.static_field()
    def __init__(
        self,
        n_dims: int,
        n_embd: int,
        n_heads: int,
        key: PRNGKeyArray,
        width_size: int = 128,
        depth: int = 8,
        max_token_size: int = 8,
        n_layers: int = 6,
        dropout_rate: float = 0.2,
        query_multihead_dim: int = 8,
        kv_multihead_dim: int = 6,
        
    ) -> None:
        key, *subkeys = jax.random.split(
            key, 20
        )  # let's just split 20 for now, we'll probably need them later
        self.input_embedding = eqx.nn.Embedding(n_dims, n_embd, key=subkeys[0])

        self.masked_mhas = [
            eqx.nn.MultiheadAttention(
                num_heads=n_heads,
                query_size=n_embd,
                output_size=n_embd,
                query_multihead_dim=query_multihead_dim,
                kv_multihead_dim=kv_multihead_dim,
                key=subkeys[1],
            )
            for _ in range(n_layers)
        ]

        self.feedforwards = [
            eqx.nn.MLP(
                in_size=n_embd,
                out_size=n_embd,
                width_size=width_size,
                key=subkeys[2],
                depth=depth,
            )
            for _ in range(n_layers)
        ]

        self.ff_dropout = eqx.nn.Dropout(dropout_rate)
        self.positional_encoding = get_positional_encoding(max_token_size, n_embd)

        self.rms_norm = RMSNorm(dim=n_embd)

        self.output = eqx.nn.Linear(
            in_features=n_embd, out_features=n_dims, key=subkeys[4], use_bias=False
        )

        self.dropout_rate = dropout_rate
        self.n_layers = n_layers

    def __call__(
        self, x, key: Optional[PRNGKeyArray | None] = None, inference: bool = False
    ):
        mha_key = None
        ff_key = None
        if key is not None:
            key, mha_key, ff_key = jax.random.split(key, 3)
        x = jax.vmap(self.input_embedding)(x)
        x += self.positional_encoding

        T = x.shape[0]
        mask = jnp.tril(jnp.ones(shape=(T, T))) == 1

        for i in range(self.n_layers):
            masked_mha_output = x
            if key is not None:
                key, mha_key, ff_key = jax.random.split(key, 3)
            mha = self.masked_mhas[i]
            feedforward = self.feedforwards[i]

            masked_mha_output = mha(
                query=masked_mha_output,
                key_=masked_mha_output,
                value=masked_mha_output,
                mask=mask,
                inference=inference,
                key=mha_key,
            )
            masked_mha_output = self.rms_norm(masked_mha_output)
            masked_mha_output = masked_mha_output + x  # residual connection
            
            ff = jax.vmap(feedforward)(masked_mha_output)
            if ff_key is not None:
                partial_dropout = ft.partial(self.ff_dropout, key=ff_key, inference=False)
            else:
                partial_dropout = ft.partial(self.ff_dropout, inference=True)
            ff = partial_dropout(ff)
            x = self.rms_norm(ff + masked_mha_output)  # residual connection
        x = jax.vmap(self.output)(x)
        # x = jax.nn.softmax(x) # we don't softmax here, because we want the raw logits for our loss function
        # but you can totally softmax here and inverse that later;
        return x


def generate_text(
    transformer: Transformer,
    decode,
    key,
    max_len: int = 100,
    max_seq_len: int = 8,
    
):
    start_tokens = jnp.array([0] * max_seq_len)  # we start with a sequence of zeros
    generated_string = ""
    for _ in range(max_len):
        key, subkey = jax.random.split(key)
        logits = transformer(
            start_tokens, inference=True
        )  # [max_seq_len, vocab_size]; this generates a distribution over the vocabulary
        # ic(logits.shape)
        # we can sample from this distribution to get the next token
        next_token = jax.random.categorical(logits=logits, axis=-1, key=subkey)[
            -1
        ]  # we take the last token
        # next token needs to have the same dimensions as start tokens
        next_token = jnp.expand_dims(next_token, axis=0)
        # ic(next_token)
        start_tokens = jnp.concatenate(
            [start_tokens[1:], next_token]
        )  # we shift the tokens to the left and append the next token
        # ic(int(next_token))
        this_decoded = decode([int(next_token)])
        generated_string += this_decoded
        print(decode([int(next_token)]), end="", flush=True)
    return generated_string


def main():
    N_EMBD = 384
    N_LAYERS = 6
    MAX_T = 256
    LEARNING_RATE = 3e-4
    DROPOUT_RATE = 0.2
    WIDTH_SIZE = 128
    BATCH_SIZE = 64
    DEPTH = 4
    N_HEADS = 16
    QUERY_MULTIHEAD_DIM = N_HEADS
    KV_MULTIHEAD_DIM = N_HEADS - 2
    SHUFFLE = False
    key = jax.random.PRNGKey(42)
    data = get_data(batch_size=BATCH_SIZE, block_size=MAX_T, shuffle=SHUFFLE)

    train_dataloader, test_dataloader, vocabulary_size, decode = (
        data.train_dataloader,
        data.test_dataloader,
        data.vocab_size,
        data.decode,
    )
    key = jax.random.PRNGKey(420)
    INPUT_DIMS = vocabulary_size
    if not INPUT_DIMS:
        raise ValueError("vocabulary size is 0")

    def loss_fn(
        transformer: Transformer,
        x: Array,
        y: Array,
        key: Optional[PRNGKeyArray | None] = None,
        inference: bool = False,
    ):
        partial_transformer = ft.partial(transformer, key=key, inference=inference)
        logits = eqx.filter_vmap(partial_transformer)(x)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)

        return jnp.mean(loss)

    def evaluate(transformer: Transformer, test_dataloader, key: PRNGKeyArray):
        loss = 0
        jitted_loss_fn = eqx.filter_jit(loss_fn)
        for x, y in test_dataloader:
            x = jnp.array(x.numpy())
            y = jnp.array(y.numpy())
            loss += jitted_loss_fn(transformer, x, y, inference=True, key=key)

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
        n_dims=INPUT_DIMS,
        n_embd=N_EMBD,
        n_heads=N_HEADS,
        max_token_size=MAX_T,
        n_layers=N_LAYERS,
        dropout_rate=DROPOUT_RATE,
        depth=DEPTH,
        width_size=WIDTH_SIZE,
        query_multihead_dim=QUERY_MULTIHEAD_DIM,
        kv_multihead_dim=KV_MULTIHEAD_DIM,
        key=key,
    )
    # start_loss = evaluate(transformer, test_dataloader)
    # print(f"{start_loss=}")
    key, subkey = jax.random.split(key)
    optimiser = optax.adamw(learning_rate=LEARNING_RATE)
    opt_state = optimiser.init(eqx.filter(transformer, eqx.is_inexact_array))
    generate_text(transformer, decode, max_seq_len=MAX_T, key=subkey)

    # wandb.init(
    # # set the wandb project where this run will be logged
    #     project="MultiheadAttention Test",
    #     # track hyperparameters and run metadata
    #     config={
    #     "learning_rate": learning_rate,
    #     "architecture": "transformer",
    #     "dataset": "Tiny Shakespeare",
    #     "training_steps": 5000
    #     }
    # )

    ic("starting training")
    start_time = time.time()
    for i, (x, y) in enumerate(train_dataloader):
        x = jnp.array(x.numpy())
        y = jnp.array(y.numpy())
        key, subkey, sk3, sk4 = jax.random.split(key, 4)
        transformer, opt_state, loss = step(
            transformer, opt_state, optimiser, x, y, key=subkey
        )
        if i % 100 == 0:
            eval_loss = evaluate(transformer, test_dataloader, key=sk4)
            # wandb.log({"train_loss": loss, "eval_loss": eval_loss})
            ic(f"step {i}: train loss: {loss}, eval loss: {eval_loss}")
        if i % 1000 == 0:
            generated_text = generate_text(transformer, decode, max_seq_len=MAX_T, key=sk3)
            ic(generated_text)
            # wandb.log({"generated_text": generated_text})
        if i == 5000:
            ic("early stopping")
    end_time = time.time()
    ic("done training")
    ic("training time:", end_time - start_time)
    key, subkey, sk2 = jax.random.split(key, 3)
    generate_text(transformer, decode, max_seq_len=MAX_T, key=subkey)
    ic(evaluate(transformer, test_dataloader, sk2))


if __name__ == "__main__":
    ic.enable()
    main()
