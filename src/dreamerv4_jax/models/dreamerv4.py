"""
TODO:
- Implement sharding
"""

import sys
import time

import jax
import jax.numpy as jnp
import tokamax
from absl import app, flags, logging
from flax import nnx


class DreamerV4MLP(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dtype: jnp.dtype = jnp.bfloat16,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.gate_proj = nnx.Linear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            param_dtype=dtype,
            rngs=rngs,
        )
        self.up_proj = nnx.Linear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            param_dtype=dtype,
            rngs=rngs,
        )
        self.down_proj = nnx.Linear(
            intermediate_size,
            hidden_size,
            use_bias=False,
            param_dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        intermediate_parallel = tokamax.gated_linear_unit(
            hidden_states,
            jnp.stack([self.gate_proj.kernel[...], self.up_proj.kernel[...]], axis=1),
            activation=jax.nn.swish,
        )
        output = self.down_proj(intermediate_parallel)
        return output


class DreamerV4Attention(nnx.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, num_kv_heads: int, layer_id: int = 0
    ):
        pass


def main(_):
    jax.config.update("jax_platforms", "cuda")
    rngs = nnx.Rngs(jax.random.PRNGKey(0))
    model = DreamerV4MLP(
        hidden_size=1024, intermediate_size=4096, dtype=jnp.bfloat16, rngs=rngs
    )
    hidden_states = rngs.normal((10, 100, 1024), dtype=jnp.bfloat16)

    @nnx.jit
    def forward(hidden_states: jax.Array) -> jax.Array:
        return model(hidden_states)

    autotune_result = tokamax.autotune(forward.lower(hidden_states).lowered)

    start_time = time.time()
    for _ in range(100):
        forward(hidden_states)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")


if __name__ == "__main__":
    app.run(main)
