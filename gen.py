import os
import requests
from jax.config import config
import socket

tpu_address = socket.gethostbyname(socket.gethostname())
url = f'http://{tpu_address}:8475/requestversion/tpu_driver0.1_dev20210607'
print(url)
requests.post(url)

# The following is required to use TPU Driver as JAX's backend.
config.FLAGS.jax_xla_backend = "tpu_driver"
config.FLAGS.jax_backend_target = "grpc://" + tpu_address

import time

import jax
from jax.experimental import maps
import numpy as np
import optax
import transformers

from mesh_transformer.checkpoint import read_ckpt_lowmem
from mesh_transformer.sampling import nucleaus_sample
from mesh_transformer.transformer_shard import CausalTransformer

params = {
  "layers": 28,
  "d_model": 4096,
  "n_heads": 16,
  "n_vocab": 50400,
  "norm": "layernorm",
  "pe": "rotary",
  "pe_rotary_dims": 64,

  "seq": 2048,
  "cores_per_replica": 8,
  "per_replica_batch": 32,
}

per_replica_batch = params["per_replica_batch"]
cores_per_replica = params["cores_per_replica"]
seq = params["seq"]


params["sampler"] = nucleaus_sample

# here we "remove" the optimizer parameters from the model (as we don't need them for inference)
params["optimizer"] = optax.scale(0)

mesh_shape = (jax.device_count() // cores_per_replica, cores_per_replica)
devices = np.array(jax.devices()).reshape(mesh_shape)

maps.thread_resources.env = maps.ResourceEnv(maps.Mesh(devices, ('dp', 'mp')))

tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')

total_batch = per_replica_batch * jax.device_count() // cores_per_replica

network = CausalTransformer(params)

network.state = read_ckpt_lowmem(network.state, "step_1/", devices.shape[1])

network.state = network.move_xmap(network.state, np.zeros(cores_per_replica))




def infer(context, top_p=0.9, temp=1.0, gen_len=512):
    tokens = tokenizer.encode(context)

    provided_ctx = len(tokens)
    pad_amount = seq - provided_ctx

    padded_tokens = np.pad(tokens, ((pad_amount, 0),)).astype(np.uint32)
    batched_tokens = np.array([padded_tokens] * total_batch)
    length = np.ones(total_batch, dtype=np.uint32) * len(tokens)

    start = time.time()
    output = network.generate(batched_tokens, length, gen_len, {"top_p": np.ones(total_batch) * top_p, "temp": np.ones(total_batch) * temp})

    samples = []
    decoded_tokens = output[1][0]

    for o in decoded_tokens[:, :, 0]:
      samples.append(f"\033[1m{context}\033[0m{tokenizer.decode(o)}")

    print(f"completion done in {time.time() - start:06}s")
    return samples

# print(infer("EleutherAI is")[0])
#
#
# top_p = 0.9 #@param {type:"slider", min:0, max:1, step:0.1}
# temp = .9 #@param {type:"slider", min:0, max:1, step:0.1}
#
#
# for i in range(10):
#     context = """In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."""
#     print(infer(top_p=top_p, temp=temp, gen_len=512, context=context)[0])
