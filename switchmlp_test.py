# %%
%load_ext autoreload
%autoreload 2
import torch
import os
os.environ['LOCAL_RANK'] = "0"
#os.environ["MASTER_ADDR"] = "172.17.0.3"
#os.environ["MASTER_ADDR"] = "172.17.0.1"
os.environ["MASTER_ADDR"] = "localhost"
#172.17.0.1
master_ip = os.getenv('MASTER_ADDR', 'localhost')
print("MASTER IP", master_ip) 

from megatron.core.transformer.switch_mlp import SwitchMLP
from tests.unit_tests.test_utilities import Utils
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_layer_specs import gpt_layer_with_transformer_engine_spec_moe
print("DONE")
# %%


Utils.initialize_model_parallel(1,1)
model_parallel_cuda_manual_seed(123)
print("done intializing")
# %%

transformer_config = TransformerConfig(num_layers=2, hidden_size=12, num_attention_heads=4, num_moe_experts= 2, use_cpu_initialization=True, expert_model_parallel_size=1)
switch_mlp = SwitchMLP(transformer_config,
                gpt_layer_with_transformer_engine_spec_moe.submodules.mlp.submodules)

# %%
#Utils.destroy_model_parallel()


def test_gpu_forward():
    switch_mlp.cuda()
    # [sequence length, batch size, hidden size]
    hidden_states = torch.randn((32, 2, switch_mlp.config.hidden_size))
    hidden_states = hidden_states.cuda()
    output, output_bias = switch_mlp(hidden_states)
    assert output.shape[0] == 32
    assert output.shape[1] == 2
    assert output.shape[2] == switch_mlp.config.hidden_size
    assert output_bias.shape[2] == switch_mlp.config.hidden_size
    assert output.dtype == torch.float32
    assert output.device.type == 'cuda'
    assert output_bias.device.type == 'cuda'
    print("SUCCES!")


test_gpu_forward()
# %%
