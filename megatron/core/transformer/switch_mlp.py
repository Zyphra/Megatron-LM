# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.parallel_state import (
    get_tensor_and_expert_parallel_group,
    get_tensor_model_parallel_group,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig

from .mlp import MLP, MLPSubmodules


def sinkhorn(cost, tol=0.0001):
    "Sinkhorn based MoE routing function"
    cost = torch.exp(cost)
    d0 = torch.ones(cost.size(0), device=cost.device, dtype=cost.dtype)
    d1 = torch.ones(cost.size(1), device=cost.device, dtype=cost.dtype)

    eps = 0.00000001
    error = 1e9
    d1_old = d1
    num_iterations = 0
    while error > tol:
        d0 = (1 / d0.size(0)) * 1 / (torch.sum(d1 * cost, 1) + eps)
        d1 = (1 / d1.size(0)) * 1 / (torch.sum(d0.unsqueeze(1) * cost, 0) + eps)
        error = torch.mean(torch.abs(d1_old - d1))
        d1_old = d1
        num_iterations +=1 
    print("SINKHORN NUM ITERS: ", num_iterations)
    return d1 * cost * d0.unsqueeze(1)

# sinkhorn seems at least reasonable. We can do topk on top of sinkhorn if we want to


class SwitchMLP(MegatronModule):
    """
    Top-1 Mixture of Experts Layer. Routes input to one of N MLP "experts"
    Curently supports Sinkhorn based expert routing.
    """

    def __init__(self, config: TransformerConfig, submodules: MLPSubmodules):
        super().__init__(config=config)
        self.config: TransformerConfig = config

        self.router = torch.nn.Linear(self.config.hidden_size, self.config.num_moe_experts)
        self.add_bias = config.add_bias_linear
        self.sequence_parallel = config.sequence_parallel
        self.route_algo = sinkhorn
        self.router_activation = torch.sigmoid
        self.expert_parallel_size = parallel_state.get_expert_model_parallel_world_size()
        self.expert_parallel_size = 1
        print("EXPERT PARALLEL: ",self.expert_parallel_size)
        print("NUM EXPERTS: ", self.config.num_moe_experts)
        assert self.config.num_moe_experts % self.expert_parallel_size == 0
        self.num_local_experts = self.config.num_moe_experts // self.expert_parallel_size
        local_expert_indices_offset = (
            parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
        )
        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]

        self.local_experts = torch.nn.ModuleList()
        for _ in range(self.num_local_experts):
            expert = MLP(self.config, submodules, is_expert=True)
            self.local_experts.append(expert)

    def gather_indices(self, local_indices):
        """ Gather tensors and concatenate along the first dimension."""
        group = get_tensor_and_expert_parallel_group()
        world_size = torch.distributed.get_world_size(group=group)
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return local_indices

        dim_size = list(local_indices.size())
        dim_size[0] = dim_size[0] * world_size

        # TODO pre allocate memory
        output = torch.empty(
            dim_size, dtype=local_indices.dtype, device=torch.cuda.current_device()
        )
        torch.distributed._all_gather_base(output, local_indices.contiguous(), group=group)
        return output

    def forward(self, hidden_states):
        print("IN FORWARD")
        hidden_shape = hidden_states.shape
        print("HIDDEN SHAPE: ", hidden_shape)
        route = self.router(hidden_states)
        print("ROUTE: ", route.shape)
        route = route.view(-1, self.config.num_moe_experts)
        print("ROUTE AFTER RESHAPE: ", route.shape)
        if self.training:
            with torch.no_grad():
                norm_route = self.route_algo(
                    route.detach().to(dtype=torch.float32)
                )  # explicit fp32 conversion for stability
                print("NORM ROUTE: ", norm_route.shape)
                _, max_ind = torch.max(norm_route, dim=1)
            route = self.router_activation(route)
            #print("after router activation: ", route.shape)
            max_prob = route[torch.arange(route.size(0)), max_ind] # so why do we even sigmoid it makes no difference?
            #print("max prob: ", max_prob.shape)
            #route = self.router_activation(route)
            #max_prob = route[torch.arange(route.size(0)), max_ind]
            #print("max prob after sigmoid: ", max_prob)
        else:
            route = self.router_activation(route)
            max_prob, max_ind = torch.max(route, dim=1)
        
        print("max ind: ", max_ind.shape)
        print("max ind: ", max_ind) # for each batch, for each sequence element, we select a token
        max_prob = torch.unsqueeze(max_prob, 1)
        hidden_states = hidden_states.view(-1, hidden_shape[-1])
        print("hidden_states after review: ", hidden_states.shape)

        if self.sequence_parallel or (self.expert_parallel_size > 1):
            global_hidden_states = tensor_parallel.gather_from_sequence_parallel_region_to_moe(
                hidden_states
            )
            print("GLOBAL HIDDEN STATES: ", global_hidden_states.shape)
            
            global_indices = self.gather_indices(max_ind)
            print("GLOBAL INDICES: ", global_indices.shape)
        else:
            global_hidden_states = hidden_states
            global_indices = max_ind
            print("globals: ", global_hidden_states.shape, global_indices.shape)

        output_total = torch.zeros_like(global_hidden_states)
        if self.add_bias:
            output_bias_total = torch.zeros_like(global_hidden_states)

        for expert_num, expert in enumerate(self.local_experts):
            local_expert_index = self.local_expert_indices[expert_num]
            local_indices = (global_indices == local_expert_index).nonzero()
            print("LOCAL INDICES: ", local_indices[:,0])
            hidden = global_hidden_states[local_indices, :] # this is across tokens? tokens are assigned?
            print("HIDDEN: ", hidden.shape)
            output, output_bias = expert(hidden)
            # why do we separate the biases
            print("OUTPUT: ", output.shape)

            output_total[local_indices, :] = output
            if self.add_bias:
                output_bias = output_bias.expand_as(output)
                output_bias_total[local_indices, :] = output_bias

        if self.sequence_parallel or (self.expert_parallel_size > 1):
            output_total = tensor_parallel.reduce_scatter_to_sequence_parallel_region_from_moe(
                output_total
            )
            if self.add_bias:
                output_bias_total = tensor_parallel.reduce_scatter_to_sequence_parallel_region_from_moe(
                    output_bias_total
                )
                # bias is duplicated across tensor parallelism ranks;
                # reduce scatter reduces bias across tensor parallel_ranks
                output_bias_total = (
                    output_bias_total / parallel_state.get_tensor_model_parallel_world_size()
                )

        output_total = output_total * max_prob
        # okay this is super weird so we weight by the maximum probability that we get
        # in the end which makes no actual sense!? -- I guess it does but this makes no sense given we are doing top-1
        output_total = output_total.view(hidden_shape)
        if self.add_bias:
            output_bias_total = output_bias_total * max_prob
            output_bias_total = output_bias_total.view(hidden_shape)
        else:
            output_bias_total = None

        return output_total, output_bias_total
