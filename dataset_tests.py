
# %%

%load_ext autoreload
%autoreload 2
import os
import torch
from torch import Tensor
from functools import partial
from typing import Union
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.arguments import parse_args
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.blended_megatron_dataset_config import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import GPTDataset
from megatron.core.datasets.indexed_dataset import MMapIndexedDataset
import megatron.model
from megatron import get_tokenizer
from megatron.core.models.gpt import GPTModel
from megatron.training import pretrain
from megatron.core.transformer.spec_utils import import_module
from megatron.tokenizer.tokenizer import build_tokenizer

from megatron.utils import (
    get_ltor_masks_and_position_ids,
    get_batch_on_this_cp_rank,
    average_losses_across_data_parallel_group
)
from megatron.arguments import core_transformer_config_from_args
from megatron.core.models.gpt.gpt_layer_specs import (
    gpt_layer_with_transformer_engine_spec,
    gpt_layer_with_transformer_engine_spec_moe
)
from megatron.training import build_train_valid_test_data_iterators
from megatron.training import build_train_valid_test_data_loaders
print("DONE")


# %%

import torch
import os
os.environ['LOCAL_RANK'] = "0"
#os.environ["MASTER_ADDR"] = "172.17.0.3"
#os.environ["MASTER_ADDR"] = "172.17.0.1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "6001"
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

def core_gpt_dataset_config_from_args(args):
    return GPTDatasetConfig(
        is_built_on_rank=is_dataset_built_on_rank,
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=[args.data_path],
        #blend_per_split=[args.train_data_path, args.valid_data_path, args.test_data_path],
        split=args.split,
        path_to_cache=args.data_cache_path,
        return_document_ids=args.retro_return_doc_ids
    )

def is_dataset_built_on_rank():
    return (mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()) and mpu.get_tensor_model_parallel_rank() == 0



def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    #args = get_args()
    #args = parse_args()
    args = Args()
    print("num samples: ",train_val_test_num_samples)

    print_rank_0("> building train, validation, and test datasets for GPT ...")
    gpt_config = core_gpt_dataset_config_from_args(args)
    print("GPT CONFIG: ", gpt_config)
    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        GPTDataset,
        train_val_test_num_samples,
        gpt_config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


class Args:
    seed = 1
    seq_length = 1024
    data_path = "/workspace/gpt-neox/data/enwik8/enwik8_text_document"
    train_data_path = "/workspace/gpt-neox/data/enwik8/enwik8_text_document"
    valid_data_path = "/workspace/gpt-neox/data/enwik8/enwik8_text_document"
    test_data_path = "/workspace/gpt-neox/data/enwik8/enwik8_text_document"
    split = "100,0,0"
    data_cache_path = None
    retro_return_doc_ids = False
    tokenizer_type = "GPT2BPETokenizer"
    merge_file  = "/workspace/gpt-neox/data/gpt2-merges.txt"
    vocab_file = "/workspace/gpt-neox/data/gpt2-vocab.json"
    rank = 0
    make_vocab_size_divisible_by =1 
    tensor_model_parallel_size = 1
    iteration = 0
    train_samples = 100000
    train_iters = 2000
    eval_interval = 1000
    eval_iters = 5
    global_batch_size = 100000
    consumed_train_samples = 0
    dataloader_type = "single"
    micro_batch_size = 5000
    num_workers = 0
    skip_train = False
    consumed_valid_samples = 0
    
    
    
 # %%
 
#train_valid_test_datasets_provider([100,0,0])
args = Args()
global _GLOBAL_ARGS
_GLOBAL_ARGS = args
megatron.global_vars.set_args(args)
print(_GLOBAL_ARGS)
iterators = build_train_valid_test_data_iterators(train_valid_test_datasets_provider)#
#print("ITERATORS: ", iterators)
train_dataloader, valid_dataloader, test_dataloader = build_train_valid_test_data_loaders(train_valid_test_datasets_provider)

# %%

batch = next(iterators[0])
print(batch["text"].shape)
print(batch["text"][0,:])




# %%
print(train_dataloader)
#dir(train_dataloader)
dataset = train_dataloader.dataset
print(dataset)
# so this just pulls out a section of length sequence length
outputs = dataset._query_document_sample_shuffle_indices(0)
print(outputs)
arr, _ = outputs
print(len(arr))
tokenizer = build_tokenizer(args)
print(tokenizer)
tokens = tokenizer.detokenize(arr)
print(tokens)


# %%
# let's try to see if we can readlines
import numpy as np

base_path = "/workspace/gpt-neox/data/enwik8/enwik8_text_document"
ipath = "/workspace/gpt-neox/data/enwik8/enwik8_text_document.idx"
bpath = "/workspace/gpt-neox/data/enwik8/enwik8_text_document.bin"

args = Args()
tokenizer = build_tokenizer(args)
print(tokenizer)
ds = MMapIndexedDataset(base_path)
print(ds)
#item = ds.__getitem__(0)
item = ds.get(0,0, 1000)
print(item)
print(type(item))
print(len(item))
#tokenizer = get_tokenizer()
tokens = tokenizer.detokenize(item)
print("TOKENS: ", tokens)

#newfp = np.memmap(bpath, mode='r', order = 'C')
#print(newfp.max())
#print(newfp)


#with open(ipath, "r") as f:
#    lines = f.readlines()
#    print("LINE: ", lines[0])




# %%
