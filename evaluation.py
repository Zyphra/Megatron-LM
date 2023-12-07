# File containing evaluation scripts and evaluator object from lm eval harness

from megatron import get_args

from megatron.initialize import initialize_megatron
from megatron.core import mpu
from megatron.eval_harness import Evaluator

import os
import sys

from lm_eval.tasks import ALL_TASKS
import argparse


def extract_keyword_args(filestr, keyword):
    gpt_split = filestr.split(keyword)
    if len(gpt_split) <=1:
        raise ValueError("Config provided does not have a GPT_ARGS variable provided")
    arg_splits = gpt_split[1].split("\"")
    gpt_args = arg_splits[1]
    gpt_args = gpt_args.replace("\n","").replace("\\","").replace("\t","")
    gpt_args = ' '.join(gpt_args.split())
    return gpt_args.strip().split(" ")


def extract_data_paths(filestr, checkpoint_path):
    vocab_file = filestr.split("VOCAB_FILE=")[1].split("\n")[0]
    merge_file = filestr.split("MERGE_FILE=")[1].split("\n")[0]
    data_path = filestr.split("DATA_PATH=")[1].split("\n")[0]
    return ["--data-path", data_path, "--vocab-file", vocab_file, "--merge-file", merge_file, "--load", checkpoint_path]
    

def parse_config_file_update_argv(config_path, checkpoint_path):
    with open(config_path,"r") as f:
        filestr = f.read()
    
    sys.argv = [""]
    sys.argv += extract_keyword_args(filestr, "GPT_ARGS")
    sys.argv += extract_data_paths(filestr, checkpoint_path)



def add_text_generate_args(parser):
    group = parser.add_argument_group(title='text generation')
    group.add_argument("--port", type=int, default=5000,
                       help='port for text generation server to run on')
    return parser


def init_megatron():
    initialize_megatron(
        extra_args_provider=add_text_generate_args,
        args_defaults={
            'tokenizer_type': 'HFAutoTokenizer',
            'hf_autotokenizer_model': 'EleutherAI/gpt-neox-20b',
            'no_load_rng': True,
            'no_load_optim': True
        }
    )


if __name__ == '__main__':
    # EXAMPLE COMMAND:
    # torchrun --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6000 evaluation.py --config /opt/Megatron-LM/examples/megarun_slurm/moe_1p3B_8E_bare.sh --checkpoint /checkpoints/megarun/ckpts_1p3b_bf16 --task-list openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,lambada_openai
    # task list openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,lambada_openai,lambada_standard
    parser = argparse.ArgumentParser(description='Download evaluation harness', allow_abbrev=False)
    parser.add_argument('--config', type=str, help='Path to the model config file.')
    parser.add_argument('--checkpoint', type=str, help='Path to the model config file.')
    parser.add_argument('--task-list', type=str, default="", help="Pass in a comma separated task list.")
    parser.add_argument('--results-path', type=str, default="./results.json", help="Path for a json file with results.")
    
    args = parser.parse_args()
    
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    
    config_path = args.config
    checkpoint_path = args.checkpoint
    task_list = args.task_list
    if task_list == "":
        task_list = None
    if task_list == "all":
        task_list = ALL_TASKS
    
    # parse the config file
    parse_config_file_update_argv(config_path, checkpoint_path)
    
    # initialize megatron with the correct args
    init_megatron()
    
    # begin evaluation
    evaluator = Evaluator(checkpoint_path=checkpoint_path, task_list=task_list)
    results = evaluator.evaluate()

    megatron_args = get_args()
    print(f"RESULTS for [rank {megatron_args.rank}, local_rank {megatron_args.local_rank}]: ", results)
    # TODO: get a better understanding of how evaluator is workin in a parallel setting
    # gpt-neox is taking resulst from the zero-th rank: 
    # https://github.com/EleutherAI/gpt-neox/blob/efea81f5df397f733a98be13cb7bd1d66e94be27/evaluate.py#L43
    if megatron_args.rank == 0:
        evaluator.write_results(results, args.results_path)
