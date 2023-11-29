"""Sample Generate GPT"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import socket
from megatron import get_args
from megatron import print_rank_0
from megatron.core import mpu
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.model import GPTModel
from megatron.training import get_model
from megatron.arguments import core_transformer_config_from_args
from megatron.text_generation_server import MegatronServer
from megatron.text_generation import generate_and_post_process
from megatron.text_generation import beam_search_and_post_process
import torch


from flask import Flask, request, jsonify

# To run on multiGPU node use `torchrun --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6000 generation_test.py`

#/opt/Megatron-LM/examples/megarun_slurm/moe_1p3B_8E_bare.sh --checkpoint /checkpoints/megarun/ckpts_1p3B_bf16 

os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "6000"

config_path = "/opt/Megatron-LM/examples/megarun_slurm/moe_1p3B_8E_bare.sh"
#config_path = "/opt/Megatron-LM/examples/moe_1p3B_8E_bare_r0.sh"




with open(config_path,"r") as f:
    filestr = f.read()
    
#print(file_str)

def extract_keyword_args(filestr, keyword):
    gpt_split = filestr.split(keyword)
    if len(gpt_split) <=1:
        raise ValueError("Config provided does not have a GPT_ARGS variable provided")
    arg_splits = gpt_split[1].split("\"")
    gpt_args = arg_splits[1]
    gpt_args = gpt_args.replace("\n","").replace("\\","").replace("\t","")
    gpt_args = ' '.join(gpt_args.split())
    return gpt_args.strip().split(" ")
print(extract_keyword_args(filestr, "GPT_ARGS"))

#VOCAB_FILE=/datasets/SlimPajama-627B_megatron/gpt-neox-20b-tokenizer/vocab.json
#MERGE_FILE=/datasets/SlimPajama-627B_megatron/gpt-neox-20b-tokenizer/merges.txt
#DATA_PATH=/datasets/SlimPajama-627B_megatron/gpt-neox-20b-tokenizer/train_text_document

def extract_data_paths(filestr):
    vocab_file = filestr.split("VOCAB_FILE=")[1].split("\n")[0]
    merge_file = filestr.split("MERGE_FILE=")[1].split("\n")[0]
    #vocab_file = "/datasets/SlimPajama-627B_megatron/gpt-neox-20b-tokenizer/vocab.json"
    #merge_file = "/datasets/SlimPajama-627B_megatron/gpt-neox-20b-tokenizer/merges.txt"
    #checkpoint_path = "/workspace/ckpts_bf16_125m"
    checkpoint_path = "/checkpoints/megarun/ckpts_1p3b_bf16"
    #checkpoint_path = "/checkpoints/megarun/ckpts_1p3B"
    
    data_path = filestr.split("DATA_PATH=")[1].split("\n")[0]
    return ["--data-path", data_path, "--vocab-file" , vocab_file, "--merge-file" , merge_file,"--load" , checkpoint_path]
    
print(extract_data_paths(filestr))


sys.argv = ["generation_test.py"] # a hack to get around the jupyter issues in the sys.argc which we are messing with
sys.argv += extract_keyword_args(filestr, "GPT_ARGS")
sys.argv += extract_data_paths(filestr)
print(sys.argv)
#sys.argv += extract_keyword_args(filestr, "DATA_ARGS")

def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    args = get_args()
    print("IN MODEL PRIVIDER ARGHS: ", args)
    args.recompute_granularity = None # enforce for inference
    config = core_transformer_config_from_args(args)

    print_rank_0('building GPT model ...')
    model = GPTModel(config, num_tokentypes=0, parallel_output=False, pre_process=pre_process, post_process=post_process)

    return model

def add_text_generate_args(parser):
    group = parser.add_argument_group(title='text generation')
    group.add_argument("--port", type=int, default=5000,
                       help='port for text generation server to run on')
    return parser

initialize_megatron(extra_args_provider=add_text_generate_args,
                    args_defaults={'tokenizer_type': 'HFAutoTokenizer',
                                    'hf_autotokenizer_model': 'EleutherAI/gpt-neox-20b',
                                    'no_load_rng': True,
                                    'no_load_optim': True})

if torch.distributed.get_rank() == 0:
    print("WE ARE RANK 0!")


# 'tokenizer_type': 'GPT2BPETokenizer'
#'tokenizer_type': 'HFAutoTokenizer',
#                                    'hf_autotokenizer_model': 'EleutherAI/gpt-neox-20b',
args = get_args()
print("ARGS: ", args)

model = get_model(model_provider, wrap_with_ddp=False)
_ = load_checkpoint(model, None, None)
assert len(model) == 1, "Above condition should have caught this"
model = model[0]

def model_generate(prompt, num_tokens, temperature):
    print("INSIDE GENERATE")
    prompts = [str(prompt)]
    tokens_to_generate = num_tokens
    logprobs = True
    top_k = 0.0
    top_p = 0.0
    temperature = temperature
    top_p_decay = 0.0
    top_p_bound = 0.0
    add_BOS = False
    stop_on_double_eol = False
    stop_on_eol = False
    random_seed = -1
    prevent_newline_after_colon = False
    print("SENDING TO GENERATE")
    #print("MODEL: ", model)

    response, response_seg, response_logprobs, _ = \
        generate_and_post_process(
        model,
        prompts=prompts,
        tokens_to_generate=tokens_to_generate,
        return_output_log_probs=logprobs,
        top_k_sampling=top_k,
        top_p_sampling=top_p,
        top_p_decay=top_p_decay,
        top_p_bound=top_p_bound,
        temperature=temperature,
        add_BOS=add_BOS,
        use_eod_token_for_early_termination=True,
        stop_on_double_eol=stop_on_double_eol,
        stop_on_eol=stop_on_eol,
        prevent_newline_after_colon=prevent_newline_after_colon,
        random_seed=random_seed)
    print("RESPNOSE RECEIVED")
    return response

#if torch.distributed.get_rank() == 0:
#response = model_generate(">>> Python 3.10", 100, 1.0)
#print("RESPONSE : ", response)
#exit()
#exit()
# SERVER

PORT = 5000

if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0 and torch.distributed.get_rank() == 0:
    server = MegatronServer(model)
    server.run("0.0.0.0",port=PORT)

while True:
    choice = torch.cuda.LongTensor(1)
    torch.distributed.broadcast(choice, 0)
    if choice[0].item() == 0:
        try:
            generate_and_post_process(model)
        except ValueError as ve:
            pass
    elif choice[0].item() == 1:
        try:
            beam_search_and_post_process(model)
        except ValueError as ve:
            pass



FLASK = False
INTERACTIVE = False

if FLASK:
    if torch.distributed.get_rank() == 0:
        # only run the flask server on rank 0
        app = Flask(__name__)

        @app.route('/')
        def home():
            return "Hello, World!"

        @app.route('/generate', methods = ['POST'])
        def generate_text():
            print("IN GENERATE TEXT")
            data = request.get_json()
            prompt = data['prompt']
            temperature = data.get('temperature', 1.0)
            num_tokens = data.get('num_tokens', 100)
            print("PARSED RESPONSE: ", prompt, temperature, num_tokens)
            

            response = generate(prompt, num_tokens, temperature)
            
            return jsonify({'output': output})


        app.run(debug=True)
        
        
if INTERACTIVE:
    print_rank_0("Welcome to interactive program. Type 'quit' to exit.")
    while True:
        user_input = input("> ")
        print("USER INPUT: " + str(user_input))
        num_tokens = 100
        temperature = 1.0
        if user_input.lower() == ":q":
            break
        elif ":tokens" in user_input.lower():
            splits = user_input.lower().split(" ")
            num_tokens = int(splits[1])
            
        else: 
            print("SENDING TO MODEL")
            response = model_generate(user_input, num_tokens, temperature )
            print(f"Processed response: {response}")