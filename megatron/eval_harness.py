import lm_eval 
from lm_eval.base import BaseLM
from lm_eval import tasks

from megatron import get_tokenizer
from megatron import is_last_rank
from megatron import print_rank_0
from megatron import get_args
from megatron.training import get_model
from megatron.text_generation.forward_step import ForwardStep
from megatron.text_generation.generation import _build_attention_mask_and_position_ids
from megatron.arguments import core_transformer_config_from_args
from megatron.text_generation import generate_and_post_process
from megatron.model import GPTModel
from megatron.checkpointing import load_checkpoint

import torch
import json


def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    args = get_args()
    args.recompute_granularity = None # enforce for inference
    config = core_transformer_config_from_args(args)

    print_rank_0('building GPT model ...')
    model = GPTModel(config, num_tokentypes=0, parallel_output=False, pre_process=pre_process, post_process=post_process)

    return model


class MegatronEvaluateHarness(BaseLM):
    def __init__(
        self, 
        model,
        tokenizer,
        max_batch_size=512,
        max_length=1024,
        top_k_sampling=0.0,
        top_p_sampling=0.0,
        top_p_decay=0.0,
        top_p_bound=0.0,
        temperature=1.0,
        random_seed=1234,
    ):
        super(MegatronEvaluateHarness, self).__init__()

        self.max_batch_size = max_batch_size
        self._max_length = max_length
        self.tokenizer = tokenizer
        self.model = model
        
        self.top_k_sampling = top_k_sampling
        self.top_p_sampling = top_p_sampling
        self.top_p_decay = top_p_decay
        self.top_p_bound = top_p_bound
        self.temperature = temperature
        self.random_seed = random_seed
    
        self.vocab_size = self.tokenizer.vocab_size

        self.is_main = args.rank == 0
        self.is_local_main = args.local_rank == 0  
        
    @property
    def eot_token_id(self):
        self.tokenizer.eos_token_id
        
    @property
    def max_length(self):
        return self._max_length
    
    @property
    def max_gen_toks(self):
        return self.max_length
    
    @property
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    @property
    def batch_size(self):
        return self.max_batch_size
        
    def tok_encode(self, string):
        return self.tokenizer.tokenize(string)
    
    def tok_decode(self, tokens):
        return self.tokenizer.detokenize(tokens.cpu().numpy())
    
    def _model_call(self, inps):
        self.forward_step = ForwardStep(self.model, max_batch_size=self.max_batch_size, max_sequence_length=self.max_length)
        with torch.no_grad():
            attention_mask, position_ids = _build_attention_mask_and_position_ids(inps)
            logits = self.forward_step(inps, position_ids, attention_mask)
            return logits
        
    def _model_generate(self, context, max_length, eos_token_id):
        response, response_seg, response_logprobs, _ = \
            generate_and_post_process(
                self.model,
                prompts=context,
                tokens_to_generate=max_length,
                return_output_log_probs=True,
                top_k_sampling=self.top_k_sampling,
                top_p_sampling=self.top_p_sampling,
                top_p_decay=self.top_p_decay,
                top_p_bound=self.top_p_bound,
                temperature=self.temperature,
                add_BOS=True,
                use_eod_token_for_early_termination=True,
                stop_on_double_eol=False,
                stop_on_eol=False,
                prevent_newline_after_colon=False,
                random_seed=self.random_seed,
            )
        return response, response_seg, response_logprobs
            
            
class Evaluator():
    def __init__(self, checkpoint_path=None, model=None, results_path="./results.json", tokenizer=None, task_list=None):
        
        if tokenizer is None:
            try:
                self.tokenizer = get_tokenizer()
            except Exception as e:
                print("Failed to create a tokenizer and none provided: ", e)
                
        if checkpoint_path is None and model is None:
            raise ValueError("Either a model object or a checkpoint file must be provided to the evaluator.")
        
        if model is not None:
            self.model = model
        else: 
            model = get_model(model_provider, wrap_with_ddp=False)
            _ = load_checkpoint(model, None, None)
            assert len(model) == 1, "Above condition should have caught this"
            self.model = model[0]
            
        print_rank_0(f"TASK LIST: {task_list}")
        if task_list is None:
            self.task_list = ["lambada_openai", "hellaswag"]
        else:
            self.task_list = task_list.split(",")
        
        # Download tasks only on local main rank
        args = get_args()
        if args.local_rank == 0:
            task_dict = tasks.get_task_dict(self.task_list)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        task_dict = tasks.get_task_dict(self.task_list)
        
        self.task_dict = task_dict
        self.results_path = results_path
        
        
    def evaluate(self, max_batch_size=64, max_length=2048, adaptive_seqlen=False, num_fewshot=0, eval_fp32=False):
        
        adaptor = MegatronEvaluateHarness(self.model, self.tokenizer, max_batch_size=max_batch_size, max_length=max_length)
        
        results = lm_eval.evaluator.evaluate(adaptor, self.task_dict, False, num_fewshot, None)
        return results
    
    def write_results(self, results, results_path=None):
        if results_path is None:
            results_path = self.results_path
        
        with open(results_path, "w") as f:
            json.dump(results, f)