#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_API_KEY=6f0443d34d41df289b878635a247d89381c06271

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=/workspace/ckpts
VOCAB_FILE=/data/gpt2-vocab.json
MERGE_FILE=/data/gpt2-merges.txt
DATA_PATH=/data/redpajama-arxiv-test_text_sentence

WANDB_PROJECT=moe
WANDB_EXP_NAME=moe_train_eval_harness_test
WANDB_SAVE_DIR=/workspace/wandb

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 64 \
    --global-batch-size 1024 \
    --lr 0.00015 \
    --train-iters 570000 \
    --lr-decay-iters 570000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 0.0 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16 \
    --recompute-granularity selective \
    --use-flash-attn \
    --accumulate-allreduce-grads-in-fp32  
    --attention-dropout 0.0 
    --hidden-dropout 0.0
    --swiglu
    "
    #--fp8-format hybrid 
    #--transformer-impl transformer_engine
#"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 949,50,1 \
    --num-workers 0 \
    --distributed-timeout-minutes 120
"

OUTPUT_ARGS="
    --log-interval 10 \
    --save-interval 500 \
    --eval-interval 50 \
    --eval-iters 50 \
    --eval-harness-interval 1 \
    --eval-harness-tasks openbookqa \    
    --wandb-project $WANDB_PROJECT \
    --wandb-exp-name $WANDB_EXP_NAME \
    --wandb-save-dir $WANDB_SAVE_DIR
"


torchrun $DISTRIBUTED_ARGS /workspace/Megatron-LM/pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl 
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH
