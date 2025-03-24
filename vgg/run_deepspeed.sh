deepspeed --hostfile=hostfile --no_ssh --node_rank=$2 --num_nodes=2 --num_gpus=1 --master_addr="$1" --master_port=29500 train_deepspeed.py --deepspeed
