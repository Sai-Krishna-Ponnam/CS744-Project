export MASTER_ADDR="$1"
export MASTER_PORT="29500"

torchrun --nnodes=2 --nproc_per_node=1 --node_rank=$2 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train.py
