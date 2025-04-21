deepspeed --num_nodes=2 --num_gpus=4 \
  --node_rank=0 \
  --master_addr=10.10.1.1 \
  --master_port=12345 \
  traindeepspeed.py \
    -a vgg16 \
    --deepspeed \
    --deepspeed_config /config/ds_config.json \
    --dummy \
    -b 64 \
    --epochs 5
