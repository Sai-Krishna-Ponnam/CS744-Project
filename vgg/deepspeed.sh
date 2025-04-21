deepspeed --hostfile hostfile.txt \
  --no_ssh \
  --node_rank 0 \
  --master_addr 10.10.1.1 \
  --master_port 12345 \
  traindeepspeed.py \
  -a vgg16 \
  --deepspeed \
  --deepspeed_config /users/devd/CS744-Project/vgg/config/ds_config2.json \
  --dummy \
  -b 64 \
  --epochs 5
