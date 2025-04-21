deepspeed --hostfile hostfile.txt \
  traindeepspeed.py \
  -a vgg16 \
  --deepspeed \
  --deepspeed_config /config/ds_config.json \
  --dummy \
  -b 64 \
  --epochs 5
