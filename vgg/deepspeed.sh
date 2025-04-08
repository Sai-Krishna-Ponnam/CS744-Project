deepspeed --num_nodes=1 --num_gpus=2\
 --master_port=12345 traindeepspeed.py\
  -a vgg16 --deepspeed --deepspeed_config /scr/dmehrotra/CS744-Project/config/ds_config.json \
  --dummy \
  -b 64 \
--epochs 5 \
