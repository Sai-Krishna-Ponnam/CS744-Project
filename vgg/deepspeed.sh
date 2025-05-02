#set wandb offline
#export WANDB_MODE=offline
CONFIG_DIR="/users/devd/CS744-Project/vgg/config"
CONFIG_FILES=("ds_config1.json" "ds_config2.json" "ds_config3.json" "ds_config4.json")
BATCH_SIZES=("16" "32" "64" "128" "256")

for CONFIG in "${CONFIG_FILES[@]}"; do
  for BATCH in "${BATCH_SIZES[@]}"; do
    echo "Running config $CONFIG with batch size $BATCH"

    deepspeed --hostfile hostfile.txt \
      --no_ssh \
      --node_rank 0 \
      --master_addr 10.10.1.1 \
      --master_port 12345 \
      traindeepspeed.py \
      -a vit_b_16 \
      --deepspeed \
      --deepspeed_config "$CONFIG_DIR/$CONFIG" \
      --dummy \
      -b "$BATCH" \
      --epochs 15 \
      | tee "logs/${CONFIG%.json}_b$BATCH.log"

    echo "Finished config $CONFIG with batch size $BATCH"
    echo "---------------------------------------------"
  done
done
