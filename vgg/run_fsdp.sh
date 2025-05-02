export MASTER_ADDR="10.10.1.1"
export MASTER_PORT="29500"

CONFIG_FILES=("0" "1" "2")
BATCH_SIZES=("16" "32" "64" "128" "256")

for CONFIG in "${CONFIG_FILES[@]}"; do
  for BATCH in "${BATCH_SIZES[@]}"; do
    echo "Running config $CONFIG with batch size $BATCH"

    torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT train_fsdp.py \
        --batch_size "$BATCH" \
        --num_epochs 15 \
        --sharding_strategy "$CONFIG"

            echo "Finished config $CONFIG with batch size $BATCH"
    echo "---------------------------------------------"
  done
done

