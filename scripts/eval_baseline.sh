export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
MASTER_PORT=$((RANDOM % 101 + 20000))

CHECKPOINT="/data1/code/seu004/models/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3"
OUTPUT="./results/baseline_run"
EVAL_SPLIT="val_seen"
NUM_FRAMES=32
NUM_HISTORY=8

echo "CHECKPOINT: ${CHECKPOINT}"
echo "OUTPUT: ${OUTPUT}"
echo "EVAL_SPLIT: ${EVAL_SPLIT}"

torchrun --nproc_per_node=2 --master_port=$MASTER_PORT streamvln/evaluate_scene_graph.py \
    --model_path $CHECKPOINT \
    --output_path $OUTPUT \
    --eval_split $EVAL_SPLIT \
    --num_frames $NUM_FRAMES \
    --num_history $NUM_HISTORY