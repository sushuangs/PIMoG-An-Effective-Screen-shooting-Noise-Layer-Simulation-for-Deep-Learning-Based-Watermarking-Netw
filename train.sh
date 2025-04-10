python main.py --num_workers 16 \
    --image_dir '/data/experiment/data/bcdn_edge/train' \
    --image_val_dir '/data/experiment/data/bcdn_edge/val' \
    --num_epoch 200 \
    --batch_size 32 \
    --model_name 'gtos_I' \
    --distortion 'Identity'