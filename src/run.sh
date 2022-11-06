#!/bin/bash
python src/main.py \
    --file "/Users/dlh/Desktop/screen_photo_simulator/data/sample_images/large_1.jpg" \
    --savepath "./" \
    --save "output" \
    --canvas-dim 1024 \
    --gamma 1 \
    --type "fixed" \
    --psnr \