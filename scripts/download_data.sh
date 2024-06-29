#!/bin/bash

if [ -d "SNLI" ]; then
    echo "SNLI data already exists. Exiting."
    exit 0
else
    echo "Downloading SNLI data..."
    gdown https://drive.google.com/drive/folders/1FKCDYTnlJWl_4l4EpVG9SBYcwIj0SVeE?usp=sharing
fi