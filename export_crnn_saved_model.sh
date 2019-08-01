#!/usr/bin/env bash

set -eux

MODELS_ROOT=${MODELS_ROOT:-"/tmp"}

if [[ ! -e checkpoints/checkpoint ]]; then
    mkdir -p checkpoints
    wget -O crnn_synth90k.zip https://uc9cca175051d2525d64f94e4e5e.dl.dropboxusercontent.com/zip_by_token_key\?_download_id\=47905535386837847802756166283378560665273523961239171781204706724\&_notify_domain\=www.dropbox.com\&dl\=1\&key\=AkBmphiS5ltKjgS9WyHJYOC69m8zR_n7stXNEo09je8gX7qtUwijqugNAcybGddhPziVrKgbLF3isBkKuv8vkYC0pVS5hu6PBr8yskJcD9daIuH2WGait5pYcwa6JslHfzXDDF3RVfWjN8O_-bAP9GFd2ApjOVzFlOFBi0B_iK2K0UH6cbVrVzl7EvKYCVLAwws42ZUVGLhXzKO4Lln40VvBa3EcPDW3CBLBM-h6GL_KTw
    unzip crnn_synth90k.zip -d checkpoints
    rm crnn_synth90k.zip
fi

PYTHONPATH=$(pwd) python export_saved_model.py \
    --export_dir $MODELS_ROOT/crnn/1 \
    --ckpt_path checkpoints/shadownet.ckpt \
    --char_dict_path data/char_dict/char_dict_en.json \
    --ord_map_dict_path data/char_dict/ord_map_en.json

