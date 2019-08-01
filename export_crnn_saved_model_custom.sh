#!/usr/bin/env bash

set -eux

PYTHONPATH=$(pwd) python export_saved_model.py \
    --export_dir model/vitals_synth_model_pb \
    --ckpt_path model/vitals_synth_model/shadownet_2019-06-28-18-22-09.ckpt-2000000 \
    --char_dict_path model/vitals_synth_model/char_dict.json \
    --ord_map_dict_path model/vitals_synth_model/ord_map.json

rm -rf /tmp/crnn/2
mkdir -p /tmp/crnn/2
mv -f model/crnn_syn90k_saved_model_pb/* /tmp/crnn/2
