#! /bin/sh
# 安装站点依赖
pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=0 && uvicorn server:app --host 0.0.0.0 --port 8080