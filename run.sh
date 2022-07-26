
#eval
CUDA_VISIBLE_DEVICES=0 nohup python ./eval.py >> log/eval.log 2>&1 &

#train
# CUDA_VISIBLE_DEVICES=0 nohup python ./train.py >> log/train.log 2>&1 &

#train and eval
# CUDA_VISIBLE_DEVICES=0 nohup python ./run.py >> log/train_eval.log 2>&1 &
