export PYTHONPATH="`pwd`:${PYTHONPATH}"
if [ $# != 2 ]
then 
  echo "Please specify 1) gpus. 2) Target dataset ('Biovid' or 'StressID' or 'BAH_DB')."
  exit
fi

gpus=${1}
ds=${2}


CUDA_VISIBLE_DEVICES=${gpus} python clip_autt.py \
    --current_ds=${ds} \
    --train_t_adpt_cl=True \
    --is_video_clip=True 
