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
    --load_t_adpt_cl_mod=True \
    --is_video_clip=True \
    --eval_au_tar_sb=True \
