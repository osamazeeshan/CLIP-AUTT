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
    --adapt_tar_sub=True \
    --adapt_per_video=True \
    --au_prompt_tune=True \
    --key_frame_sel=True \
    --is_video_clip=True \