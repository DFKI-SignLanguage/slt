srun \
--container-image=/netscratch/enroot/hufe_slt_0.2_dlcc_pytorch_20.10.sqsh \
--container-mounts=/netscratch/software:/netscratch/software:ro,/netscratch/hufe/wmt_slt:/netscratch/slt,/ds/text:/data,/home/hufe/slt:/workspace \
--gpus=1 \
--mem=64G \
-p RTXA6000-SLT \
python -m signjoey train configs/focusnews.yaml
#python -m signjoey test configs/focusnews.yaml --ckpt '/netscratch/slt/model - 01 08 2022 - 15:17:58/best.ckpt'
#--pty bash
#python -m signjoey test configs/focusnews.yaml --ckpt '/netscratch/slt/model - 01 08 2022 - 15:17:58/best.ckpt'
#python -m signjoey train configs/focusnews.yaml