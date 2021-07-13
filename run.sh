srun \
--container-image=/netscratch/hufe/slt_image.sqsh \
--container-mounts=/netscratch/$USER:/netscratch/$USER,/ds/text:/data,./slt:/workspace \
--gpus=1 \
--nodes=1 \
--ntasks=1 \
--mem=42G \
python -m signjoey train configs/sign.yaml
