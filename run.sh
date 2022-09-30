srun \
--container-image=/netscratch/enroot/hufe_slt_0.2_dlcc_pytorch_20.10.sqsh \
--container-mounts=/netscratch/software:/netscratch/software:ro,/netscratch/hufe/wmt_slt:/netscratch/slt,/ds/text:/data,/home/hufe/slt:/workspace \
--gpus=1 \
--mem=64G \
python -m signjoey train configs/phoenix.yaml

# -p RTXA6000-SLT \