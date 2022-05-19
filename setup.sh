mkdir -p "/netscratch/$USER/slt/models"

echo "srun \
--container-image=/netscratch/enroot/hufe_slt_dlcc_pytorch_20.10.sqsh \
--container-mounts=/netscratch/software:/netscratch/software:ro,/netscratch/$USER/slt:/netscratch/slt,/ds/text:/data,/home/$USER/slt:/workspace \
--gpus=1 \
--mem=64G \
python -m signjoey train configs/sign.yaml" > ./slt/run.sh

./slt/run.sh
