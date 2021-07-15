mkdir -p "/netscratch/$USER/slt/models"

echo "srun \
--container-image=/netscratch/enroot/hufe_slt_dlcc_pytorch_20.10.sqsh \
--container-mounts=/netscratch/$USER/slt:/netscratch,/ds/text:/data,slt:/workspace \
--gpus=1 \
--nodes=1 \
--ntasks=1 \
--mem=42G \
python -m signjoey train configs/sign.yaml" > ./slt/run.sh

./slt/run.sh
