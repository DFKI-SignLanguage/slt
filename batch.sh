sbatch \
  --array "0-1%2" --job-name phoenix_translation --partition batch \
  --wrap "srun \
  --container-image=/netscratch/enroot/hufe_slt_0.2_dlcc_pytorch_20.10.sqsh \
  --container-mounts=/netscratch/software:/netscratch/software:ro,/netscratch/hufe/slt:/netscratch/slt,/ds/text:/data,/home/hufe/slt:/workspace \
  --gpus=1 \
  --mem=64G \
  wandb agent sign-language-translation/workspace/w8nh81uv"