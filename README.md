## Content

This repository contains the geometrical augmentation extension to the slt module by camgöz.
Additionally it contains small scripts for running the code on the Slurm cluster of DFKI and to run a weights and biases sweep.

## Slurm Configuration

Before doing anything - read the __complete__ [SLURM documentation](http://projects.dfki.uni-kl.de/km-publications/web/ML/core/hpc-doc/)

The command for starting a training is:
```
srun \
--container-image=/netscratch/enroot/hufe_slt_0.2_dlcc_pytorch_20.10.sqsh \
--container-mounts=/netscratch/hufe/wmt_slt:/netscratch/slt,/ds/text:/data,/home/hufe/slt:/workspace \
--gpus=1 \
--mem=64G \
-p RTXA6000-SLT \
python -m signjoey train configs/phoenix.yaml
```

Lets go through the keyword one by one
* `container-image` specifies the enroot container(based on docker) - the specified container has all the dependencies installed that are needed for this project. One important note is, that the container is set up for my weights and biases(wandb) account. When somebody else contains the project her/him should change the wandb credentials and then save the enroot container to keep the changes. Check out [How to build a new container on top of an existing container](http://projects.dfki.uni-kl.de/km-publications/web/ML/core/hpc-doc/docs/slurm-cluster/custom-software/#modify-an-existing-image) and [How to login in wandb](https://docs.wandb.ai/quickstart). After logging into the new account and saving the new enroot container, make sure to specify the correct container in the run script.
* `--container-mounts` specifies which folder will be shared between the gpu cluster and the enroot container. If we specify `/home/hufe/slt:/workspace` for example, it means that the folder `/home/hufe/slt` is accessbile in the enroot container under `/workspace`. Every file operation will appear in both folders. In the current setting all the models are saved in `/netscratch/hufe/wmt_slt`, the data is read from `/ds/text` and the source files are provided from `home/hufe/slt`
* `--gpus=1` makes sure that there is a gpu in the enroot container
* `--mem=64G` makes sure there is enrough ram in the enroot container
* `-p RTXA6000-SLT` specifies the gpu that should be selected, this can be left out. We should have priority on this gpu, so this can yield to faster allocation.

After these keywords you can call any bash command as you would usually do.
So for example `python -m signjoey train configs/phoenix.yaml` would start a training with the config file phoenix.yaml.

## Weights and Biases
A weights and biases hyperparamter sweep can be used to extensively search the hyperparameter space. The sweep can be start in the wandb user interface or on the commandline. Once this is done the sweep get an id assigned like `w8nh81uv`. Once this state is achieved worker can be started by initalizing a slurm configuration above and calling `wandb agent sign-language-translation/workspace/w8nh81uv` (may be subject to change if another person hosts this repo, due to a changed wandb account).
Such a configuration could look like that

```
srun \
--container-image=/netscratch/enroot/hufe_slt_0.2_dlcc_pytorch_20.10.sqsh \
--container-mounts=/netscratch/hufe/wmt_slt:/netscratch/slt,/ds/text:/data,/home/hufe/slt:/workspace \
--gpus=1 \
--mem=64G \
-p RTXA6000-SLT \
wandb agent sign-language-translation/workspace/w8nh81uv
```


## Training Configuration

The configuration of a single experiment is done via a yaml file. These yaml files files are stored in the `configs/` folder. 
* The `data_path` points to the root directory, where the training data is located
* The `train`, `dev` and `test` keyword specify the name of the specific file in that directory
* The `feature_size` keyword specifies how many dimensions each datapoint the dataset contains
* The `model_dir` keyword specifies the location in the **container** where all the results and checkpoints shall be safed to 
* The `recognition_loss_weight` keyword specifies wheter or not glosses are used in the training process - we don`t use glosses so we shall set this to 0
* The `eval_metric` can be `bleu` or `chrf` (maybe something else to - check in code) and is used to evaluate the model. This is used for tuning the learning rate scheduler. If the task is hard there will be no inital learning in bleu, thus the learning rate will shrink quickly. That problem occured many times in our research and leads to short runs (~20-30 mins). Make sure that the appropriate `eval_metric` is selected
* The `min_epochs` keyword was an desperate attempt to solve these short runs. This keyword is not solving the underlying issue, of setting up the learning rate scheduler(lr-scheduler), which caused a lot of trouble. Ideally this keyword is patched out of the pipeline again in the future. When the lr-scheduler would normally stop the run but the number of training steps is below the number of min_epochs, the run keeps on running. BUT! only with a super small learning rate, since the lr-scheduler deminished it to be almost zero.
* The `scheduling` keyword describes the kind of scheduler that is used. This should be investigated. Currently it is set to `platue`. The documentation can be found [here](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html). The schedulers are built in file `builders.py` in the function `build_scheduler`.
* `decrease_factor` and `patience` are scheduler specific keywords and are exaplained in the [documentation](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html)
* `use_geometric_augmentation` highlights wheter geometric augmentation is used or not.
* `max_x`, `max_y` and `max_z` depict how much every batch can be rotated around the respective axis. If `max_x` is 60 that means for every batch the rotation around the x-axis is randomly drawn from the range [-60°, +60°].
* The `order` keyword describes the order in which the rotations are applied - `xyz` means the x axis is first applied, then the y rotation and then z rotation
The other keywords should be self explainatory.

## Dataset format

When creating a new dataset the right format must be employed to make it compatible to this pipeline.
The dataset must consist of 3 files - a train, a dev and a test set, which can be specified in the configuration file. 

### Content of the files

The files contain a simple list, which is compressed using the (pickle module)[https://docs.python.org/3/library/pickle.html]. Let the list containing the data be named `train_list` the file can be created by
```
with open(f'dataset.train', 'wb') as f:
    pickle.dump(train_list, f)
```

The format of the list is as follows:

Each entry is a dictionary with the keywords 'name', 'signer', 'gloss', 'text' and 'sign'.

* `name` works as an identifier for each datapoint in the dataset
* `signer` refers to the name of the translator/signer in the video
* `gloss` aren't used in this project - we can leave them as an empty string
* `text` is the target text of the translation
* `sign` is a pytorch tensor   


=======================================================================

# Sign Language Transformers (CVPR'20)

This repo contains the training and evaluation code for the paper [Sign Language Transformers: Sign Language Transformers: Joint End-to-end Sign Language Recognition and Translation](https://www.cihancamgoz.com/pub/camgoz2020cvpr.pdf). 

This code is based on [Joey NMT](https://github.com/joeynmt/joeynmt) but modified to realize joint continuous sign language recognition and translation. For text-to-text translation experiments, you can use the original Joey NMT framework.
 
## Requirements
* Download the feature files using the `data/download.sh` script.

* [Optional] Create a conda or python virtual environment.

* Install required packages using the `requirements.txt` file.

    `pip install -r requirements.txt`

## Usage

  `python -m signjoey train configs/sign.yaml` 

! Note that the default data directory is `./data`. If you download them to somewhere else, you need to update the `data_path` parameters in your config file.   
## ToDo:

- [X] *Initial code release.*
- [X] *Release image features for Phoenix2014T.*
- [ ] Share extensive qualitative and quantitative results & config files to generate them.
- [ ] (Nice to have) - Guide to set up conda environment and docker image.

## Reference

Please cite the paper below if you use this code in your research:

    @inproceedings{camgoz2020sign,
      author = {Necati Cihan Camgoz and Oscar Koller and Simon Hadfield and Richard Bowden},
      title = {Sign Language Transformers: Joint End-to-end Sign Language Recognition and Translation},
      booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year = {2020}
    }

## Acknowledgements
<sub>This work was funded by the SNSF Sinergia project "Scalable Multimodal Sign Language Technology for Sign Language Learning and Assessment" (SMILE) grant agreement number CRSII2 160811 and the European Union’s Horizon2020 research and innovation programme under grant agreement no. 762021 (Content4All). This work reflects only the author’s view and the Commission is not responsible for any use that may be made of the information it contains. We would also like to thank NVIDIA Corporation for their GPU grant. </sub>
