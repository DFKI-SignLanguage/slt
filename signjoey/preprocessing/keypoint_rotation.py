import os
import pickle
import random
import torch
from torchtext import data
from torchtext.data import BucketIterator
import sys


sys.path.append(r"C:\\Users\\areeb\Downloads\\SignLanguage_DataAugmentation")

from torchtext.data import Dataset
from signjoey.batch import Batch
from signjoey.data import make_data_iter
from signjoey.dataset import SignTranslationDataset


from signjoey.vocabulary import (
    UNK_TOKEN,
    EOS_TOKEN,
    BOS_TOKEN,
    PAD_TOKEN,
)

ROTATION_CFG_DEFAULT = {
    'do_rotate': 1,
    'x': 10,
    'y': 10,
    'z': 10,
    'order': 'xyz',
}

BATCH_TYPE_DEFAULT = 'sentence'
BATCH_SIZE = 1

DATA_CFG = {
    'data_path': "C:\\Users\\areeb\\Downloads\\SignLanguage_DataAugmentation\\",
    'filename_raw': "phoenix14t.pami0.test",
    'feature_size': 1728
}


def rotate_dataset(data_cfg: dict, rotation_cfg: dict):
    data_path = data_cfg.get("data_path", "./data") #C:\Users\areeb\Downloads\SignLanguage_DataAugmentation\
    raw_path = os.path.join(data_path, data_cfg["filename_raw"])
    pad_feature_size = data_cfg["feature_size"]
    txt_lowercase = False

    def tokenize_features(features):
        ft_list = torch.split(features, 1, dim=0)
        return [ft.squeeze() for ft in ft_list]


    def stack_features(features, something):
        return torch.stack([torch.stack(ft, dim=0) for ft in features], dim=0)

    sequence_field = data.RawField()
    signer_field = data.RawField()

    sgn_field = data.Field(
        use_vocab=False,
        init_token=None,
        dtype=torch.float32,
        preprocessing=tokenize_features,
        tokenize=lambda features: features,  # TODO (Cihan): is this necessary?
        batch_first=True,
        include_lengths=True,
        postprocessing=stack_features,
        pad_token=torch.zeros((pad_feature_size,)),
    )

    gls_field = data.Field(
        pad_token=PAD_TOKEN,
        # tokenize=tokenize_text,
        batch_first=True,
        lower=False,
        include_lengths=True,
    )

    txt_field = data.Field(
        init_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        # tokenize=tokenize_text,
        unk_token=UNK_TOKEN,
        batch_first=True,
        lower=txt_lowercase,
        include_lengths=True,
    )

    raw_dataset = SignTranslationDataset(
        path=raw_path,
        fields=(sequence_field, signer_field, sgn_field, gls_field, txt_field),
    )

    raw_dataset = make_data_iter(dataset=raw_dataset,
        batch_size=32,
        batch_type=BATCH_TYPE_DEFAULT,
        shuffle=False,
        train=False,)
    rotated_dataset_list = []

    # data_iter = make_data_iter(
    #     dataset,
    #     batch_size=batch_size,
    #     batch_type=batch_type,
    #     shuffle=False,
    #     train=False,
    # )



    #for i in range(0,len(raw_dataset.dataset[0].sgn)):
    #print("\nThis is ","th iteration: ",raw_dataset.dataset[1].sgn[0].shape)
    print("This is ",raw_dataset.__dict__)

    #input() 
    
    for example in iter(raw_dataset):   
        #print("\n\nThis is dict: \n",example.__dict__)
        #input()
        dim1, dim2, _ = example.sgn.shape

        # bring the data into 3d
        # batch.sgn = batch.sgn.view((dim1, dim2, 576, 3)) # 576 for 1.728 dimensions
        example.sgn = example.sgn.view((dim1, dim2, 236, 3))  # 236 for 708 dimensions

        # draw random angle values that direct the amount of rotation
        theta_x = torch.deg2rad(
            torch.Tensor([rotation_cfg['x']]))
        theta_y = torch.deg2rad(
            torch.Tensor([rotation_cfg['y']]))
        theta_z = torch.deg2rad(
            torch.Tensor([rotation_cfg['z']]))

        # calculate the cos and sin for the rotation mat
        c_x = torch.cos(theta_x)
        s_x = torch.sin(theta_x)
        c_y = torch.cos(theta_y)
        s_y = torch.sin(theta_y)
        c_z = torch.cos(theta_z)
        s_z = torch.sin(theta_z)

        # setup the rotation mat for every axis
        rot_x = torch.Tensor([[1, 0, 0], [0, c_x, -s_x], [0, s_x, c_x]]).cuda()
        rot_y = torch.Tensor([[c_y, 0, s_y], [0, 1, 0], [-s_y, 0, c_y]]).cuda()
        rot_z = torch.Tensor([[c_z, -s_z, 0], [s_z, c_z, 0], [0, 0, 1]]).cuda()

        # concat them following the oder given by the config
        lookup = {
            'x': rot_x,
            'y': rot_y,
            'z': rot_z
        }
        pos = list(rotation_cfg['order'])
        rot = lookup[pos[0]] @ lookup[pos[1]] @ lookup[pos[2]]

        # reshape after applying the matrix -> flatten it again
        example.sgn = torch.matmul(example.sgn, rot)
        # batch.sgn = batch.sgn.view((dim1, dim2, 1728))
        example.sgn = example.sgn.view((dim1, dim2, 708))

        dict_instance_rotated = {
            'name': example.sequence,
            'signer': example.signer,
            'gloss': example.gls,
            'text': example.txt,
            'sign': example.sgn,
        }

        rotated_dataset_list.append(dict_instance_rotated)

    return rotated_dataset_list


def write_dataset_list(filename, dataset_list):
    with open(filename, 'wb') as f:
        pickle.dump(dataset_list, f)


if __name__ == '__main__':
    rotated_dataset = rotate_dataset(DATA_CFG, ROTATION_CFG_DEFAULT)
    write_dataset_list(OUTPUT_FILENAME, rotated_dataset)

