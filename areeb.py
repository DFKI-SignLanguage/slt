import gzip
from copy import deepcopy
import pickle


def read_gzip(filename):
    #with gzip.open(filename, "rb") as f:
    #    loaded_object = pickle.load(f)
    #    return loaded_object
    with open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object

rotated_dataset_list=[]
rotation_cfg={    'x': 10,
    'y': 10,
    'z': 10,
    'order': 'xyz'}


#def augment(example, type="", values=()):
def augment(example):
    # do the transformation
    # return augmented data
    dim1, dim2= example['sign'].shape

    # bring the data into 3d
    example['sign'] = example['sign'].view((dim1, dim2, 576, 3)) # 576 for 1.728 dimensions
    #example['sign'] = example['sign'].view((dim1, dim2, 236, 3)) # 576 for 1.728 dimensions
    #example.sgn = example.sgn.view((dim1, dim2, 236, 3))  # 236 for 708 dimensions

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
    example['sign'] = torch.matmul(example['sign'], rot)
    example['sign'] = example['sign'].view((dim1, dim2, 1728))
    #example.sgn = example.sgn.view((dim1, dim2, 708))

    dict_instance_rotated = {
        'name': example['sequence'],
        'signer': example['signer'],
        'gloss': example['gls'],
        'text': example['txt'],
        'sign': example['sign'],
    }

    #rotated_dataset_list.append(dict_instance_rotated)
    return dict_instance_rotated


#augmentations = {"rotation": [(1, 2, 2), (1, 2, 2), (1, 2, 2)], "scale": [(1)]}


def augment_features(features):
    out = deepcopy(features)

    #out=[]
    for entry in features:
        #for aug_type in augmentations:
            #or params in augmentations[aug_type]:
                #etentry = deepcopy(entry)
                #etentry["sign"] = augment(etentry["sign"], type=aug_type, values=params)
        etentry = deepcopy(entry)
        etentry["sign"] = augment(etentry)
        out.append(etentry)
                #out.append(etentry)
    return out

if __name__ == '__main__':
    rotated_dataset = read_gzip(r"C:\\Users\\areeb\\Downloads\\SignLanguage_DataAugmentation\\slt\\signjoey.test")
    rotated_dataset= augment_features(rotated_dataset)
    print(rotated_dataset)
