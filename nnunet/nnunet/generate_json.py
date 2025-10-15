# 该功能可实现nnUNet的Json文件生成

import os

from batchgenerators.utilities.file_and_folder_operations import save_json, subfiles
from typing import Tuple
import numpy as np


# 获取文件夹内独立文件 【列表】

def get_identifiers_from_splitted_files(folder: str):
    uniques = np.unique([i[:-7] for i in subfiles(folder, suffix='.nii.gz', join=False)])
    return uniques


def generate_dataset_json(output_file: str, imagesTr_dir: str, imagesTs_dir: str, modalities: Tuple,
                          labels: dict, dataset_name: str, license: str = "Hebut AI", dataset_description: str = "",
                          dataset_reference="oai-zib", dataset_release='11/2021'):
    """
    :param output_file: This needs to be the full path to the dataset.json you intend to write, so
    output_file='DATASET_PATH/dataset.json' where the folder DATASET_PATH points to is the one with the
    imagesTr and labelsTr subfolders
    :param imagesTr_dir: path to the imagesTr folder of that dataset
    :param imagesTs_dir: path to the imagesTs folder of that dataset. Can be None
    :param modalities: tuple of strings with modality names. must be in the same order as the images (first entry
    corresponds to _0000.nii.gz, etc). Example: ('T1', 'T2', 'FLAIR').
    :param labels: dict with int->str (key->value) mapping the label IDs to label names. Note that 0 is always
    supposed to be background! Example: {0: 'background', 1: 'edema', 2: 'enhancing tumor'}
    :param dataset_name: The name of the dataset. Can be anything you want
    :param license:
    :param dataset_description:
    :param dataset_reference: website of the dataset, if available
    :param dataset_release:
    :return:
    """
    # 获取文件夹内各个独立的文件
    train_identifiers = get_identifiers_from_splitted_files(imagesTr_dir)
    # imagesTs_dir 文件夹可以为空，只要有训练的就行
    if imagesTs_dir is not None:
        test_identifiers = get_identifiers_from_splitted_files(imagesTs_dir)
    else:
        test_identifiers = []

    json_dict = {}
    json_dict['name'] = "tumor"
    json_dict['description'] = "mask Segmentation"
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = dataset_reference
    json_dict['licence'] = license
    json_dict['release'] = dataset_release
    json_dict['modality'] = {"0": "CT"}   # 这里填入模态信息，0表示只有一个模态，还可以加入“1”：“MRI”之类的描述，
    json_dict['labels'] = {
        "0": "background",
        "1": "lnm",
    }

    # 下面这些内容不需要查看和更改
    json_dict['numTraining'] = len(train_identifiers)
    json_dict['numTest'] = len(test_identifiers)
    json_dict['training'] = [
        {'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i
        in
        train_identifiers]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in test_identifiers]

    output_file += "dataset.json"
    if not output_file.endswith("dataset.json"):
        print("WARNING: output file name is not dataset.json! This may be intentional or not. You decide. "
              "Proceeding anyways...")
    save_json(json_dict, os.path.join(output_file))


if __name__ == "__main__":
    # 自行修改文件路径，当前在windows环境下操作
    output_file = r'D:\segmentation\nnUNet-Windows-master\DATASET\nnUNet_raw\nnUNet_raw_data\Task09_veintumor'
    imagesTr_dir = r'D:\segmentation\nnUNet-Windows-master\DATASET\nnUNet_raw\nnUNet_raw_data\Task09_veintumor\imagesTr'
    imagesTs_dir = r'D:\segmentation\nnUNet-Windows-master\DATASET\nnUNet_raw\nnUNet_raw_data\Task09_veintumor\imagesTs'
    labelsTr = r'D:\segmentation\nnUNet-Windows-master\DATASET\nnUNet_raw\nnUNet_raw_data\Task09_veintumor\labelsTr'





    # 只需要给出空定义，具体内容在上面的函数中修改
    modalities = ''
    labels = {

    }
    get_identifiers_from_splitted_files(output_file)
    generate_dataset_json(output_file,
                          imagesTr_dir,
                          imagesTs_dir,
                          labelsTr,
                          modalities,
                          labels
                          )
