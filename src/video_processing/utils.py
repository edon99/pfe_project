import yaml
import json
from torch import cuda


def get_labels_dics():
    # Get tactical map keypoints positions dictionary
    json_path = "src/constants/2d-map-labels-position.json"
    with open(json_path, 'r') as f:
        keypoints_map_pos = json.load(f)

    yaml_path = "src/constants/index.yaml"
    with open(yaml_path, 'r') as file:
        classes_names_dic = yaml.safe_load(file)
    labels_dic = classes_names_dic['ball-players']
    classes_names_dic = classes_names_dic['pitch-keypoints']

    return keypoints_map_pos, classes_names_dic, labels_dic


def get_device_type():
    if cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'
