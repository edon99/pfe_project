import yaml
import json

def get_labels_dics():
    # Get tactical map keypoints positions dictionary
    json_path = "../constants/2d-map-labels-position.json"
    with open(json_path, 'r') as f:
        keypoints_map_pos = json.load(f)

    yaml_path = "../constants/classes.yaml"
    with open(yaml_path, 'r') as file:
        classes_names_dic = yaml.safe_load(file)
    ball_players_dic = classes_names_dic['ball-players']
    pitch_keypoints_dic = classes_names_dic['pitch-keypoints']

    return ball_players_dic, pitch_keypoints_dic, keypoints_map_pos