from ultralytics import YOLO

def load_players_model():
    return YOLO('src/models/players.pt')
def load_keypoints_model():
    return YOLO('src/models/keypoints.pt')
def load_models():
    return load_players_model(), load_keypoints_model()