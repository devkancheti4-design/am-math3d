"""3DTrajMaster's pose numbers, rebuilt from its dataset files: the convention its inference code feeds the model, written
here independently (its repository carries no license, so none of its code is copied).
  The files store 4 x 4 matrices row by row as text, '[a b c d] [e f g h] ...', with the translation in the last row,
  in Unreal units (cm) and Unreal axes (forward, right, up).  The model reads each pose as columns: the axes taken in
  the order right, up, forward, the translation in metres, every entity relative to its location's camera target, and
  the whole expressed in the frame of the camera whose video it is learning (camera 12 at inference)."""
import json, os
import numpy as np

AXES = [1, 2, 0, 3]                                             # (forward, right, up, t) -> (right, up, forward, t)


def matrix(text):
    rows = [r.strip(" []") for r in text.strip().split("] [")]
    return np.array([[float(v) for v in r.split()] for r in rows])


def as_columns(M, origin_cm=(0.0, 0.0, 0.0)):
    """a stored matrix -> the model's 4 x 4: columns right, up, forward, translation (metres, from origin_cm)"""
    P = M.T.copy()
    P[:3, 3] = (P[:3, 3] - np.asarray(origin_cm, float)) / 100.0
    return P[:, AXES]


def cameras(root):
    """the dataset's 12 cameras around the target (Hemi12), in the order the model indexes them"""
    cams = json.load(open(os.path.join(root, "Hemi12_transforms.json")))
    return np.stack([as_columns(matrix(cams[k])) for k in cams if "C_" in k])


def poses_in_cam(root, video_name, cam_index=12, scene="Desert"):
    """every entity, every one of the 99 frames: its 4 x 4 pose in the camera's frame (x right, y up, z forward, m)"""
    objs = json.load(open(os.path.join(root, scene, video_name, video_name + ".json")))
    locs = {l["name"]: l for l in json.load(open(os.path.join(root, scene, "location_data.json")))}
    target = locs[video_name.split("_")[1]]["coordinates"]["CameraTarget"]["position"]
    world_to_cam = np.linalg.inv(cameras(root)[cam_index - 1])
    frames = sorted(objs, key=int)
    return np.stack([np.stack([world_to_cam @ as_columns(matrix(objs[f][e]["matrix"]), target) for f in frames])
                     for e in range(len(objs[frames[0]]))])
