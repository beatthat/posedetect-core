import numpy as np
from .results import highest_confidence_box


def track_one(
    boxes_frames, keypoint_frames, cls_ix=1  # class index for person in detectron
):
    """
    for testing, assuming we have a sequence of box/keypoints detections,
    e.g. from a video, and that the video has a single (person) subject,
    we want to extract the box and keypoints FOR JUST THAT ONE (person)
    on each frame
    Args:
        boxes_frames: list of frames and for each frame,
            cls_boxes as output by detectron
        keypoint_frames: list of frames and for each frame,
            cls_keypoints as output by detectron
        cls_ix: what detectron class are we looking for (1/person by default)
    Returns:
        dict {
            boxes: a list of frames but now for each frame
                there is a list containing single box (or none)
            keypoints: a list of frames but now for each frame
                there is a list containing keypoints
                for only a single subject (or none)
        }
    """
    result_boxes = []
    result_keypoints = []
    for f, cls_boxes in enumerate(boxes_frames):
        # No (person) detected on this frame so add an empty list
        # for result boxes and keypoints.
        # This is why we wrap the results for each frame
        # in a list: handle the case of frames with no detection
        if len(cls_boxes) == 0:
            result_boxes.append([])
            result_keypoints.append([])
            continue
        bIx, b = highest_confidence_box(cls_boxes[cls_ix])
        result_boxes.append([b])
        result_keypoints.append([keypoint_frames[f][cls_ix][bIx]])
    return dict(boxes=result_boxes, keypoints=result_keypoints)


def extract_data(npz_file):
    """
    given a data archive or (PathLike to a data archive)
    ensure the archive is loaded and then returns
    the two top-level components.
    This function exists mainly to document
    the structure of .npz data archives included with VideoPose3D in the data folder
    and simplify loading/access.
    Args:
        npz_file (PathLike|numpy.lib.npyio.NpzFile): NpzFile or path to NpzFile
    Returns:
        positions_2d Dictionary of Subject => Action => [camera, frame, keypoint, dim]
        metadata Dictionary of metadata about positions_2d
    """
    if not isinstance(npz_file, np.lib.npyio.NpzFile):
        npz_file = np.load(npz_file)
    positions_2d = npz_file["positions_2d"].item()
    metadata = npz_file["metadata"].item()
    return positions_2d, metadata


def get_poses(positions_2d, subject="S1", action="Default", camera=0):
    """
    given the positions_2d dictionary from a VideoPose3D data archive @see extract_data
    and a subject, action and camera index,
    returns the numpy ndarray of poses with shape
    [n_frames, n_key_points, n_dimensions (e.g. 2 for x, y)]
    Args:
        positions_2d Dictionary of Subject => Action => [camera, frame, keypoint, dim]
        subject The subject to use, default is 'S1'
        action The action to use, default is 'Default' but common real example is 'Walking 1'
        camera The camera to use, default is 0
    Returns:
        numpy.ndarray of poses with shape [n_frames, n_keypoints_per_frame, n_dims_per_keypoint]
    Example:
        data_file = np.load('data_2d_h36m_detectron_ft_h36m.npz')
        positions_2d, = extract_data(data_file)
        poses = get_poses(positions_2d, action='Walking 1')
        print(poses.shape) # [3000, 17, 2] 3000 frames, 17 keypoints/frame, 2 dims/keypoint (x,y)
    """

    if not isinstance(positions_2d, dict):
        # print(f'expected positions_2d dict, encountered {type(positions_2d)}')
        return None

    subject_data = positions_2d[subject]

    if not isinstance(subject_data, dict):
        # print(f'no subject {subject} found in data')
        return None

    action_data = subject_data[action]

    if not isinstance(action_data, list):
        # print(f'no action {action} found for {subject} in data')
        return None

    if len(action_data) < camera + 1:
        # print(f'no camera {camera} found for action {action} and {subject} in data')
        return None

    poses = action_data[camera]

    return poses


def poses_to_archive(poses, subject="S1", action="Default"):
    """
    given a sequence of poses, create an archive
    in the format required by VideoPose3D.
    """

    s = dict()
    s[action] = np.array([poses])  # array of cameras but we have only one camera
    positions_2d = dict()
    positions_2d[subject] = s

    metadata = dict(
        {
            "layout_name": "h36m",
            "num_joints": 17,
            "keypoints_symmetry": np.array(
                [[4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]]
            ),
        }
    )

    archive = dict({"positions_2d": positions_2d, "metadata": metadata})

    return archive
