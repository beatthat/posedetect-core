from abc import abstractmethod, ABC
import os
from importlib import import_module

import numpy as np


def detect_boxes_and_keypoints_task(task, video_path, result_url_format):
    from posedetect_core.d2.services import PoseEstimationService

    pose_estimation_service = PoseEstimationService()
    print("boxes_and_keypoints for {}".format(video_path))
    info = dict(current_frame=0, total_frames=0)

    def _on_progress(current_frame, total_frames):
        info["total"] = total_frames
        info["current"] = current_frame
        task.update_state(
            state="PROGRESS",
            meta={
                "current": current_frame,
                "total": total_frames,
                "status": "processing video frames",
            },
        )

    res_dict = pose_estimation_service.poses_from_video(
        video_path, on_progress=_on_progress
    )
    result_filename = "boxes_and_keypoints.npz"
    output_dir = os.path.join(os.path.dirname(video_path), "out")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, result_filename)
    np.savez(output_path, **res_dict)
    result_url = result_url_format.format(task.request.id)
    return {
        "current": info["current"] + 1,
        "total": info["current"] + 1,
        "status": "success",
        "result_file": output_path,
        "result": result_url,
    }


class BoxesAndKeypointsModel(ABC):
    @abstractmethod
    def generate_predictions(self, images):
        """Generator yields inferred boxes and keypoints for a sequence of images

        Args:
            images: iterable images

        Returns:
            yields for each image
                i: (int) the image index

                im: the image

                cls_boxes: (list) where each element represents a class (e.g. person is index 1)
                    and the contents of each element is a list bounding boxes
                    of detected regions of interest for the class.
                    The format of each detected box is as follows
                    [xMin, xMax, yMin, yMax, probability]
                 cls_segms: (list) where each element represents a class (e.g. person is index 1)
                    and the contents of each element is segmentation data.
                    The segments for each class correspond to the detected boxes
                    from cls_boxes, so typically the first step is always to look
                    at boxes and their probabilities etc.
                 cls_keyps: (list) where each element represents a class (e.g. person is index 1)
                    and the contents of each element is a list of keypoint detections.

                    The keypoint detections for each class correspond to the detected boxes
                    from cls_boxes, so typically the first step is always to look
                    at boxes and their probabilities etc.
        """
        yield 0, None, [], [], []


class BoxesAndKeypointsModelFactory(ABC):
    """
    A factory that creates a BoxesAndKeypointsModel
    """

    @abstractmethod
    def create(self):
        return None


__factories_by_module_path = {}


def create_boxes_and_keypoints_model(module_path):
    """
        Creates a BoxesAndKeypointsModel given a module path

        Args:
            module_path: (str) path to the module (which should register its BoxesAndKeypointsModelFactory on import)
        Returns:
            model: (BoxesAndKeypointsModel)
    """
    print("create_boxes_and_keypoints_model path={}".format(module_path))
    return create_boxes_and_keypoints_model_factory(module_path).create()


def register_boxes_and_keypoints_model_factory(module_path, fac):
    """
        Register a BoxesAndKeypointsModelFactory for a module_path

        Args:
            module_path: (str) path to the module
            fac: (BoxesAndKeypointsModelFactory) the factory
    """
    print("register_boxes_and_keypoints_model_factory path={}".format(module_path))
    assert isinstance(fac, BoxesAndKeypointsModelFactory)
    __factories_by_module_path[module_path] = fac


def create_boxes_and_keypoints_model_factory(module_path):
    """
        Creates a BoxesAndKeypointsModelFactory given module path.

        Args:
            module_path: (str) path to the module (which should register its BoxesAndKeypointsModelFactory on import)
        Returns:
            model: (BoxesAndKeypointsModelFactory)
    """
    if module_path not in __factories_by_module_path:
        import_module(module_path)
    fac = __factories_by_module_path[module_path]
    assert isinstance(fac, BoxesAndKeypointsModelFactory)
    return fac
