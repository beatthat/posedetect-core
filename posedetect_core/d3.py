from abc import abstractmethod, ABC
from importlib import import_module
from numpy import ndarray
from typing import Any, Callable, Dict, Optional


class PoseEstimates3DModel(ABC):
    @abstractmethod
    def predict(
        self,
        video_path: str,
        on_progress_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> ndarray:
        raise NotImplementedError()

    @abstractmethod
    def visualize(
        self,
        dataset,
        input_keypoints,
        prediction,
        input_video_path,
        output_video_path,
        width_of=1080,
        height_of=1080,
        bitrate=3000,
        camera=0,
        downsample=1,
        limit=-1,
        size=5,
        skip=0,
        subject="S1",
        test_time_augmentation=False,
        **kwargs
    ):
        raise NotImplementedError()


class PoseEstimates3DModelFactory(ABC):
    """
    A factory that creates a PoseEstimates3DModel
    """

    @abstractmethod
    def create(self) -> PoseEstimates3DModel:
        raise NotImplementedError()


__factories_by_module_path = {}


def create_model(module_path: str) -> PoseEstimates3DModel:
    """
        Creates a PoseEstimates3DModel given a module path

        Args:
            module_path: (str) path to the module (which should register its PoseEstimates3DModelFactory on import)
        Returns:
            model: (PoseEstimates3DModel)
    """
    print("create_model path={}".format(module_path))
    return create_model_factory(module_path).create()


def register_model_factory(module_path: str, fac: PoseEstimates3DModelFactory) -> None:
    """
        Register a PoseEstimates3DModelFactory for a module_path

        Args:
            module_path: (str) path to the module
            fac: (PoseEstimates3DModelFactory) the factory
    """
    print("register_boxes_and_keypoints_model_factory path={}".format(module_path))
    assert isinstance(fac, PoseEstimates3DModelFactory)
    __factories_by_module_path[module_path] = fac


def create_model_factory(module_path) -> PoseEstimates3DModelFactory:
    """
        Creates a PoseEstimates3DModelFactory given module path.

        Args:
            module_path: (str) path to the module (which should register its PoseEstimates3DModelFactory on import)
        Returns:
            model: (PoseEstimates3DModelFactory)
    """
    if module_path not in __factories_by_module_path:
        import_module(module_path)
    fac = __factories_by_module_path[module_path]
    assert isinstance(fac, PoseEstimates3DModelFactory)
    return fac
