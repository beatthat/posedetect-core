from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os

from .frame_generators import Video2Frames
from .keypoint_processors import CollectBoxesAndKeypoints
from .boxes_and_keypoints import create_boxes_and_keypoints_model


class PoseEstimationService:
    def __init__(self):
        boxes_and_keypoints_module = os.environ.get(
            "BOXES_AND_KEYPOINTS_MODULE", "posedetect_detectron.d2"
        )
        self.logger = logging.getLogger(__name__)
        self.model = create_boxes_and_keypoints_model(boxes_and_keypoints_module)

    def poses_from_video(self, video, on_progress=None):
        """
            Args:
                video: (str) path to a video file

                on_progress: (function(cur_frame, tot_frames))
                    Optional progress function, called on the completion
                    of each frame

            Returns:
                dict(keypoints, boxes)
        """
        v2f = Video2Frames(video)
        # set up a generator that yields all the image frames from our video
        images = v2f.generator()

        def _on_frame_complete(i):
            self.logger.info("_on_frame_complete {}".format(i))
            if on_progress:
                on_progress(i, v2f.count)

        # set up a process that will collect keypoints and boxes
        # for our sequence of image frames
        process = CollectBoxesAndKeypoints(
            self.model.generate_predictions(images),
            on_frame_complete=_on_frame_complete,
        )
        process.execute()
        return dict(keypoints=process.get_keypoints(), boxes=process.get_boxes())
