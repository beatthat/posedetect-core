import os
import pytest

import numpy as np
from pyshould import should
from pyshould.expect import expect

from posedetect_core.d2.boxes_and_keypoints import detect_boxes_and_keypoints_task
from .test_helpers import (
    expect_progress_callbacks,
    expect_person_class_box_and_keypoint_detections_for_every_frame,
    expect_box_detections_for_every_frame_are_5_element_arrays,
    expect_number_of_keypoints_detections_on_each_frame_matches_the_number_of_box_detections,
    expect_keypoints_detections_for_every_frame_are_4x17_ndarrays_for_17_keypoints,
    expect_keypoints_detections_for_every_frame_accurate_within_margin_of_error,
    fixture_path,
    get_video_dims,
)


class __AttrDict(dict):
    def __getattr__(self, item):
        return self[item]


@pytest.mark.parametrize(
    "video_dir,expected_frames_header,expected_frames_processed",
    [("resources/videos/one_person_no_cuts_01", 73, 61)],
)
def it_accurately_estimates_boxes_and_keypoints_for_a_video(
    video_dir, expected_frames_header, expected_frames_processed
):
    collected_progress = []

    def collect_progress(state, meta):
        collected_progress.append((state, meta))

    # we need a fake celery task to run the detect job
    task = __AttrDict(
        {"request": __AttrDict({"id": "test"}), "update_state": collect_progress}
    )
    video_dir = fixture_path(video_dir)
    video_path = os.path.join(video_dir, "video.mp4")
    video_dims = get_video_dims(video_path)
    result_meta = detect_boxes_and_keypoints_task(task, video_path, "anything{}")
    expect(result_meta).to_have_the_entry("current", 61)
    expect(result_meta).to_have_the_entry("total", 61)
    expect(result_meta).to_have_the_entry("status", "success")
    expect(result_meta).to_have_the_entry("result", "anythingtest")
    expect(result_meta).to_have_the_key("result_file")
    expect(result_meta["result_file"]).to_end_with("boxes_and_keypoints.npz")
    expect_progress_callbacks(
        collected_progress, expected_frames_header, expected_frames_processed
    )
    ground_truth_path = os.path.join(video_dir, "keypoints_ground_truth.npz")
    expected_result = np.load(ground_truth_path)
    result_npz = np.load(result_meta["result_file"])
    result = {k: result_npz[k] for k in result_npz.files}
    expect_person_class_box_and_keypoint_detections_for_every_frame(
        expected_result, result
    )
    expect_box_detections_for_every_frame_are_5_element_arrays(result)
    expect_number_of_keypoints_detections_on_each_frame_matches_the_number_of_box_detections(
        result
    )
    expect_keypoints_detections_for_every_frame_are_4x17_ndarrays_for_17_keypoints(
        result
    )
    expect_keypoints_detections_for_every_frame_accurate_within_margin_of_error(
        expected_result, result, video_dims
    )
