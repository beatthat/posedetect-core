# import os
# import pytest

# import numpy as np
# from pyshould import should

# from posedetect_core.d2.boxes_and_keypoints import detect_boxes_and_keypoints_task
# from .test_helpers import (
#     get_video_dims,
#     expect_progress_callbacks,
#     expect_person_class_box_and_keypoint_detections_for_every_frame,
#     expect_box_detections_for_every_frame_are_5_element_arrays,
#     expect_number_of_keypoints_detections_on_each_frame_matches_the_number_of_box_detections,
#     expect_keypoints_detections_for_every_frame_are_4x17_ndarrays_for_17_keypoints,
#     expect_keypoints_detections_for_every_frame_accurate_within_margin_of_error,
# )


# class __AttrDict(dict):
#     def __getattr__(self, item):
#         return self[item]


# @pytest.mark.parametrize(
#     "video_dir,expected_frames_header,expected_frames_processed",
#     [("/tests/api/features/resources/videos/one_person_no_cuts_01", 73, 61)],
# )
# def test_detect_boxes_and_keypoints_task(
#     video_dir, expected_frames_header, expected_frames_processed
# ):
#     collected_progress = []

#     def collect_progress(state, meta):
#         collected_progress.append((state, meta))

#     # we need a fake celery task to run the detect job
#     task = __AttrDict(
#         {"request": __AttrDict({"id": "test"}), "update_state": collect_progress}
#     )
#     video_path = os.path.join(video_dir, "video.mp4")
#     video_dims = get_video_dims(video_path)
#     result_meta = detect_boxes_and_keypoints_task(task, video_path, "anything{}")
#     result_meta | should.be_equal_to(
#         {
#             "current": 61,
#             "total": 61,
#             "status": "success",
#             "result": "anythingtest",
#             "result_file": "/tests/api/features/resources/videos/one_person_no_cuts_01/out/boxes_and_keypoints.npz",
#         }
#     )
#     expect_progress_callbacks(
#         collected_progress, expected_frames_header, expected_frames_processed
#     )
#     ground_truth_path = os.path.join(video_dir, "keypoints_ground_truth.npz")
#     expected_result = np.load(ground_truth_path)
#     result_npz = np.load(result_meta["result_file"])
#     result = {k: result_npz[k] for k in result_npz.files}
#     expect_person_class_box_and_keypoint_detections_for_every_frame(
#         expected_result, result
#     )
#     expect_box_detections_for_every_frame_are_5_element_arrays(result)
#     expect_number_of_keypoints_detections_on_each_frame_matches_the_number_of_box_detections(
#         result
#     )
#     expect_keypoints_detections_for_every_frame_are_4x17_ndarrays_for_17_keypoints(
#         result
#     )
#     expect_keypoints_detections_for_every_frame_accurate_within_margin_of_error(
#         expected_result, result, video_dims
#     )
