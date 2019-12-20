# import os
# import pytest

# import numpy as np
# from pyshould import should

# from posedetect_core.estimate_poses import estimate_poses_task
# from .test_helpers import (
#     expect_3d_keypoint_estimates_for_every_frame_accurate_within_margin_of_error,
#     get_video_dims,
#     fixture_path,
# )


# class __AttrDict(dict):
#     def __getattr__(self, item):
#         return self[item]


# @pytest.mark.parametrize(
#     "video_dir,video_file,expected_frames_header,expected_frames_processed",
#     [
#         ("resources/videos/one_person_no_cuts_01", "video.mp4", 73, 61),
#         (
#             "resources/videos/converts_mov_to_mp4",
#             "video.mov",
#             73,
#             62,  # NOTE: the transcode seems to add a frame,
#             # accomodating for this in the test for now,
#             # but may be a source of problems in the future?
#         ),
#     ],
# )
# def test_estimate_poses_task(
#     video_dir, video_file, expected_frames_header, expected_frames_processed
# ):
#     collected_progress = []

#     def collect_progress(state, meta):
#         collected_progress.append((state, meta))

#     # we need a fake celery task to run the detect job
#     task = __AttrDict(
#         {"request": __AttrDict({"id": "test"}), "update_state": collect_progress}
#     )
#     video_dir = fixture_path(video_dir)
#     video_path = os.path.join(video_dir, video_file)
#     video_dims = get_video_dims(video_path)
#     result_meta = estimate_poses_task(task, video_path, "anything{}")
#     result_meta | should.be_equal_to(
#         {
#             "status": "success",
#             "resultFile": f"{video_dir}/out/pose_estimations_3d.npz",
#             "resultUrl": "anythingtest",
#             "resultPoses3dUrl": "anythingtest/poses3d",
#             "resultPoses2dUrl": "anythingtest/poses2d",
#             "resultVisualizationFile": f"{video_dir}/out/pose_estimates_3d.mp4",
#             "resultVisualizationUrl": "anythingtest/visualization",
#         }
#     )
#     # expect_progress_callbacks(
#     #     collected_progress, expected_frames_header, expected_frames_processed
#     # )
#     result_npz = np.load(result_meta["resultFile"])
#     result = {k: result_npz[k] for k in result_npz.files}
#     result_kps_3d = result["output_keypoints_3d"]
#     ground_truth_path = os.path.join(video_dir, "keypoints_3d_ground_truth.npy")
#     expected_kps_3d = np.load(ground_truth_path)
#     expected_result_shape = (expected_frames_processed, 17, 3)
#     expected_kps_3d.shape | should.be_equal_to(expected_result_shape)
#     result_kps_3d.shape | should.be_equal_to(expected_result_shape)
#     expect_3d_keypoint_estimates_for_every_frame_accurate_within_margin_of_error(
#         expected_kps_3d, result_kps_3d, video_dims
#     )
