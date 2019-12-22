from os import path
from typing import Tuple

import cv2
from hamcrest import assert_that, equal_to, less_than, only_contains
import numpy as np
from pyshould import should, all_of

import posedetect_core


def fixture_path(rel_path: str) -> str:
    return path.join(
        path.dirname(path.abspath(posedetect_core.__file__)),
        "data",
        "fixtures",
        rel_path,
    )


def expect_progress_callbacks(
    collected_progress, expected_frames_header, expected_frames_processed
):
    assert expected_frames_processed == len(
        collected_progress
    ), "should receive one progress update per frame"
    assert [
        (
            "PROGRESS",
            {
                "current": i,
                "total": expected_frames_header,
                "status": "processing video frames",
            },
        )
        for i, x in enumerate(collected_progress)
    ] == collected_progress, "should receive one progress update per frame"


def expect_person_class_box_and_keypoint_detections_for_every_frame(
    expected_result, result
):
    boxes_gt = expected_result["boxes"]
    boxes_result = result["boxes"]
    len(boxes_result) | should.be_equal_to(len(boxes_gt))
    keypoints_gt = expected_result["keypoints"]
    keypoints_result = result["keypoints"]
    len(keypoints_result) | should.be_equal_to(len(keypoints_gt))
    cls_ix_person = get_person_class_index()
    # each element in boxes is a list indexed by class ids,
    # e.g. boxes[i][cls_ix_person] are box detections for cls_ix_person
    # (which happens to be 1) in frame i
    all_of([len(x) for x in boxes_result]).should.be_greater_than(cls_ix_person)
    # # the same goes for keypoints...
    all_of([len(x) for x in keypoints_result]).should.be_greater_than(cls_ix_person)


def expect_box_detections_for_every_frame_are_5_element_arrays(result):
    """
    the result box detections for every frame are 5-element arrays (xmin, xmax, ymin, ymax, prob)')
    """
    boxes_result = result["boxes"]
    cls_ix_person = get_person_class_index()
    all_of(
        [frame_cls_boxes[cls_ix_person].shape[1] for frame_cls_boxes in boxes_result],
        equal_to(5),
    )


def expect_number_of_keypoints_detections_on_each_frame_matches_the_number_of_box_detections(
    result
):
    boxes_result = result["boxes"]
    kps_result = result["keypoints"]
    cls_ix_person = get_person_class_index()
    kps_count_per_frame = [
        len(frame_cls_kps[cls_ix_person]) for frame_cls_kps in kps_result
    ]
    boxes_count_per_frame = [
        len(frame_cls_boxes[cls_ix_person]) for frame_cls_boxes in boxes_result
    ]
    kps_count_per_frame | should.contain(*boxes_count_per_frame)


def expect_keypoints_detections_for_every_frame_are_4x17_ndarrays_for_17_keypoints(
    result
):
    kps_result = result["keypoints"]
    cls_ix_person = get_person_class_index()
    for frame, cls_kps in enumerate(kps_result):
        frame_kps_shapes = [x.shape for x in cls_kps[cls_ix_person]]
        assert_that(
            frame_kps_shapes,
            only_contains((4, 17)),
            "shape of detected cls_keypoints for frame {} cls_ix={}".format(
                frame, cls_ix_person
            ),
        )


def expect_3d_estimates_for_every_frame_for_same_number_of_keypoints_as_ground_truth(
    expected_3d_kps, result_3d_kps
):
    type(expected_3d_kps) | should.be_equal_to(np.ndarray)
    type(result_3d_kps) | should.be_equal_to(np.ndarray)
    result_3d_kps.shape | should.be_equal_to(expected_3d_kps.shape)


def expect_3d_keypoint_estimates_for_every_frame_accurate_within_margin_of_error(
    expected_result, result_kps, video_dims
):
    normalizer = np.ones((17, 3))
    normalizer[:, 0] *= video_dims[0]
    normalizer[:, 1] *= video_dims[1]
    normalizer[:, 1] *= max(video_dims[0], video_dims[1])
    frames_sqe = []
    for frame, pair in enumerate(zip(result_kps, expected_result)):
        frame_res, frame_gt = pair
        fsqe = np.sum(((frame_res - frame_gt) / normalizer) ** 2)
        frames_sqe.append(fsqe)
    frames_sqe = np.array(frames_sqe, dtype=float)
    msqe = np.mean(frames_sqe)
    max_normalized_error = 0.03
    msqe | should.be_less_than(
        max_normalized_error
    )  # temp, must normalize error and come up w a max for msqe


def expect_keypoints_detections_for_every_frame_accurate_within_margin_of_error(
    expected_result, result, video_dims
):
    result_boxes = result["boxes"]
    result_kps = result["keypoints"]
    expected_kps = expected_result["keypoints"]
    cls_ix_person = get_person_class_index()
    result_kps_track_one = track_one(result_boxes, result_kps, cls_ix_person)[
        "keypoints"
    ]
    missed_frame_error = 1.0  # TODO: normalize all errors
    normalizer = np.ones((2, 17))
    normalizer[0] = normalizer[0] * video_dims[0]
    normalizer[1] = normalizer[1] * video_dims[1]
    frames_sqe = []
    for frame, pair in enumerate(zip(result_kps_track_one, expected_kps)):
        frame_res = pair[0]
        frame_gt = pair[1]
        # number of detections in the ground-truth data
        # needs to be at most 1 per frame.
        # This does NOT imply detectron is supposed to detect
        # no more than one person per frame.
        # Rather, the ground-truth data is polished to ensure
        # a single 'tracked' subject
        assert_that(
            len(frame_gt),
            less_than(2),
            "number of detections in ground truth (should be at most 1 per frame)",
        )
        # in result from track_one, every frame should have either
        #    - a list with *ONE* set of keypoints
        #    or
        #    - an empty list
        if len(frame_res) != len(frame_gt):
            frames_sqe.append(missed_frame_error)
            continue
        if len(frame_res) == 0:
            # correctly detected frame with no person
            # from each frame, we want the keypoints
            # and then only the first two elements, x and y
            continue
        frame_res = frame_res[0][2:]
        frame_gt = frame_gt[0][2:]
        fsqe = np.sum(((frame_res - frame_gt) / normalizer) ** 2)
        frames_sqe.append(fsqe)
    frames_sqe = np.array(frames_sqe, dtype=float)
    msqe = np.mean(frames_sqe)
    max_normalized_error = 0.03
    msqe | should.be_less_than(
        max_normalized_error
    )  # temp, must normalize error and come up w a max for msqe


def get_person_class_index():
    return 1


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


def highest_confidence_box(boxes):
    """
    given an enumerable of boxes, each an array in form [ xmin, ymin, xmax, ymax, confidence]
    return the box with the highest confidence
    """
    i, b = max(enumerate(boxes), key=lambda e: e[1][4])
    return i, b


def get_video_dims(video_path: str) -> Tuple[int, int]:
    vid = cv2.VideoCapture(video_path)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    return (width, height)
