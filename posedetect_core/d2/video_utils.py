import cv2
import logging
from os import path

def get_video_dims(video_path):
    vid = cv2.VideoCapture(video_path)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    return (width, height)


def get_frame_count(video_path_or_capture):
    logging.warning(f"get_frame_count for path {path.abspath(video_path_or_capture)}")
    cap = cv2.VideoCapture(video_path_or_capture)
    property_id = int(cv2.CAP_PROP_FRAME_COUNT)
    return int(cv2.VideoCapture.get(cap, property_id))
