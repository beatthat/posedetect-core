import cv2
import math
import time


class Video2Frames(object):
    def __init__(
        self, video_path, frame_start=0, frame_count=None, frame_stride=1, logger=None
    ):
        """
        Provides a generator that yields image frames given a video path,
        while also providing the total frames in the video.

        Args:
            video_path (PathLike): path to a video file

            frame_start: frame number to yield first. Defaults to 0.

            frame_count: number of frames to yield total. Defaults None (end of video)

            frame_stride: yield every Nth frame. Defaults to 1.

            logging (bool, optional): enable verbose console logging. Defaults to False.
        """
        self.video_path = video_path
        self.frame_start = frame_start
        self.frame_count = frame_count
        self.frame_stride = frame_stride
        self.logger = logger
        self.__new_cap()

    def __del__(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def __new_cap(self):
        self.cap = cv2.VideoCapture(self.video_path)
        property_id = int(cv2.CAP_PROP_FRAME_COUNT)
        self.frames_total = int(cv2.VideoCapture.get(self.cap, property_id))
        self.count = int(
            math.floor((self.frames_total - self.frame_start) / self.frame_stride)
        )

    def generator(self):
        """
            Returns:
                yields image frames
        """
        if not self.cap:
            self.__new_cap()
        f = 0
        while self.cap.isOpened():
            yf = f - self.frame_start
            f += 1
            ret, frame = self.cap.read()
            if yf < 0 or yf % self.frame_stride != 0:
                continue
            yf /= self.frame_stride
            if self.frame_count is not None and yf >= self.frame_count:
                break
            t = time.time()
            if not ret:
                break
            if self.logger:
                self.logger.info(
                    "read frame {} and yield as {} in {}s".format(
                        str(f), str(yf), str(time.time() - t)
                    )
                )
            yield frame
        self.cap.release()
        self.cap = None
