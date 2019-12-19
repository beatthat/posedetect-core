from abc import ABCMeta, abstractmethod


class KeypointProcessorInterface:
    __metaclass__ = ABCMeta

    @abstractmethod
    def execute(self):
        pass

    @abstractmethod
    def on_frame(self, im, i, cls_boxes, cls_segms, cls_keyps):
        """
        Args:
            im - the image frame
            i - the index of the image in the sequence
            cls_boxes - detected boxes (of person class)
            cls_segms - detected segments
            cls_keyps - detected key points
        """
        pass

    @abstractmethod
    def on_done(self):
        """
        """
        pass


class KeypointProcessor(KeypointProcessorInterface):
    def __init__(self, keypoints_gen, on_frame_complete=None):
        """
        Args:
            keypoints_gen: (generator) yields detections for a series of frames
                For each frame yields the input for a call to on_frame

            on_frame_complete: (function(int framenum))
                Optional function called at the end of processing each frame
                with frame num. This is useful for providing progress feedback
                on video-process jobs that may be long running.
        """
        self.keypoints_gen = keypoints_gen
        self.on_frame_complete = on_frame_complete

    def execute(self):
        for i, im, cls_boxes, cls_segms, cls_keyps in self.keypoints_gen:
            self.on_frame(im, i, cls_boxes, cls_segms, cls_keyps)
            if self.on_frame_complete:
                self.on_frame_complete(i)

        if self.on_done is not None:
            self.on_done()

    def on_done(self):
        pass


class CollectBoxesAndKeypoints(KeypointProcessor):
    """
    Collects a list of boxes and keypoints
    """

    def __init__(self, keypoints_gen, on_frame_complete=None):
        """
        Args:
            keypoints_gen: (generator) yields detections for a series of frames
                For each frame yields the input for a call to on_frame

            on_frame_complete: (function(int framenum))
                Optional function called at the end of processing each frame
                with frame num. This is useful for providing progress feedback
                on video-process jobs that may be long running.
        """
        self.keypoints = []
        self.boxes = []
        super(CollectBoxesAndKeypoints, self).__init__(keypoints_gen, on_frame_complete)

    # def execute(): super().execute()

    def on_frame(self, im, i, cls_boxes, cls_segms, cls_keyps):
        """
        Args:
            im - the image frame
            i - the index of the image in the sequence
            cls_boxes - detected boxes (of person class)
            cls_segms - detected segments
            cls_keyps - detected key points
        """

        self.boxes.append(cls_boxes)
        self.keypoints.append(cls_keyps)

    def get_boxes(self):
        return self.boxes

    def get_keypoints(self):
        return self.keypoints
