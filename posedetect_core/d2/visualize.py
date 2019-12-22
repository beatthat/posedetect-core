from detectron.datasets import dummy_datasets
from detectron.utils.keypoints import get_keypoints
from detectron.utils.vis import vis_one_image

from .keypoint_processors import KeypointProcessor


class Frames2Visualizations(KeypointProcessor):

    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    def __init__(self, keypoints_gen, output_dir):
        """
        return functions for on_frame and on_done
        that generates a visualization pdf
        for each frame of the input video (or images)

        Args:
            keypoints_gen: (generator) yields detections for a series of frames
                For each frame yields the input for a call to on_frame

            output_dir - the directory to which visualization files will be written

        Returns:
            a frame-action function (see details below)
        """
        self.output_dir = output_dir
        super(Frames2Visualizations, self).__init__(keypoints_gen)

    def on_frame(self, im, i, cls_boxes, cls_segms, cls_keyps):
        """
        Args:
            im - the image frame
            i - the index of the image in the sequence
            cls_boxes - detected boxes (of person class)
            cls_segms - detected segments
            cls_keyps - detected key points
        """

        print("\n\n\n\nvis[{}] to {}".format(i, self.output_dir))

        vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            "frame {}".format(i),
            self.output_dir,
            cls_boxes,
            cls_segms,
            cls_keyps,
            dataset=self.dummy_coco_dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=0.7,
            kp_thresh=2,
        )


def draw_box(b, ax, color="r"):
    """
    draw a bounding box on the given matplot lib axis
    """
    assert len(b) >= 4, "box should have 2 2D points but encountered {}".format(b)
    ax.plot([b[0], b[0], b[2], b[2], b[0]], [b[1], b[3], b[3], b[1], b[1]], c=color)


def label_keypoints(kps, ax, type="coco"):
    """
    apply labels to a set of keypoints, e.g. 'left_knee',
    on a matplotlib axis.
    Initial support is assumes COCO keypoints
    """

    assert type == "coco", "keypoint type {} not supported".format(type)

    labels, _ = get_keypoints()

    #     print('label points={}'.format(zip(labels, kps[0], kps[1])))
    for label, x, y in zip(labels, kps[0], kps[1]):
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(-20, 20),
            textcoords="offset points",
            ha="right",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        )
