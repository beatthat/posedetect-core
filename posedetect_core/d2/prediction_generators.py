from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import logging
import numpy as np
import time

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.utils.c2 as c2_utils


def generate_predicitions_from_frames(images, config_file, weights):
    """Generator yields inferred boxes and keypoints for each image in a provided iterable of images
    Args:
        images: iterable images
        config_file: Detectron configuration file
        weights: pretrained weights
    Returns:
        yields i, im, cls_boxes, cls_segms, cls_keyps
    """
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(config_file)
    cfg.NUM_GPUS = 1
    weights = cache_url(weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine.initialize_model_from_cfg(weights)
    for i, im in enumerate(images):
        logger.info("Processing frame {}".format(i))
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model, im, None, timers=timers
            )
        logger.info("Inference time: {:.3f}s".format(time.time() - t))
        for k, v in timers.items():
            logger.info(" | {}: {:.3f}s".format(k, v.average_time))
        if i == 0:
            logger.info(
                "Note: inference on the first image will be slower than the "
                "rest (caches and auto-tuning need to warm up)"
            )
        yield i, im, cls_boxes, cls_segms, cls_keyps


def generate_predictions_from_npz(images, npz_file):
    """Generator yields pre-calculated boxes and keypoints
    for each image in a provided iterable of images.
    This is mainly useful only for debugging,
    e.g. inspect the keypoints in a generated npz archive of keypoints
    against the images they were generated from.
    Args:
        images: iterable images
        config_file: Detectron configuration file
        weights: pretrained weights
    Returns:
        yields i, im, cls_boxes, cls_segms, cls_keyps
    """
    if not isinstance(npz_file, np.lib.npyio.NpzFile):
        npz_file = np.load(npz_file)
    keypoints = npz_file["keypoints"]
    boxes = npz_file["boxes"]
    for i, im in enumerate(images):
        cls_boxes = list(boxes[i])
        cls_keyps = list(keypoints[i])

        yield i, im, cls_boxes, None, cls_keyps
