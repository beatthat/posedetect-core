import logging
import numpy as np
import os

import ffmpy

from posedetect_core.d3 import create_model

logger = logging.getLogger(__name__)


def estimate_poses_task(task, input_video_path, result_url_format):
    output_dir = os.path.join(os.path.dirname(input_video_path), "out")
    os.makedirs(output_dir, exist_ok=True)
    stem, ext = os.path.splitext(input_video_path)
    video_path = (
        input_video_path
        if ext == ".mp4"
        else os.path.join(
            output_dir,
            f"{os.path.basename(stem)}.mp4"
            # if we have to convert the input video to mp4,
            # put the converted input in the out directory
        )
    )
    if video_path != input_video_path:
        ff = ffmpy.FFmpeg(
            inputs={input_video_path: None},
            outputs={video_path: ("-y", "-an", "-vcodec", "libx264")},
        )
        ff.run()
    model = create_model("posedetect_vp3d.d3")
    pose3d_result = model.predict(video_path, task.update_state)
    result_filename = "pose_estimations_3d.npz"
    output_path = os.path.join(output_dir, result_filename)
    result_dict = {k: v for k, v in pose3d_result.items() if k != "dataset_3d"}
    np.savez(output_path, **result_dict)
    result_url = result_url_format.format(task.request.id)
    result_visualization_path = f"{output_dir}/pose_estimates_3d.mp4"
    model.visualize(
        pose3d_result["dataset_3d"],
        pose3d_result["input_keypoints_2d"],
        pose3d_result["output_keypoints_3d"],
        pose3d_result["input_video_path"],
        result_visualization_path,
        **pose3d_result["kwargs"],
    )
    return {
        # "current": info["current"] + 1,
        # "total": info["current"] + 1,
        "status": "success",
        "resultFile": output_path,
        "resultUrl": result_url,
        "resultPoses2dUrl": f"{result_url}/poses2d",
        "resultPoses3dUrl": f"{result_url}/poses3d",
        "resultVisualizationFile": result_visualization_path,
        "resultVisualizationUrl": f"{result_url}/visualization",
    }
