import logging
from celery import Celery
from .d2.boxes_and_keypoints import detect_boxes_and_keypoints_task
from .estimate_poses import estimate_poses_task

config = dict()

# Celery configuration
config["CELERY_BROKER_URL"] = "redis://redis:6379/0"
config["CELERY_RESULT_BACKEND"] = "redis://redis:6379/0"

config["CELERY_ACCEPT_CONTENT"] = ["json"]  # type: ignore
config["CELERY_TASK_SERIALIZER"] = "json"
config["CELERY_EVENT_SERIALIZER"] = "json"
config["CELERY_RESULT_SERIALIZER"] = "json"

# Initialize Celery
celery = Celery("pose-estimation", broker=config["CELERY_BROKER_URL"])
celery.conf.update(config)

logging.basicConfig()
logger = logging.getLogger(__name__)


@celery.task(bind=True)
def detect_boxes_and_keypoints(task, video_path, result_url_format):
    return detect_boxes_and_keypoints_task(task, video_path, result_url_format)


@celery.task(bind=True)
def estimate_poses(task, video_path, result_url_format):
    return estimate_poses_task(task, video_path, result_url_format)
