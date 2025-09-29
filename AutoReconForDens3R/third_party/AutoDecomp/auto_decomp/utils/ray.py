import os


def is_ray_environment():
    # FIXME: import ray set these params?
    return "RAY_JOB_ID" in os.environ or "RAY_RAYLET_PID" in os.environ
