from typing import Optional
import os

from pydantic import BaseSettings

class Settings(BaseSettings):
    # dataloading parameters
    class_names: list = ['d100', 'd20', 'd12', 'd10', 'd8', 'd6', 'd4']
    img_size: int = 416
    num_workers: int = 4
    loading_batch_size: int = 1

    # NMS parameters
    conf_thres: float = 0.5
    nms_thres: float = 0.5

    # model parameters
    model_path: str = os.getenv('MODEL_PATH')

settings = Settings()