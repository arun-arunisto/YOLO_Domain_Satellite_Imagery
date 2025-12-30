from dataclasses import dataclass


@dataclass
class TrainConfig:
    # core
    model: str = "yolov8s"
    data: str = "data/dataset.yaml"

    # Training
    imgsz: int = 640
    epochs: int = 50
    batch: int = 16
    seed: int = 42

    # output
    project: str = "runs/dota"
    name: str = "baseline"

    # hardware
    device: str = "auto" # auto/cpu/0/0,1
    workers: int = 0

    #optional
    pretrained: bool = True