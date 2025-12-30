from ultralytics import YOLO
from .config import TrainConfig
from .utils import set_seed
import torch

class YoloTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        set_seed(cfg.seed)
        if cfg.device == "auto":
            cfg.device = "0" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(cfg.model)

    def train(self):
        print("[INFO] Starting YOLO training")
        self.model.train(
            data=self.cfg.data,
            epochs=self.cfg.epochs,
            batch=self.cfg.batch,
            seed=self.cfg.seed,
            project=self.cfg.project,
            name=self.cfg.name,
            device=self.cfg.device,
            workers=self.cfg.workers,
            pretrained=self.cfg.pretrained
        )