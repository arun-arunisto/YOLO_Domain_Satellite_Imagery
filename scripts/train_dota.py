from train import TrainConfig, YoloTrainer

def main():
    cfg = TrainConfig(
        model="yolov8s.pt",
        data="/home/royalbrothers/open_source_yolo_project/data/dataset.yaml",
        imgsz=640,
        epochs=50,
        batch=16,
        seed=42,
        project="runs/dota",
        name="baseline"
    )

    trainer = YoloTrainer(cfg)
    trainer.train()

if __name__ == "__main__":
    main()
