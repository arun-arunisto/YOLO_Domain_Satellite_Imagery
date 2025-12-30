from train import YoloEvaluator

evaluator = YoloEvaluator(
    model_path="/home/royalbrothers/open_source_yolo_project/scripts/runs/dota/baseline/weights/best.pt",
    data_yaml="/home/royalbrothers/open_source_yolo_project/data/dataset.yaml"
)

metrics = evaluator.evaluate(imgsz=640)
print(metrics)