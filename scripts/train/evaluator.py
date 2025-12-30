from ultralytics import YOLO

class YoloEvaluator:
    def __init__(self, model_path: str, data_yaml: str):
        self.model = YOLO(model_path)
        self.data = data_yaml
    
    def evaluate(self, imgsz: int = 640):
        print("[INFO] Running evaluation")
        metrics = self.model.val(
            data=self.data,
            imgsz=imgsz,
        )

        return {
            "mAP50_95":metrics.box.map,
            "mAP50": metrics.box.map50,
            "recall":metrics.box.mr,
        }