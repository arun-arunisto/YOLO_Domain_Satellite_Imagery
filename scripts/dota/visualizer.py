import os
import random
import cv2

class YoloVisualizer:
    """
    Visual sanity checker for YOLO datasets.

    - Loads images + YOLO labels
    - Draws bounding boxes
    - Lets you visually confirm correctness
    """
    def __init__(self, images_dir: str, labels_dir: str, class_names: dict, window_name: str = "YOLO Sanity Check"):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.class_names = class_names
        self.window_name = window_name
    
    def _load_yolo_labels(self, label_path, img_w, img_h):
        boxes = []

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                class_id, xc, yc, w, h = map(float, parts)
                class_id = int(class_id)

                # Convert YOLO → pixel coordinates
                box_w = w * img_w
                box_h = h * img_h
                x_center = xc * img_w
                y_center = yc * img_h

                xmin = int(x_center - box_w / 2)
                ymin = int(y_center - box_h / 2)
                xmax = int(x_center + box_w / 2)
                ymax = int(y_center + box_h / 2)

                boxes.append((class_id, xmin, ymin, xmax, ymax))

        return boxes
    
    def _draw_boxes(self, image, boxes):
        for class_id, xmin, ymin, xmax, ymax in boxes:
            label = self.class_names.get(class_id, str(class_id))
            color = (0, 255, 0)

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(
                image,
                label,
                (xmin, max(ymin - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        return image
    
    def visualize_random(self, num_images: int = 5):
        images = [
            f for f in os.listdir(self.images_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        if not images:
            raise RuntimeError("No images found")

        sample = random.sample(images, min(num_images, len(images)))

        for img_name in sample:
            img_path = os.path.join(self.images_dir, img_name)
            label_path = os.path.join(
                self.labels_dir, os.path.splitext(img_name)[0] + ".txt"
            )

            if not os.path.exists(label_path):
                print(f"[WARN] No label for {img_name}")
                continue

            image = cv2.imread(img_path)
            if image is None:
                continue

            img_h, img_w = image.shape[:2]
            boxes = self._load_yolo_labels(label_path, img_w, img_h)
            image = self._draw_boxes(image, boxes)

            cv2.imshow(self.window_name, image)
            key = cv2.waitKey(0)

            # Press ESC to exit early
            if key == 27:
                break

        cv2.destroyAllWindows()