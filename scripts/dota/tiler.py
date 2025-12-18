import os
import cv2
import math


class YoloTiler:
    """
    Tiles images + YOLO labels into fixed-size patches.

    Pipeline:
    - Image -> tiles
    - Adjust bounding boxes per tile
    - Drop tiny / invalid boxes
    """

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        output_images_dir: str,
        output_labels_dir: str,
        tile_size: int = 1024,
        overlap: int = 200,
        min_box_size: int = 10,
    ):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.output_images_dir = output_images_dir
        self.output_labels_dir = output_labels_dir

        self.tile_size = tile_size
        self.stride = tile_size - overlap
        self.min_box_size = min_box_size

        os.makedirs(self.output_images_dir, exist_ok=True)
        os.makedirs(self.output_labels_dir, exist_ok=True)

    def _yolo_to_pixel(self, xc, yc, w, h, img_w, img_h):
        box_w = w * img_w
        box_h = h * img_h
        x_center = xc * img_w
        y_center = yc * img_h

        xmin = x_center - box_w / 2
        ymin = y_center - box_h / 2
        xmax = x_center + box_w / 2
        ymax = y_center + box_h / 2

        return xmin, ymin, xmax, ymax

    def _pixel_to_yolo(self, xmin, ymin, xmax, ymax, tile_w, tile_h):
        xc = ((xmin + xmax) / 2) / tile_w
        yc = ((ymin + ymax) / 2) / tile_h
        w = (xmax - xmin) / tile_w
        h = (ymax - ymin) / tile_h
        return xc, yc, w, h

    def _clip_box(self, xmin, ymin, xmax, ymax, x0, y0, x1, y1):
        xmin = max(xmin, x0)
        ymin = max(ymin, y0)
        xmax = min(xmax, x1)
        ymax = min(ymax, y1)
        return xmin, ymin, xmax, ymax

    def tile_single(self, image_name: str):
        img_path = os.path.join(self.images_dir, image_name)
        label_path = os.path.join(
            self.labels_dir, os.path.splitext(image_name)[0] + ".txt"
        )

        image = cv2.imread(img_path)
        if image is None:
            return

        img_h, img_w = image.shape[:2]

        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cid, xc, yc, w, h = map(float, parts)
                    boxes.append((int(cid), xc, yc, w, h))

        x_steps = math.ceil((img_w - self.tile_size) / self.stride) + 1
        y_steps = math.ceil((img_h - self.tile_size) / self.stride) + 1

        for yi in range(y_steps):
            for xi in range(x_steps):
                x0 = xi * self.stride
                y0 = yi * self.stride
                x1 = min(x0 + self.tile_size, img_w)
                y1 = min(y0 + self.tile_size, img_h)

                tile = image[y0:y1, x0:x1]
                if tile.size == 0:
                    continue

                tile_boxes = []

                for cid, xc, yc, w, h in boxes:
                    xmin, ymin, xmax, ymax = self._yolo_to_pixel(
                        xc, yc, w, h, img_w, img_h
                    )

                    xmin, ymin, xmax, ymax = self._clip_box(
                        xmin, ymin, xmax, ymax, x0, y0, x1, y1
                    )

                    bw = xmax - xmin
                    bh = ymax - ymin

                    if bw < self.min_box_size or bh < self.min_box_size:
                        continue

                    xmin -= x0
                    ymin -= y0
                    xmax -= x0
                    ymax -= y0

                    xc_n, yc_n, w_n, h_n = self._pixel_to_yolo(
                        xmin, ymin, xmax, ymax, x1 - x0, y1 - y0
                    )

                    if w_n <= 0 or h_n <= 0:
                        continue

                    tile_boxes.append(
                        f"{cid} {xc_n:.6f} {yc_n:.6f} {w_n:.6f} {h_n:.6f}"
                    )

                if not tile_boxes:
                    continue

                tile_name = (
                    f"{os.path.splitext(image_name)[0]}"
                    f"_x{xi}_y{yi}.jpg"
                )

                cv2.imwrite(
                    os.path.join(self.output_images_dir, tile_name), tile
                )

                with open(
                    os.path.join(
                        self.output_labels_dir,
                        tile_name.replace(".jpg", ".txt"),
                    ),
                    "w",
                ) as f:
                    f.write("\n".join(tile_boxes))

    def tile_all(self):
        images = [
            f for f in os.listdir(self.images_dir)
            if f.lower().endswith((".jpg", ".png"))
        ]

        print(f"[INFO] Tiling {len(images)} images")

        for img in images:
            self.tile_single(img)

        print("[DONE] Tiling complete")
