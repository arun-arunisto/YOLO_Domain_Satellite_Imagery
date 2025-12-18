import os
from PIL import Image
from .classes import DOTA_CLASSES

class DotoYoloConverter:
    """
    Converts DOTA v1.0 annotations (OBB) into YOLO format (HBB).

    - Keeps float precision
    - Converts OBB -> HBB
    - Ignores difficult objects by default
    """

    def __init__(self, images_dir: str, dota_labels_dir: str, output_labels_dir: str, ignore_difficult: bool = True):
        self.images_dir = images_dir
        self.dota_labels_dir = dota_labels_dir
        self.output_labels_dir = output_labels_dir
        self.ignore_difficult = ignore_difficult

        os.makedirs(self.output_labels_dir, exist_ok=True)
    
    @staticmethod
    def _obb_to_hbb(coords):
        xs = coords[0::2]
        ys = coords[1::2]
        return min(xs), min(ys), max(xs), max(ys)
    
    @staticmethod
    def _hbb_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
        xc = ((xmin + xmax) / 2) / img_w
        yc = ((ymin + ymax) / 2) / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h
        return xc, yc, w, h
    
    def _find_image(self, base_name):
        for ext in (".png", ".jpg", ".jpeg"):
            path = os.path.join(self.images_dir, base_name + ext)
            if os.path.exists(path):
                return path
        return None
    
    def _convert_single_file(self, label_file):
        base_name = os.path.splitext(label_file)[0]

        img_path = self._find_image(base_name)
        if img_path is None:
            print(f"[WARN] Image not found for {base_name}")
            return

        img = Image.open(img_path)
        img_w, img_h = img.size

        yolo_lines = []

        with open(os.path.join(self.dota_labels_dir, label_file), "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 10:
                    continue

                coords = list(map(float, parts[:8]))
                class_name = parts[8]
                difficult = int(parts[9])

                if self.ignore_difficult and difficult == 1:
                    continue

                if class_name not in DOTA_CLASSES:
                    continue

                class_id = DOTA_CLASSES[class_name]

                xmin, ymin, xmax, ymax = self._obb_to_hbb(coords)
                xc, yc, w, h = self._hbb_to_yolo(
                    xmin, ymin, xmax, ymax, img_w, img_h
                )

                if w <= 0 or h <= 0:
                    continue

                yolo_lines.append(
                    f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
                )

        if yolo_lines:
            out_path = os.path.join(self.output_labels_dir, base_name + ".txt")
            with open(out_path, "w") as f:
                f.write("\n".join(yolo_lines))
    
    def convert(self):
        label_files = [
            f for f in os.listdir(self.dota_labels_dir) if f.endswith(".txt")
        ]

        print(f"[INFO] Converting {len(label_files)} annotation files")

        for label_file in label_files:
            self._convert_single_file(label_file)

        print("[DONE] DOTA → YOLO conversion complete")
