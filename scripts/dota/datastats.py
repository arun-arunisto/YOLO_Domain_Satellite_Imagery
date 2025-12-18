import os
from collections import defaultdict
import numpy as np

class DotaDatastats:
    """
    Computes dataset statistics for YOLO-formatted labels:
    - per class counts
    - per class image counts
    - bbox sizes and aspect ratios
    - image-level object density
    - empty image identification
    """
    def __init__(self, image_dir: str, label_dir: str, class_names: dict):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.class_names = class_names

        # Stats containers
        self.class_counts = defaultdict(int)
        self.image_counts = defaultdict(int)
        self.bbox_widths = defaultdict(list)
        self.bbox_heights = defaultdict(list)
        self.aspect_ratios = defaultdict(list)
        self.objects_per_image = defaultdict(int)
        self.empty_images = []

    def _load_label(self, path):
        try:
            with open(path, "r") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            return lines
        except:
            return []
    
    def compute(self):
        images = [f for f in os.listdir(self.image_dir) if f.lower().endswith(".jpg")]
        total_images = len(images)

        for img in images:
            stem = img.replace(".jpg", "")
            label_path = os.path.join(self.label_dir, stem + ".txt")

            if not os.path.exists(label_path):
                self.empty_images.append(img)
                continue

            lines = self._load_label(label_path)
            if not lines:
                self.empty_images.append(img)
                continue

            for line in lines:
                cls, xc, yc, w, h = line.split()
                cls = int(cls)
                w, h = float(w), float(h)

                self.class_counts[cls] += 1
                self.image_counts[cls] += 1
                self.objects_per_image[img] += 1

                # Store bbox stats
                self.bbox_widths[cls].append(w)
                self.bbox_heights[cls].append(h)
                self.aspect_ratios[cls].append(w / h if h > 0 else 0)

        return {
            "total_images": total_images,
            "total_objects": sum(self.class_counts.values()),
            "class_counts": dict(self.class_counts),
            "image_counts": dict(self.image_counts),
            "bbox_width_avg": {c: float(np.mean(vals)) for c, vals in self.bbox_widths.items()},
            "bbox_height_avg": {c: float(np.mean(vals)) for c, vals in self.bbox_heights.items()},
            "aspect_ratio_avg": {c: float(np.mean(vals)) for c, vals in self.aspect_ratios.items()},
            "objects_per_image": dict(self.objects_per_image),
            "empty_images": self.empty_images,
        }
    
    def print_summary(self):
        stats = self.compute()

        print("\n===== DATASET STATISTICS =====\n")
        print(f"Total images: {stats['total_images']}")
        print(f"Total labeled objects: {stats['total_objects']}")
        print(f"Empty images: {len(stats['empty_images'])}")

        print("\n--- Per-Class Object Counts ---")
        for cls, count in stats["class_counts"].items():
            print(f"{cls} ({self.class_names.get(cls, 'Unknown')}): {count}")

        print("\n--- Avg BBox Width/Height (Normalized) ---")
        for cls in stats["bbox_width_avg"]:
            print(f"{cls}: W={stats['bbox_width_avg'][cls]:.4f}, H={stats['bbox_height_avg'][cls]:.4f}")

        print("\n--- Avg Aspect Ratio ---")
        for cls, ar in stats["aspect_ratio_avg"].items():
            print(f"{cls}: AR={ar:.4f}")

        print("\n--- Object Density (Top 10 images) ---")
        densest = sorted(stats["objects_per_image"].items(), key=lambda x: x[1], reverse=True)[:10]
        for img, count in densest:
            print(f"{img}: {count} objects")

        print("\n===== END OF DATASET SUMMARY =====\n")

        return stats