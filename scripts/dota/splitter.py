import os
import random
import shutil

class TrainValSplitter:
    """
    Splits a YOLO tiled dataset into train / val sets.
    """
    def __init__(self, images_dir: str, labels_dir: str, output_dir: str, val_ratio: float = 0.2, seed: int = 42):
        assert 0 < val_ratio < 1, "val_ratio must be between 0 and 1"
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.output_dir = output_dir
        self.val_ratio = val_ratio
        self.seed = seed

        self.train_img_dir = os.path.join(output_dir, "images/train")
        self.val_img_dir = os.path.join(output_dir, "images/val")
        self.train_lbl_dir = os.path.join(output_dir, "labels/train")
        self.val_lbl_dir = os.path.join(output_dir, "labels/val")

        for d in [self.train_img_dir, self.val_img_dir, self.train_lbl_dir, self.val_lbl_dir]:
            os.makedirs(d, exist_ok=True)
    
    def split(self):
        images = [
            f for f in os.listdir(self.images_dir)
            if f.lower().endswith((".jpg", ".png"))
        ]

        random.seed(self.seed)
        random.shuffle(images)

        split_idx = int(len(images) * (1 - self.val_ratio))
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        print(f"[INFO] Total tiles: {len(images)}")
        print(f"[INFO] Train: {len(train_images)} | Val: {len(val_images)}")

        self._copy_pairs(train_images, train=True)
        self._copy_pairs(val_images, train=False)

        print("[DONE] Train/Val split completed")
    
    def _copy_pairs(self, image_list, train: bool):
        for img_name in image_list:
            src_img = os.path.join(self.images_dir, img_name)
            src_lbl = os.path.join(
                self.labels_dir, os.path.splitext(img_name)[0] + ".txt"
            )

            if train:
                dst_img = os.path.join(self.train_img_dir, img_name)
                dst_lbl = os.path.join(
                    self.train_lbl_dir,
                    os.path.splitext(img_name)[0] + ".txt",
                )
            else:
                dst_img = os.path.join(self.val_img_dir, img_name)
                dst_lbl = os.path.join(
                    self.val_lbl_dir,
                    os.path.splitext(img_name)[0] + ".txt",
                )

            shutil.copy2(src_img, dst_img)

            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, dst_lbl)
