from dota.visualizer import YoloVisualizer
from dota.classes import DOTA_CLASSES

if __name__ == "__main__":
    visualizer = YoloVisualizer(
        images_dir="/home/royalbrothers/open_source_yolo_project/dataset_tiles/images",
        labels_dir="/home/royalbrothers/open_source_yolo_project/dataset_tiles/labels",
        class_names=DOTA_CLASSES
    )
    visualizer.visualize_random(num_images=10)