from dota.datastats import DotaDatastats
from dota.classes import CLASS_NAMES

if __name__ == "__main__":
    image_dir = "/home/royalbrothers/open_source_yolo_project/dataset_tiles/images"
    label_dir = "/home/royalbrothers/open_source_yolo_project/dataset_tiles/labels"
    stats_calculator = DotaDatastats(image_dir=image_dir, label_dir=label_dir, class_names=CLASS_NAMES)
    stats_calculator.print_summary()