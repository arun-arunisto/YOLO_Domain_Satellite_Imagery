from dota.converter import DotoYoloConverter

if __name__ == "__main__":
    converter = DotoYoloConverter(
        images_dir="/home/royalbrothers/open_source_yolo_project/dataset_before_split/images",
        dota_labels_dir="/home/royalbrothers/open_source_yolo_project/dataset_before_split/labels",
        output_labels_dir="/home/royalbrothers/open_source_yolo_project/dataset_before_split/labels",
        ignore_difficult=True
    )
    converter.convert()