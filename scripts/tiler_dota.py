from dota.tiler import YoloTiler

if __name__ == "__main__":
    tiler = YoloTiler(
        images_dir="/home/royalbrothers/open_source_yolo_project/dataset_before_split/images",
        labels_dir="/home/royalbrothers/open_source_yolo_project/dataset_before_split/labels",
        output_images_dir="/home/royalbrothers/open_source_yolo_project/dataset_tiles/images",
        output_labels_dir="/home/royalbrothers/open_source_yolo_project/dataset_tiles/labels",
        tile_size=1024,
        overlap=200,
        min_box_size=10
    )
    tiler.tile_all()