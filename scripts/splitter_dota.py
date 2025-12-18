from dota.splitter import TrainValSplitter

if __name__ == "__main__":
    train_val_splitter = TrainValSplitter(
        images_dir="/home/royalbrothers/open_source_yolo_project/dataset_tiles/images",
        labels_dir="/home/royalbrothers/open_source_yolo_project/dataset_tiles/labels",
        output_dir="/home/royalbrothers/open_source_yolo_project/dataset"
    )
    train_val_splitter.split()