import time
import os
import collections
import torch
import gzip
from multiprocessing import Process
from functools import partial

import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # No need to use GPU


DatasetInfo = collections.namedtuple(
    "DatasetInfo",
    ["basepath", "train_size", "test_size", "frame_size", "sequence_size"],
)
Context = collections.namedtuple("Context", ["frames", "cameras"])
Scene = collections.namedtuple("Scene", ["frames", "cameras"])
Query = collections.namedtuple("Query", ["context", "query_camera"])
TaskData = collections.namedtuple("TaskData", ["query", "target"])


_DATASETS = dict(
    jaco=DatasetInfo(
        basepath="jaco", train_size=3600, test_size=400, frame_size=64, sequence_size=11
    ),
    mazes=DatasetInfo(
        basepath="mazes",
        train_size=1080,
        test_size=120,
        frame_size=84,
        sequence_size=300,
    ),
    rooms_free_camera_with_object_rotations=DatasetInfo(
        basepath="rooms_free_camera_with_object_rotations",
        train_size=2034,
        test_size=226,
        frame_size=128,
        sequence_size=10,
    ),
    rooms_ring_camera=DatasetInfo(
        basepath="rooms_ring_camera",
        train_size=2160,
        test_size=240,
        frame_size=64,
        sequence_size=10,
    ),
    rooms_free_camera_no_object_rotations=DatasetInfo(
        basepath="rooms_free_camera_no_object_rotations",
        train_size=2160,
        test_size=240,
        frame_size=64,
        sequence_size=10,
    ),
    shepard_metzler_5_parts=DatasetInfo(
        basepath="shepard_metzler_5_parts",
        train_size=900,
        test_size=100,
        frame_size=64,
        sequence_size=15,
    ),
    shepard_metzler_7_parts=DatasetInfo(
        basepath="shepard_metzler_7_parts",
        train_size=900,
        test_size=100,
        frame_size=64,
        sequence_size=15,
    ),
)


def _get_dataset_files(dataset_info, mode, root):
    """Generates lists of files for a given dataset version."""
    basepath = dataset_info.basepath
    base = os.path.join(root, basepath, mode)
    if mode == "train":
        num_files = dataset_info.train_size
    else:
        num_files = dataset_info.test_size

    files = sorted(os.listdir(base))

    return [os.path.join(base, file) for file in files]


def encapsulate(frames, cameras):
    return Scene(cameras=cameras, frames=frames)


def show_frame(frames, scene, views):
    import matplotlib

    matplotlib.use("qt5agg")
    import matplotlib.pyplot as plt

    plt.imshow(frames[scene, views])
    plt.show()


def _parse_batch(example, dataset_info):
    feature_map = {
        "frames": tf.io.FixedLenFeature([dataset_info.sequence_size], dtype=tf.string),
        "cameras": tf.io.FixedLenFeature(
            [dataset_info.sequence_size * 5],
            dtype=tf.float32,
        ),
    }
    features = tf.io.parse_example(example, features=feature_map)
    # Process
    frames = tf.concat(features["frames"], axis=0)
    raw_pose_params = features["cameras"]
    cameras = tf.reshape(raw_pose_params, [-1, dataset_info.sequence_size, 5])
    return frames, cameras


def write_data(path, frame, cam):
    scene = encapsulate(frame.numpy(), cam.numpy())
    with gzip.open(path, "wb") as f:
        torch.save(scene, f)


if __name__ == "__main__":

    import sys
    from os.path import join
    from multiprocessing import Pool
    from tqdm import tqdm

    # from tqdm.notebook import tqdm

    if len(sys.argv) < 3:
        print(" [!] you need to give a dataset and dataset root path")
        print(
            "[example] python convert2torch.py shepard_metzler_5_parts /datasets/gqn_datasets"
        )
        exit()

    DATASET = sys.argv[1]
    root_path = sys.argv[2]
    dataset_info = _DATASETS[DATASET]
    batch_size = 1024
    # Prepare thread pool
    pool = Pool(32)

    torch_dataset_path = join(root_path, DATASET + "-torch")
    torch_dataset_path_train = join(torch_dataset_path, "train")
    torch_dataset_path_test = join(torch_dataset_path, "test")

    os.mkdir(torch_dataset_path)
    os.mkdir(torch_dataset_path_train)
    os.mkdir(torch_dataset_path_test)

    parse_batch = partial(_parse_batch, dataset_info=dataset_info)

    ## train
    file_names = _get_dataset_files(dataset_info, "train", root_path)

    tot = 0
    dataset = tf.data.TFRecordDataset(file_names).batch(batch_size).map(parse_batch)
    for frames, cameras in dataset:
        pbar = tqdm(zip(frames, cameras), total=batch_size)
        pbar.set_description(f"Loading..Total:{tot} ")
        for i, (frame, cam) in enumerate(pbar):
            path = os.path.join(torch_dataset_path_train, f"{tot + i}.pt.gz")
            pool.apply_async(
                write_data,
                args=(
                    path,
                    frame,
                    cam,
                ),
            )
        tot += i
    print(f" [-] {tot} scenes in the train dataset")

    ## test
    file_names = _get_dataset_files(dataset_info, "test", root_path)

    tot = 0
    dataset = tf.data.TFRecordDataset(file_names).batch(batch_size).map(parse_batch)
    for frames, cameras in dataset:
        pbar = tqdm(zip(frames, cameras), total=batch_size)
        pbar.set_description(f"Loading..Total:{tot} ")
        for i, (frame, cam) in enumerate(pbar):
            path = os.path.join(torch_dataset_path_test, f"{tot + i}.pt.gz")
            pool.apply_async(
                write_data,
                args=(
                    path,
                    frame,
                    cam,
                ),
            )
        tot += i
    pool.close()
    pool.join()
    print(f" [-] {tot} scenes in the test dataset")
