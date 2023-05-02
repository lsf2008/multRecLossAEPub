# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# from __future__ import annotations
import itertools
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch.utils.data
from pytorchvideo.data.clip_sampling import ClipSampler
from pytorchvideo.data.video import VideoPathHandler
from iopath.common.file_io import g_pathmgr
# from .labeled_video_paths import LabeledVideoPaths
# from .utils import MultiProcessSampler
import numpy as np
import math

from collections import Counter


def getLabels_(gtl, clip_start_idx, clip_end_idx):
    '''get clip label which from clip_start_idx to clip_end_idx

    Parameters
    ----------
    gtl : list
        video groundtruth of each frame
    clip_start_idx : int
        start frame index of clip
    clip_end_idx : int
        end frame index of clip
    '''
    gt =gtl['label']
    gtLen = len(gt)
    assert (clip_start_idx <= clip_end_idx), "wrong clip start and end index"
    assert (clip_end_idx <= gtLen), 'end clip index must smaller than the ground truth length'
    clip_label = gt[clip_start_idx:clip_end_idx]
    # support 1 for abnormal
    # abn_num = Counter(clip_label)
    # # nor_num=torch.sum(clip_label==0)
    # lb=torch.tensor(0)
    # if abn_thr is None:
    #     abn_thr=torch.floor((clip_end_idx-clip_start_idx)/2 )
    # if abn_num[1] >= abn_thr:
    #     lb= torch.tensor(1)
    return clip_label

def getLabels_secs(gt, clip_start, clip_end):
    if len(gt['label'])>1:
        clip_start_idx = secs_to_frm(clip_start)
        clip_end_idx = secs_to_frm(clip_end)
        return getLabels_(gt, clip_start_idx, clip_end_idx)
    else:
        return 0


def secs_to_frm(
    time_in_seconds: float,
    time_base: float=30,
    round_mode: str = "round",
) -> int:
    """
    Converts a time (in seconds) to the given time base and start_pts offset
    presentation time. Round_mode specifies the mode of rounding when converting time.

    Returns:
        pts (int): The time in the given time base.
    """
    if time_in_seconds == math.inf:
        return math.inf

    assert round_mode in ["round", "ceil"], f"round_mode={round_mode} is not supported!"

    if round_mode == "round":
        return round(time_in_seconds*time_base)
    else:
        return math.ceil(time_in_seconds * time_base)

logger = logging.getLogger(__name__)
class LabeledVideoPaths:
    def __init__(
        self, paths_and_labels: List[Tuple[str, Optional[int]]], path_prefix=""
    ) -> None:
        """
        Args:
            paths_and_labels [(str, int)]: a list of tuples containing the video
                path and integer label.
        """
        self._paths_and_labels = paths_and_labels
        self._path_prefix = path_prefix
    @classmethod
    def from_csv(cls, file_path: str):
        """
            Factory function that creates a LabeledVideoPaths object by reading a file with the
            following format:
                <path> <integer_label>
                ...
                <path> <integer_label>

            Args:
                file_path (str): The path to the file to be read.
            """
        assert g_pathmgr.exists(file_path), f"{file_path} not found."
        video_paths_and_label = []
        with g_pathmgr.open(file_path, "r") as f:
            for path_label in f.read().splitlines():
                line_split = path_label.rsplit(None, -1)

                # The video path file may not contain labels (e.g. for a test split). We
                # assume this is the case if only 1 path is found and set the label to
                # -1 if so.
                if len(line_split) == 1:
                    file_path = line_split[0]
                    label = -1
                else:
                    file_path, label = line_split[0], line_split[1:]

                    if len(label) > 1:
                        nl = [int(x) for x in label]
                    else:
                        nl=[0]
                video_paths_and_label.append((file_path, nl))

        assert (
                len(video_paths_and_label) > 0
        ), f"Failed to load dataset from {file_path}."
        return cls(video_paths_and_label)
    def path_prefix(self, prefix):
        self._path_prefix = prefix

    path_prefix = property(None, path_prefix)

    def __getitem__(self, index: int) -> Tuple[str, int]:
        """
        Args:
            index (int): the path and label index.

        Returns:
            The path and label tuple for the given index.
        """
        path, label = self._paths_and_labels[index]
        return (os.path.join(self._path_prefix, path), {"label": label})

    def __len__(self) -> int:
        """
        Returns:
            The number of video paths and label pairs.
        """
        return len(self._paths_and_labels)


class MultiProcessSampler(torch.utils.data.Sampler):
    """
    MultiProcessSampler splits sample indices from a PyTorch Sampler evenly across
    workers spawned by a PyTorch DataLoader.
    """

    def __init__(self, sampler: torch.utils.data.Sampler) -> None:
        self._sampler = sampler

    def __iter__(self):
        """
        Returns:
            Iterator for underlying PyTorch Sampler indices split by worker id.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers != 0:

            # Split sampler indexes by worker.
            video_indexes = range(len(self._sampler))
            worker_splits = np.array_split(video_indexes, worker_info.num_workers)
            worker_id = worker_info.id
            worker_split = worker_splits[worker_id]
            if len(worker_split) == 0:
                logger.warning(
                    f"More data workers({worker_info.num_workers}) than videos"
                    f"({len(self._sampler)}). For optimal use of processes "
                    "reduce num_workers."
                )
                return iter(())

            iter_start = worker_split[0]
            iter_end = worker_split[-1] + 1
            worker_sampler = itertools.islice(iter(self._sampler), iter_start, iter_end)
        else:

            # If no worker processes found, we return the full sampler.
            worker_sampler = iter(self._sampler)

        return worker_sampler

class LabeledVideoDataset(torch.utils.data.IterableDataset):
    """
    LabeledVideoDataset handles the storage, loading, decoding and clip sampling for a
    video dataset. It assumes each video is stored as either an encoded video
    (e.g. mp4, avi) or a frame video (e.g. a folder of jpg, or png)
    """

    _MAX_CONSECUTIVE_FAILURES = 10

    def __init__(
        self,
        labeled_video_paths: List[Tuple[str, Optional[dict]]],
        clip_sampler: ClipSampler,
        video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
        transform: Optional[Callable[[dict], Any]] = None,
        decode_audio: bool = True,
        decoder: str = "pyav",
    ) -> None:
        """
        Args:
            labeled_video_paths (List[Tuple[str, Optional[dict]]]): List containing
                    video file paths and associated labels. If video paths are a folder
                    it's interpreted as a frame video, otherwise it must be an encoded
                    video.

            clip_sampler (ClipSampler): Defines how clips should be sampled from each
                video. See the clip sampling documentation for more information.

            video_sampler (Type[torch.utils.data.Sampler]): Sampler for the internal
                video container. This defines the order videos are decoded and,
                if necessary, the distributed split.

            transform (Callable): This callable is evaluated on the clip output before
                the clip is returned. It can be used for user defined preprocessing and
                augmentations on the clips. The clip output format is described in __next__().

            decode_audio (bool): If True, also decode audio from video.

            decoder (str): Defines what type of decoder used to decode a video. Not used for
                frame videos.
        """
        self._decode_audio = decode_audio
        self._transform = transform
        self._clip_sampler = clip_sampler
        self._labeled_videos = labeled_video_paths
        self._decoder = decoder

        # If a RandomSampler is used we need to pass in a custom random generator that
        # ensures all PyTorch multiprocess workers have the same random seed.
        self._video_random_generator = None
        if video_sampler == torch.utils.data.RandomSampler:
            self._video_random_generator = torch.Generator()
            self._video_sampler = video_sampler(
                self._labeled_videos, generator=self._video_random_generator
            )
        else:
            self._video_sampler = video_sampler(self._labeled_videos)

        self._video_sampler_iter = None  # Initialized on first call to self.__next__()

        # Depending on the clip sampler type, we may want to sample multiple clips
        # from one video. In that case, we keep the store video, label and previous sampled
        # clip time in these variables.
        self._loaded_video_label = None
        self._loaded_clip = None
        self._next_clip_start_time = 0.0
        self.video_path_handler = VideoPathHandler()

    @property
    def video_sampler(self):
        """
        Returns:
            The video sampler that defines video sample order. Note that you'll need to
            use this property to set the epoch for a torch.utils.data.DistributedSampler.
        """
        return self._video_sampler

    @property
    def num_videos(self):
        """
        Returns:
            Number of videos in dataset.
        """
        return len(self.video_sampler)

    def __next__(self) -> dict:
        """
        Retrieves the next clip based on the clip sampling strategy and video sampler.

        Returns:
            A dictionary with the following format.

            .. code-block:: text

                {
                    'video': <video_tensor>,
                    'label': <index_label>,
                    'video_label': <index_label>
                    'video_index': <video_index>,
                    'clip_index': <clip_index>,
                    'aug_index': <aug_index>,
                }
        """
        if not self._video_sampler_iter:
            # Setup MultiProcessSampler here - after PyTorch DataLoader workers are spawned.
            self._video_sampler_iter = iter(MultiProcessSampler(self._video_sampler))

        for i_try in range(self._MAX_CONSECUTIVE_FAILURES):
            # Reuse previously stored video if there are still clips to be sampled from
            # the last loaded video.
            if self._loaded_video_label:
                video, info_dict, video_index = self._loaded_video_label
            else:
                video_index = next(self._video_sampler_iter) #通过采样取到的视频索引
                try: # info_dict 存放的怎么是label信息
                    video_path, info_dict = self._labeled_videos[video_index]

                    # video 是该文件下的视频所有帧，duration，fps等信息
                    video = self.video_path_handler.video_from_path(
                        video_path,
                        decode_audio=self._decode_audio,
                        decoder=self._decoder,
                    )
                    self._loaded_video_label = (video, info_dict, video_index)
                except Exception as e:
                    logger.debug(
                        "Failed to load video with error: {}; trial {}".format(
                            e,
                            i_try,
                        )
                    )
                    continue

            (
                clip_start,# 视频提取片段的帧开头，是秒
                clip_end,#视频提取片段的帧结尾，秒
                clip_index,#视频提取片段的索引
                aug_index,
                is_last_clip,# 是否最后片段的标签
            ) = self._clip_sampler(
                self._next_clip_start_time, video.duration, info_dict
            )
            # 如果是list，说明是多个线程进行操作
            if isinstance(clip_start, list):  # multi-clip in each sample

                # Only load the clips once and reuse previously stored clips if there are multiple
                # views for augmentations to perform on the same clips.
                if aug_index[0] == 0:
                    self._loaded_clip = {}
                    loaded_clip_list = []
                    for i in range(len(clip_start)):
                        clip_dict = video.get_clip(clip_start[i], clip_end[i])
                        if clip_dict is None or clip_dict["video"] is None:
                            self._loaded_clip = None
                            break
                        loaded_clip_list.append(clip_dict)
                    if self._loaded_clip is not None:
                        for key in loaded_clip_list[0].keys():
                            self._loaded_clip[key] = [x[key] for x in loaded_clip_list]
                        if len(info_dict)>1:
                            self.lb=getLabels_secs(info_dict, clip_start, clip_end)
                            # self.lb = getLabels_secs(info_dict, clip_start, clip_start+1)

            else:  # single clip case

                # Only load the clip once and reuse previously stored clip if there are multiple
                # views for augmentations to perform on the same clip.
                if aug_index == 0:
                    self._loaded_clip = video.get_clip(clip_start, clip_end)
                    self.lb = getLabels_secs(info_dict, clip_start, clip_end) # return temporal lenght gt
                    # return first gt of the clip
                    # self.lb = getLabels_secs(info_dict, clip_start, clip_start+1)
            self._next_clip_start_time = clip_end
            # print(f'video:{video.name},{secs_to_frm(clip_start)}:{secs_to_frm(clip_end)}')
            video_is_null = (
                self._loaded_clip is None or self._loaded_clip["video"] is None
            )
            if (
                is_last_clip[-1] if isinstance(is_last_clip, list) else is_last_clip
            ) or video_is_null:
                # Close the loaded encoded video and reset the last sampled clip time ready
                # to sample a new video on the next iteration.
                self._loaded_video_label[0].close()
                self._loaded_video_label = None
                self._next_clip_start_time = 0.0
                self._clip_sampler.reset()
                if video_is_null:
                    logger.debug(
                        "Failed to load clip {}; trial {}".format(video.name, i_try)
                    )
                    continue

            frames = self._loaded_clip["video"]
            audio_samples = self._loaded_clip["audio"]
            lb = self.lb[len(self.lb)//2+1]
            sample_dict = {
                "video": frames,
                "video_name": video.name,
                "video_index": video_index,
                "clip_index": clip_index,
                "aug_index": aug_index,
                'label': lb,
                # modify by shifeng li
                # **info_dict, # 怎么是标签？？
                # **({"audio": audio_samples} if audio_samples is not None else {}),
            }

            if self._transform is not None:
                sample_dict = self._transform(sample_dict)

                # User can force dataset to continue by returning None in transform.
                if sample_dict is None:
                    continue

            return sample_dict
        else:
            raise RuntimeError(
                f"Failed to load video after {self._MAX_CONSECUTIVE_FAILURES} retries."
            )

    def __iter__(self):
        self._video_sampler_iter = None  # Reset video sampler

        # If we're in a PyTorch DataLoader multiprocessing context, we need to use the
        # same seed for each worker's RandomSampler generator. The workers at each
        # __iter__ call are created from the unique value: worker_info.seed - worker_info.id,
        # which we can use for this seed.
        worker_info = torch.utils.data.get_worker_info()
        if self._video_random_generator is not None and worker_info is not None:
            base_seed = worker_info.seed - worker_info.id
            self._video_random_generator.manual_seed(base_seed)

        return self



def labeled_video_dataset(
    data_path: str,
    clip_sampler: ClipSampler,
    video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    video_path_prefix: str = "",
    decode_audio: bool = True,
    decoder: str = "pyav",
):
    """
    A helper function to create ``LabeledVideoDataset`` object for Ucf101 and Kinetics datasets.

    Args:
        data_path (str): Path to the data. The path type defines how the data
            should be read:

            * For a file path, the file is read and each line is parsed into a
              video path and label.
            * For a directory, the directory structure defines the classes
              (i.e. each subdirectory is a class).

        clip_sampler (ClipSampler): Defines how clips should be sampled from each
                video. See the clip sampling documentation for more information.

        video_sampler (Type[torch.utils.data.Sampler]): Sampler for the internal
                video container. This defines the order videos are decoded and,
                if necessary, the distributed split.

        transform (Callable): This callable is evaluated on the clip output before
                the clip is returned. It can be used for user defined preprocessing and
                augmentations to the clips. See the ``LabeledVideoDataset`` class for clip
                output format.

        video_path_prefix (str): Path to root directory with the videos that are
                loaded in ``LabeledVideoDataset``. All the video paths before loading
                are prefixed with this path.

        decode_audio (bool): If True, also decode audio from video.

        decoder (str): Defines what type of decoder used to decode a video.

    """
    labeled_video_paths = LabeledVideoPaths.from_csv(data_path)
    labeled_video_paths.path_prefix = video_path_prefix
    dataset = LabeledVideoDataset(
        labeled_video_paths,
        clip_sampler,
        video_sampler,
        transform,
        decode_audio=decode_audio,
        decoder=decoder,
    )
    return dataset

if __name__=='__main__':
    crop_size = 256
    num_frames = 8
    # 采样率，每秒采样多少个图像，<==> 可以计算每个图像需要多少时间，则duration为num_frames/sampling_rate
    sampling_rate = 8
    frames_per_second = 30
    device = 1

    import pytorchvideo.data as vd

    dataPath = 'E:/dataset/UCSD/UCSDped1/video/label/train.csv'
    clip_sampler = vd.make_clip_sampler('uniform', num_frames / frames_per_second, 1/frames_per_second)
    dt = labeled_video_dataset(dataPath, clip_sampler, decoder=None, decode_audio=False)
    dt = iter(dt).__next__()
    print(dt['label'])