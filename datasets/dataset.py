import os

import string
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.io import read_image
import pandas as pd
from collections import defaultdict

from PIL import Image

import torchvision.transforms as T

import glob
import config


class FerDatasets(Dataset):

    def __init__(self, imgs, labels, flag = 0):
        super(FerDatasets, self).__init__()

        self.img = imgs
        self.label = labels
        self.flag = flag

    def __getitem__(self, i):
        # img = read_image(self.img[i])

        # img = transforms.Resize(100)(img)

        img = cv2.imread(self.img[i], cv2.IMREAD_COLOR)
        img1 = cv2.resize(img, (100, 100))
        tensor = torch.from_numpy(img1.transpose(2, 0, 1))

        label = self.label[1]

        return tensor.float(), label

    def __len__(self):
        return len(self.img)


class PainDatasets(Dataset):

    def __init__(self, img_dir, label_path, transform=None, target_transform=None):
        super(PainDatasets, self).__init__()

        self.img_labels = pd.read_csv(label_path, sep=" ") # for BAH sep="," else sep=" "
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.convtensor = transforms.ToTensor()

    def __getitem__(self, i):
        
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[i, 0])

        '''
        read_image is not working:
        UserWarning: Failed to load image Python extension: libtorch_cuda_cu.so: cannot open shared object file: No such file or directory 
        warn(f"Failed to load image Python extension: {e}")
        '''

        image = (Image.open(img_path))
        label = self.img_labels.iloc[i, 1]


        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


    def __len__(self):
        return len(self.img_labels)


class PainVideoDatasets(Dataset):
    """
    Video-level dataset where each sample corresponds to one video (a sequence of frames)
    belonging to a subject, with its class label.
    """

    def __init__(self, img_dir, label_path, transform=None, seq_len=None, target_transform=None):
        super(PainVideoDatasets, self).__init__()

        # Read label file: expected format "<subject>/<video> <label>"
        self.img_labels = pd.read_csv(label_path, sep=" ", header=None)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.seq_len = seq_len  # number of frames to sample per video

        # Read label file (frame paths + labels)
        df = pd.read_csv(label_path, sep=" ", header=None)
        df.columns = ["frame_path", "label"]

        # Group frames by video (parent directory of frame)
        df["video_dir"] = df["frame_path"].apply(lambda p: os.path.dirname(p))
        grouped = df.groupby("video_dir")

        self.samples = []
        for video_dir, group in grouped:
            frames = group["frame_path"].tolist()
            label = int(group["label"].iloc[0])
            self.samples.append((video_dir, frames, label))

    def __getitem__(self, idx):
        video_dir, frame_paths, label = self.samples[idx]

        # Resolve absolute frame paths
        frame_paths = [os.path.join(self.img_dir, fp) for fp in frame_paths]

        # Sort frames by name to preserve temporal order
        frame_paths = sorted(frame_paths)

        # Sample or pad frames
        if self.seq_len is not None:
            if len(frame_paths) > self.seq_len:
                idxs = torch.linspace(0, len(frame_paths) - 1, self.seq_len).long()
                frame_paths = [frame_paths[i] for i in idxs]
            elif len(frame_paths) < self.seq_len:
                frame_paths += [frame_paths[-1]] * (self.seq_len - len(frame_paths))

        # Load frames
        frames = []
        for frame_path in frame_paths:
            img = Image.open(frame_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        frames = torch.stack(frames, dim=0)  # [T, C, H, W]

        if self.target_transform:
            label = self.target_transform(label)

        # Extract subject ID from path (first directory)
        subject_id = video_dir.split("/")[0]

        return frames, label

    def __len__(self):
        return len(self.samples)


class PainSeqDatasets(Dataset):
    """
    Handles label files that list *frame-level paths* instead of video dirs.
    Groups frames by video, then splits each into fixed-length clips.
    """

    def __init__(self, img_dir, label_path, transform=None, seq_len=24, target_transform=None):
        super(PainSeqDatasets, self).__init__()

        self.img_dir = img_dir
        self.seq_len = seq_len
        self.transform = transform
        self.target_transform = target_transform

        df = pd.read_csv(label_path, sep=" ")

        # Group frames by video
        video_groups = defaultdict(list)
        for _, row in df.iterrows():
            frame_rel_path, label = row.iloc[0], int(row.iloc[1])
            video_dir = os.path.dirname(frame_rel_path)  # e.g., S01/V001
            video_groups[video_dir].append((frame_rel_path, label))

        self.samples = []

        # Process each video group
        for video_dir, frames_info in video_groups.items():
            frame_paths = [os.path.join(self.img_dir, f) for f, _ in sorted(frames_info)]
            label = frames_info[0][1]  # same label for all frames in video

            n_frames = len(frame_paths)
            if n_frames > seq_len:
                num_clips = n_frames // seq_len
                for i in range(num_clips):
                    start = i * seq_len
                    end = start + seq_len
                    clip_frames = frame_paths[start:end]
                    self.samples.append((video_dir, clip_frames, label))

                # leftover frames
                if n_frames % seq_len != 0:
                    leftover = frame_paths[-seq_len:]
                    self.samples.append((video_dir, leftover, label))
            else:
                # pad if needed
                clip_frames = frame_paths.copy()
                if n_frames < seq_len:
                    clip_frames += [clip_frames[-1]] * (seq_len - n_frames)
                self.samples.append((video_dir, clip_frames, label))

        print(f"[INFO] Grouped {len(video_groups)} videos into {len(self.samples)} clips")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_dir, frame_paths, label = self.samples[idx]

        frames = []
        for frame_path in frame_paths:
            img = Image.open(frame_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        frames = torch.stack(frames, dim=0)  # [T, C, H, W]

        if self.target_transform:
            label = self.target_transform(label)

        return frames, label



class TemporalPainDataset(Dataset):
    def __init__(self, img_dir, label_path, transform=None, seq_len=24, temporal_transform=None, target_transform=None):
        super().__init__()

        self.img_dir = img_dir
        self.seq_len = seq_len
        self.transform = transform
        self.temporal_transform = temporal_transform  # NEW
        self.target_transform = target_transform

        df = pd.read_csv(label_path, sep=" ")

        # Group frames by video
        video_groups = defaultdict(list)
        for _, row in df.iterrows():
            frame_rel_path, label = row.iloc[0], int(row.iloc[1])
            video_dir = os.path.dirname(frame_rel_path)
            video_groups[video_dir].append((frame_rel_path, label))

        self.samples = []
        for video_dir, frames_info in video_groups.items():
            frame_paths = [os.path.join(self.img_dir, f) for f, _ in sorted(frames_info)]
            label = frames_info[0][1]
            self.samples.append((video_dir, frame_paths, label))

        print(f"[INFO] Loaded {len(self.samples)} videos with temporal transforms enabled.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_dir, frame_paths, label = self.samples[idx]

        # Apply temporal transforms BEFORE loading frames
        if self.temporal_transform:
            frame_paths = self.temporal_transform(frame_paths)

        # Load frames
        frames = []
        for frame_path in frame_paths:
            img = Image.open(frame_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        frames = torch.stack(frames, dim=0)  # [T, C, H, W]

        if self.target_transform:
            label = self.target_transform(label)

        return frames, label, video_dir


class TemporalBAHDataset(Dataset):
    def __init__(self, img_root, label_csv, transform=None, seq_len=16,
             temporal_transform=None, target_transform=None,
             fixed_window=100):
        """
        Args:
            img_root (str): Root folder containing extracted frames for videos.
            label_csv (str): CSV file containing:
                [source_list, video, segment_id, label, num_frames, start_frame, end_frame]
            transform (callable): Image transform applied to each frame.
            seq_len (int): Model temporal sequence length (used only for padding last chunk).
            fixed_window (int): Fixed number of frames per sub-sequence (e.g. 75).
            temporal_transform (callable): Optional temporal transform on list of frame paths.
            target_transform (callable): Optional transform on the label.
        """
        super().__init__()

        self.img_root = img_root
        self.transform = transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.seq_len = seq_len
        self.fixed_window = fixed_window

        # --- Read CSV ---
        df = pd.read_csv(label_csv)
        self.samples = []

        # --- Iterate through each segment ---
        for _, row in df.iterrows():
            video_rel_path = row["video"]
            label = int(row["label"])
            start_frame = int(row["start_frame"])
            end_frame = int(row["end_frame"])
            segment_id = int(row["segment_id"])

            # Frame directory
            video_frame_dir = os.path.join(img_root, video_rel_path)

            # Collect frame paths
            frame_paths = []
            for frame_idx in range(start_frame, end_frame + 1):
                frame_file = f"frame-{frame_idx}.jpg"
                frame_path = os.path.join(video_frame_dir, frame_file)
                if os.path.exists(frame_path):
                    frame_paths.append(frame_path)
                else:
                    print(f"[WARN] Missing: {frame_path}")

            num_frames = len(frame_paths)
            if num_frames == 0:
                continue

            # --- Split logic ---
            if num_frames <= fixed_window:
                # short segment, keep as is
                self.samples.append({
                    "video": video_rel_path,
                    "segment_id": segment_id,
                    "frame_paths": frame_paths,
                    "label": label
                })
            else:
                # Split into 75-frame chunks
                start = 0
                while start < num_frames:
                    end = start + fixed_window
                    subclip = frame_paths[start:end]

                    # If last chunk smaller than seq_len → pad or borrow
                    if len(subclip) < seq_len:
                        # Borrow some frames from end of previous chunk if possible
                        borrow = seq_len - len(subclip)
                        if start - borrow >= 0:
                            subclip = frame_paths[start - borrow:start] + subclip
                        else:
                            # fallback: pad by repeating last frame
                            subclip += [subclip[-1]] * borrow

                    self.samples.append({
                        "video": video_rel_path,
                        "segment_id": segment_id,
                        "frame_paths": subclip,
                        "label": label
                    })

                    # Move window forward
                    start += fixed_window

        print(f"[INFO] Loaded {len(self.samples)} fixed-window subsequences "
            f"(window={fixed_window}, seq_len={seq_len}) from {label_csv}")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frame_paths = sample["frame_paths"]
        label = sample["label"]
        video_path = sample["video"]

        # --- Temporal transform ---
        if self.temporal_transform:
            frame_paths = self.temporal_transform(frame_paths)

        # --- Subsample or pad frames ---
        if len(frame_paths) > self.seq_len:
            # Uniform sampling across segment
            indices = torch.linspace(0, len(frame_paths) - 1, self.seq_len).long()
            frame_paths = [frame_paths[i] for i in indices]
        elif len(frame_paths) < self.seq_len:
            # Repeat last frame to fill sequence
            frame_paths += [frame_paths[-1]] * (self.seq_len - len(frame_paths))

        # --- Load frames ---
        frames = []
        for frame_path in frame_paths:
            img = Image.open(frame_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        frames = torch.stack(frames, dim=0)  # [T, C, H, W]

        if self.target_transform:
            label = self.target_transform(label)

        return frames, label, video_path


class BAHDatasets(Dataset):

    def __init__(self, img_dir, label_path, transform=None, target_transform=None):
        super(BAHDatasets, self).__init__()

        self.img_labels = pd.read_csv(label_path, delimiter=',')
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.convtensor = transforms.ToTensor()
        self.pil_image = T.ToPILImage()

    def __getitem__(self, i):
        

        label = self.img_labels.iloc[i, 1]

        if self.target_transform:
            label = self.target_transform(label)

        try:

            img_path = os.path.join(self.img_dir, self.img_labels.iloc[i, 0])
            image = self.convtensor(Image.open(img_path))
            if self.transform:
                image = self.transform(self.pil_image(image))
        except:
            print(f"Skipping invalid image at index {img_path}")
            i = (i + 1) % len(self)
            image = torch.full((1, 3, 100, 100), float('nan'))
                
        return image, label

    def __len__(self):
        return len(self.img_labels)
    
class FerImageFolder(Dataset):

    def __init__(self, imgs, labels, transform = None):
        super(FerImageFolder, self).__init__()

        self.img = imgs
        self.label = labels
        self.transform = transform

    def __getitem__(self, index):

        img = cv2.imread(self.img[index], cv2.IMREAD_COLOR)
        self.transform = self.transform
        img1 = cv2.resize(img, (100, 100))
        tensor = torch.from_numpy(img1.transpose(2, 0, 1))

        label = self.label[index]

        return tensor.float(), label, index

    def __len__(self):
        return len(self.img)


# Define a custom iterator that restarts from the beginning
class TragetRestartableIterator:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(self.dataloader)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            return next(self.iterator)