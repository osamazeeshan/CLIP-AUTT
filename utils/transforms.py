from PIL import Image
from torchvision import transforms
import torch
import numpy as np


class GrayscaleToRgb:
    """Convert a grayscale image to rgb"""
    def __call__(self, image):
        image = np.array(image)
        image = np.dstack([image, image, image])
        return Image.fromarray(image)


class TemporalDownSample(object):
    def __init__(self, factor: int):
        self.factor = factor

    def __call__(self, clip):
        if isinstance(clip, list):
            clip = np.asarray(clip)
        idx = [(i % self.factor) == 0 for i in range(clip.shape[0])]
        return clip[idx]


class RandomRoll(object):
    def __init__(self, seed=0):
        self.seed = seed

    def __call__(self, seq):
        if isinstance(seq, list):
            seq = np.asarray(seq)
        start_idx = np.random.randint(0, seq.shape[0])
        return np.concatenate([seq[start_idx:], seq[:start_idx]])


class RandomSequence(object):
    def __init__(self, seq_size, on_load=False, wrap=True):
        self.seq_size = seq_size
        self.on_load = on_load
        self.wrap = wrap

    def __call__(self, clip):
        if isinstance(clip, list):
            clip = np.asarray(clip)
        if self.on_load:
            return self.call_on_load(clip)
        else:
            return self.call_on_video(clip)

    def call_on_load(self, clip):
        T = len(clip)
        rnd_start = torch.randint(T, (1,)).item()
        end_idx = rnd_start + self.seq_size

        if end_idx < T:
            new_clip = clip[rnd_start:end_idx]
        elif self.wrap:
            end_idx -= T
            new_clip = np.concatenate((clip[rnd_start:], clip[:end_idx]))
        else:
            new_clip = clip[rnd_start:]

        # Safe padding for string arrays (frame paths)
        if len(new_clip) < self.seq_size:
            pad = self.seq_size - len(new_clip)
            new_clip = list(new_clip) + [new_clip[-1]] * pad

        return new_clip
