import sqlite3
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
import pandas as pd

class CustomDataset(data.Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, tag_list_path, vocab, transform=None, is_video=False):
        """
        Args:
            root: Image or video directory.
            tag_list_path: CSV file containing image names and associated tags.
            vocab: Vocabulary wrapper.
            transform: Optional transform to be applied on a sample.
            is_video: Whether the dataset contains videos or not.
        """
        self.root = root
        self.data = pd.read_csv(tag_list_path)
        self.vocab = vocab
        self.transform = transform
        self.is_video = is_video

    def __getitem__(self, index):
        """Returns one data pair (image or video frames and caption)."""
        data = self.data.iloc[index]
        img_name = os.path.join(self.root, data['file'])
        image = Image.open(img_name).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        tags = data['tags']
        tokens = nltk.tokenize.word_tokenize(tags.lower())
        caption = [self.vocab('<start>')]
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('<end>'))
        target = torch.Tensor(caption)

        # Save captions to captions2.txt
        with open('captions.txt', 'w') as f:
            caption_text = ' '.join(tokens)
            f.write(caption_text + '\n')
        return image, target

    def __len__(self):
        return len(self.data)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (images, captions).

    Args:
        data: List of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: List indicating valid length for each caption. Length is (batch_size).
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)  # Unzip

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths


def get_loader(root, tag_list_path, vocab, transform, batch_size, shuffle, num_workers, is_video=False):
    """Returns torch.utils.data.DataLoader for custom dataset."""
    # Custom dataset
    custom_dataset = CustomDataset(root=root,
                                   tag_list_path=tag_list_path,
                                   vocab=vocab,
                                   transform=transform,
                                   is_video=is_video)

    # Data loader for custom dataset
    # This will return (images, captions, lengths) for each iteration.
    # Images: a tensor of shape (batch_size, 3, 256, 256).
    # Captions: a tensor of shape (batch_size, padded_length).
    # Lengths: a list indicating valid length for each caption. Length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader



# import torch
# import torchvision.transforms as transforms
# import torch.utils.data as data
# import os
# import pickle
# import numpy as np
# import nltk
# from PIL import Image
# from build_vocab import Vocabulary
# import ast
# import json
#
#
# class CocoDataset(data.Dataset):
#     """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
#
#     def __init__(self, root, json, vocab, transform=None, is_video=False):
#         """Set the path for images, captions and vocabulary wrapper.
#
#         Args:
#             root: image or video directory.
#             json: coco annotation file path.
#             vocab: vocabulary wrapper.
#             transform: image transformer.
#         """
#         self.root = root
#         with open(json, "r") as data_file:
#             self.data = json.load(data_file)
#         self.ids = list(self.data.keys())
#         self.vocab = vocab
#         self.transform = transform
#         self.is_video = is_video
#
#     def __getitem__(self, index):
#         """Returns one data pair (image or video frames and caption)."""
#         data = self.data
#         vocab = self.vocab
#         ann_id = self.ids[index]
#         caption = data[ann_id]['caption']
#         cat = data[ann_id]['category']
#         if self.is_video:
#             frames_path = data[ann_id]['frames']
#             frames = []
#             for frame_name in frames_path:
#                 frame = Image.open(os.path.join(self.root, frame_name)).convert('RGB')
#                 if self.transform is not None:
#                     frame = self.transform(frame)
#                 frames.append(frame)
#             frames = torch.stack(frames, dim=0)
#             # Convert caption (string) to word ids.
#             tokens = nltk.tokenize.word_tokenize(str(caption).lower())
#             caption = [vocab('<start>')]
#             caption.extend([vocab(token) for token in tokens])
#             caption.append(vocab('<end>'))
#             target = torch.Tensor(caption)
#             return frames, target
#         else:
#             img_path = os.path.join(self.root, data[ann_id]['filename'])
#             image = Image.open(img_path).convert('RGB')
#             if self.transform is not None:
#                 image = self.transform(image)
#             # Convert caption (string) to word ids.
#             tokens = nltk.tokenize.word_tokenize(str(caption).lower())
#             caption = [vocab('<start>')]
#             caption.extend([vocab(token) for token in tokens])
#             caption.append(vocab('<end>'))
#             target = torch.Tensor(caption)
#             return image, target
#
#     def __len__(self):
#         return len(self.ids)
#
#
# def collate_fn(data):
#     """Creates mini-batch tensors from the list of tuples (images or frames, captions).
#
#     We should build custom collate_fn rather than using default collate_fn,
#     because merging caption (including padding) is not supported in default.
#     Args:
#         data: list of tuple (images or frames, caption).
#             - images or frames: torch tensor of shape (num_frames, 3, 256, 256) if video or torch tensor of shape (3, 256, 256) if image.
#             - caption: torch tensor of shape (?); variable length.
#     Returns:
#         images or frames: torch tensor of shape (batch_size, num_frames, 3, 256, 256) if video or torch tensor of shape (batch_size, 3, 256, 256) if image.
#         targets: torch tensor of shape (batch_size, padded_length).
#         lengths: list; valid length for each padded caption.
#     """
#     # Sort a data list by caption length (descending order).
#     data.sort(key=lambda x: len(x[1]), reverse=True)
#     # images or frames, captions = zip(*data)
#     images_or_frames, captions = zip(*data)  # unzip
#
#     # Merge images or frames (from tuple of 3D tensor to 4D tensor).
#     if isinstance(images_or_frames[0], torch.Tensor):
#         # Video frames
#         images_or_frames = torch.stack(images_or_frames, dim=0)
#     else:
#         # Single image
#         images_or_frames = torch.stack(images_or_frames, dim=0).unsqueeze(0)
#
#     # Merge captions (from tuple of 1D tensor to 2D tensor).
#     lengths = [len(cap) for cap in captions]
#     targets = torch.zeros(len(captions), max(lengths)).long()
#     for i, cap in enumerate(captions):
#         end = lengths[i]
#         targets[i, :end] = cap[:end]
#     return images_or_frames, targets, lengths
#
#
# def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers, is_video=False):
#     """Returns torch.utils.data.DataLoader for custom coco dataset."""
#     # COCO caption dataset
#     coco = CocoDataset(root=root,
#                        json=json,
#                        vocab=vocab,
#                        transform=transform,
#                        is_video=is_video)
#
#     # Data loader for COCO dataset
#     # This will return (images or frames, captions, lengths) for each iteration.
#     # images or frames: a tensor of shape (batch_size, num_frames, 3, 224, 224) if video or (batch_size, 3, 224, 224) if image.
#     # captions: a tensor of shape (batch_size, padded_length).
#     # lengths: a list indicating valid length for each caption. length is (batch_size).
#     data_loader = torch.utils.data.DataLoader(dataset=coco,
#                                               batch_size=batch_size,
#                                               shuffle=shuffle,
#                                               num_workers=num_workers,
#                                               collate_fn=collate_fn)
#     return data_loader
