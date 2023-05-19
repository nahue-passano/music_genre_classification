import os
import torch
import numpy as np
from natsort import natsorted
from torch.utils.data import Dataset, DataLoader, random_split

DATASET_PATH = "datasets/gtzan/genres_mel_npz"
MINIBATCH_SIZE = 64
# Dataset split
TRAIN_SET_PCT = 0.8
VALID_SET_PCT = 0.15
TEST_SET_PCT = 0.05

genres_map = {
    0: "Blues",
    1: "Classical",
    2: "Country",
    3: "Disco",
    4: "HipHop",
    5: "Jazz",
    6: "Metal",
    7: "Pop",
    8: "Reggae",
    9: "Rock",
}


# Dataset class
class MusicGenreDataset(Dataset):
    def __init__(self, dataset_path: str = DATASET_PATH):
        # Sorting genres folders
        genres_folders = natsorted(os.listdir(dataset_path))
        # Initialize attributes
        self.mel_npz_paths = []
        self.genre_gt = []
        # Iterate through folders
        for gt_i, genre_i in enumerate(genres_folders):
            genre_i_path = os.path.join(dataset_path, genre_i)
            npz_from_genre = os.listdir(genre_i_path)
            npz_path = [os.path.join(genre_i_path, npz_i) for npz_i in npz_from_genre]
            # Saving path and GT in attributes
            self.mel_npz_paths = [*self.mel_npz_paths, *npz_path]
            self.genre_gt = [*self.genre_gt, *[gt_i] * len(npz_path)]

    def __getitem__(self, index):
        # Getting npz path by index
        npz_file_i = self.mel_npz_paths[index]
        gt_i = self.genre_gt[index]
        # Loading and converting NumPy to Torch's Tensor
        mel_spec_i = np.load(npz_file_i)["spectrogram"]
        mel_spec_tensor_i = torch.from_numpy(mel_spec_i).float().unsqueeze(0)
        gt_tensor_i = torch.tensor(gt_i)

        return mel_spec_tensor_i, gt_tensor_i

    def __len__(self):
        return len(self.genre_gt)


music_genre_dataset = MusicGenreDataset()

# Split dataset into train, validation and test set
train_set, valid_set, test_set = random_split(
    dataset=music_genre_dataset,
    lengths=[TRAIN_SET_PCT, VALID_SET_PCT, TEST_SET_PCT],
    generator=torch.Generator().manual_seed(42),
)

# Dataloaders
train_loader = DataLoader(
    dataset=train_set, batch_size=MINIBATCH_SIZE, shuffle=True, num_workers=4
)

valid_loader = DataLoader(
    dataset=valid_set, batch_size=MINIBATCH_SIZE, shuffle=True, num_workers=4
)

test_loader = DataLoader(
    dataset=test_set, batch_size=MINIBATCH_SIZE, shuffle=True, num_workers=4
)
