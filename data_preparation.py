import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torchvision import transforms
import glob
import os
import pathlib
from PIL import Image

_min_ = 5
_max_ = 15
im_shape = 90
batch_size = 32

np.random.seed(0)


def targets_extraction(image_array, border_x, border_y, sample_ids):

    if not isinstance(image_array, np.ndarray) or len(image_array.shape) != 2:
        raise NotImplementedError

    border_x = int(border_x[0]), int(border_x[1])
    border_y = int(border_y[0]), int(border_y[1])

    if border_x[0] < 1 or border_x[1] < 1 or border_y[0] < 1 or border_y[1] < 1:
        raise ValueError
    remaining_x = image_array.shape[0] - border_x[0] - border_x[1]
    remaining_y = image_array.shape[1] - border_y[0] - border_y[1]
    if remaining_x < 16 or remaining_y < 16:
        raise ValueError

    known_array = np.zeros_like(image_array)
    known_array[border_x[0]:-border_x[1], border_y[0]:-border_y[1]] = 1
    input_array = image_array.copy()
    input_array[known_array == 0] = 0
    target_array = image_array[known_array == 0].copy()

    return input_array, known_array, target_array, int(sample_ids)


class GetImages(Dataset):

    def __init__(self, data):
        super().__init__()
        self.paths = sorted(glob.glob(os.path.join(data + '/**/*'), recursive=True))
        self.img = [x for x in self.paths if '.' in pathlib.Path(x).parts[-1]]
        self.borders = []
        for _ in range(len(self.img)):
            border_x = random.randint(_min_, _max_), random.randint(_min_, _max_)
            border_y = random.randint(_min_, _max_), random.randint(_min_, _max_)
            self.borders.append((border_x, border_y))

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx: int):
        self.idx = idx
        resize_transforms = transforms.Compose([
            transforms.Resize(size=im_shape),
            transforms.CenterCrop(size=(im_shape, im_shape)),
        ])
        image = Image.open(self.img[self.idx])
        image = resize_transforms(image)
        np_img = np.array(image)
        border_x, border_y = self.borders[idx]
        input_array, known_array, target_array, sample_ids = targets_extraction(np_img, border_x, border_y, idx)

        return np.clip(input_array, 0, 255), known_array, target_array, sample_ids


def my_collate_fn(batch_as_list: list):
    target_arrays = [sample[2] for sample in batch_as_list]
    ids = torch.zeros(size=(len(batch_as_list), 1), dtype=torch.float32)
    max_tar_len = np.max([len(seq) for seq in target_arrays])
    stacked_targets = torch.zeros(size=(len(batch_as_list), max_tar_len), dtype=torch.float32)
    for i, target in enumerate(target_arrays):
        stacked_targets[i, :len(target)] = torch.from_numpy(target)

    tensor_inputs = torch.zeros(size=(len(batch_as_list), 2, im_shape, im_shape), dtype=torch.float32)
    for i, (input_array, known_array, target_array, idx) in enumerate(batch_as_list):
        ids[i, :len(input_array)] = idx
        tensor_inputs[i, 0, :input_array.shape[0], :input_array.shape[1]] = torch.from_numpy(input_array)
        tensor_inputs[i, 1, :input_array.shape[0], :input_array.shape[1]] = torch.from_numpy(known_array)

    return tensor_inputs, stacked_targets, ids


my_dataset = GetImages("data")
n_samples = len(my_dataset)
shuffled_indices = np.random.permutation(n_samples)
test_set_inx = shuffled_indices[:int(n_samples / 5)]
validation_set_inx = shuffled_indices[int(n_samples / 5):int(n_samples / 5) * 2]
training_set_inx = shuffled_indices[int(n_samples / 5) * 2:]

# Create PyTorch subsets from our subset-indices
test_set = Subset(my_dataset, indices=test_set_inx)
validation_set = Subset(my_dataset, indices=validation_set_inx)
training_set = Subset(my_dataset, indices=training_set_inx)

# Create data_loaders from each subset
test_loader = DataLoader(test_set,
                         collate_fn=my_collate_fn,
                         shuffle=False,
                         batch_size=batch_size,
                         num_workers=0
                         )

validation_loader = DataLoader(validation_set,
                               collate_fn=my_collate_fn,
                               shuffle=False,
                               batch_size=batch_size,
                               num_workers=0
                               )
training_loader = DataLoader(training_set,
                             collate_fn=my_collate_fn,
                             shuffle=True,
                             batch_size=batch_size,
                             num_workers=0
                             )

# for x, y, u in training_loader:
#     print(x.shape, y.shape, u.shape)


if __name__ == "__main__":
    torch.save(test_loader, 'test_loader.pkl')
    torch.save(validation_loader, 'validation_loader.pkl')
    torch.save(training_loader, 'training_loader.pkl')
    with open('training_loader.pkl', 'rb') as rb:
        training_data = torch.load(rb)
        # print(training_data)
