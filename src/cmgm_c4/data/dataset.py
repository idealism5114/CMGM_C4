"""Read frozen forecasting windows and labels using a split-specific allowlist."""
import numpy as np
from torch.utils.data import DataLoader, Dataset

from cmgm_c4.common import project_root, read


class FrozenDataset(Dataset):
    def __init__(self, split, neutral=False, allow_test=False):
        if split not in ('train', 'val') and not (split == 'test' and allow_test):
            raise ValueError('Only TRAIN/VAL are available without explicit TEST authorization')
        root = project_root()
        config = read(root / 'configs/project.json')
        path_key = 'test_path' if split == 'test' else 'data_path'
        folder = root / config['baseline_' + path_key if neutral else path_key]
        self.split = split
        self.window_fields = {'X': 'features'}
        if not neutral:
            self.window_fields.update(input_valid_mask='input_valid_mask',
                                      feature_valid_mask='feature_valid_mask')
        names = [*self.window_fields.values(), 'reference_dates']
        if not neutral:
            names.append('feature_applicable_mask')
        names.extend(f'{split}_{name}' for name in
                     ('origin_indices', 'window_indices', 'targets', 'label_valid_mask'))
        self.a = {name: np.load(folder / f'{name}.npy', mmap_mode='r', allow_pickle=False)
                  for name in names}
        reference_dates = self.a['reference_dates']
        if split != 'test' and str(reference_dates[-1]) > '2024-12-31':
            raise ValueError('TRAIN/VAL timeline contains TEST dates')
        self.origins = self.a[f'{split}_origin_indices']
        self.windows = self.a[f'{split}_window_indices']
        self.dates = reference_dates[self.origins].astype(str, copy=False)
        if len(np.unique(self.dates)) != len(self.dates):
            raise ValueError('Repeated forecast origins')
        expected = self.origins[:, None] + np.arange(-19, 1)
        if neutral and not np.array_equal(np.maximum(self.windows, -1), np.maximum(expected, -1)):
            raise ValueError('Window/origin alignment differs')

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        indices = self.windows[index]
        row = {}
        for field, name in self.window_fields.items():
            window = self.a[name][np.maximum(indices, 0)].copy()
            window[indices < 0] = 0
            row[field] = window
        if 'feature_applicable_mask' in self.a:
            row['feature_applicable_mask'] = self.a['feature_applicable_mask'].copy()
        row.update(Y=self.a[f'{self.split}_targets'][index].copy(),
                   label_valid_mask=self.a[f'{self.split}_label_valid_mask'][index].copy(),
                   origin_date=str(self.dates[index]))
        return row


def loader(split, neutral=False, allow_test=False, *, batch_size=64):
    return DataLoader(FrozenDataset(split, neutral, allow_test), batch_size=batch_size,
                      shuffle=False, drop_last=False, num_workers=0)
