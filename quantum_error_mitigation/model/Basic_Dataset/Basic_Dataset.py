import torch.utils.data as data


class Basic_Dataset(data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        return dict(X=self.X[idx], y=self.y[idx])

    def __len__(self):
        return self.X.shape[0]
