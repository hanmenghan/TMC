import torch.utils.data

class DataProvider():

    def __init__(self, cfg, dataset, batch_size=None, shuffle=True):
        super().__init__()
        self.dataset = dataset
        if batch_size is None:
            batch_size = cfg.BATCH_SIZE
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=int(cfg.WORKERS),
            drop_last=False)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data