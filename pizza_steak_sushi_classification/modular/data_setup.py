import os

from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader

num_workers = os.cpu_count()

def create_dataloader(
        train_dir: str,
        test_dir: str,
        transform: v2.Compose,
        batch_size: int,
        num_workers: int = num_workers
):

    train_data = ImageFolder(train_dir, transform = transform)
    test_data = ImageFolder(test_dir, transform = transform)

    # or we can make a custom image data

    train_dataloader = DataLoader(train_data,
                                  batch_size= batch_size,
                                  shuffle= True,
                                  num_workers= num_workers,
                                  pin_memory= True
                                  )

    test_dataloader = DataLoader(test_data,
                                 batch_size = batch_size,
                                 shuffle= True,
                                 num_workers= num_workers,
                                 pin_memory= True
                                 )

    return train_dataloader, test_dataloader

