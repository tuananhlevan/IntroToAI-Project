from torch.utils.data import DataLoader, random_split
import torch

from DownloadData import c10_test_dataset, c10_full_train_dataset
from DownloadData import c100_test_dataset, c100_full_train_dataset

torch.manual_seed(42)
BATCH_SIZE = 512

# CIFAR-10
c10_train_size = int(0.8 * len(c10_full_train_dataset))
c10_val_size = len(c10_full_train_dataset) - c10_train_size

c10_train_dataset, c10_val_dataset = random_split(c10_full_train_dataset, [c10_train_size, c10_val_size])

c10_train_loader = DataLoader(c10_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
c10_val_loader = DataLoader(c10_val_dataset, batch_size=BATCH_SIZE, shuffle=False)
c10_test_loader = DataLoader(c10_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# CIFAR-100
c100_train_size = int(0.8 * len(c100_test_dataset))
c100_val_size = len(c100_full_train_dataset) - c100_train_size

c100_train_dataset, c100_val_dataset = random_split(c100_full_train_dataset, [c100_train_size, c100_val_size])

c100_train_loader = DataLoader(c100_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
c100_val_loader = DataLoader(c100_val_dataset, batch_size=BATCH_SIZE, shuffle=False)
c100_test_loader = DataLoader(c100_test_dataset, batch_size=BATCH_SIZE, shuffle=False)