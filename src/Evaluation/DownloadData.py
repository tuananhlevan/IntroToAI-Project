import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"

c100_train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
])
c100_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
])

c10_train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])
c10_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

c10_full_train_dataset = torchvision.datasets.CIFAR10(root=DATA_DIR / "CIFAR10",
                                                      train=True,
                                                      download=True,
                                                      transform=c10_train_transform)
c10_test_dataset = torchvision.datasets.CIFAR10(root=DATA_DIR / "CIFAR10",
                                                train=False,
                                                download=True,
                                                transform=c10_test_transform)

c100_full_train_dataset = torchvision.datasets.CIFAR100(root=DATA_DIR / "CIFAR100",
                                                        train=True,
                                                        download=True,
                                                        transform=c100_train_transform)
c100_test_dataset = torchvision.datasets.CIFAR100(root=DATA_DIR / "CIFAR100",
                                                  train=False,
                                                  download=True,
                                                  transform=c100_test_transform)

torch.manual_seed(42)
BATCH_SIZE = 256

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