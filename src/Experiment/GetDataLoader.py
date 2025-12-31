import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from pathlib import Path

generator = torch.Generator().manual_seed(42)
BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "BrainTumor"

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(root=DATA_DIR / "Train")
test_dataset = datasets.ImageFolder(root=DATA_DIR / "Test")

total_size = len(train_dataset)
train_size = int(0.25 * total_size)
val_size = int(0.05 * total_size)
left = total_size - train_size - val_size

train_subset, val_subset, _ = random_split(
    train_dataset,
    [train_size, val_size, left],
    generator=generator
)
test_subset = test_dataset

class ApplyTransform(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

# Apply the transformations
train_data = ApplyTransform(train_subset, transform=transform)
val_data = ApplyTransform(val_subset, transform=transform)
test_data = ApplyTransform(test_subset, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)