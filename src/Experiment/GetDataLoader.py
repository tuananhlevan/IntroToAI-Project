import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

generator = torch.Generator().manual_seed(42)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(root="BrainTumor/Train")
test_dataset = datasets.ImageFolder(root="BrainTumor/Test")

total_size = len(train_dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size

train_subset, val_subset = random_split(
    train_dataset,
    [train_size, val_size],
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
train_data = ApplyTransform(train_subset, transform=train_transform)
val_data = ApplyTransform(val_subset, transform=test_transform)
test_data = ApplyTransform(test_subset, transform=test_transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)