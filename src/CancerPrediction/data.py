from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch

torch.manual_seed(42)

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

data_dir = "IntroToAI-Project/data_split/KidneyCancer/"
train_dataset = ImageFolder(root=data_dir + "train", transform=transform)
val_dataset = ImageFolder(root=data_dir + "val", transform=transform)
test_dataset = ImageFolder(root=data_dir + "test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)