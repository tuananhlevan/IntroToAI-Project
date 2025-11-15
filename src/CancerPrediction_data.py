from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

torch.manual_seed(42)

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder("IntroToAI-Project\data\KidneyCancer", transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)