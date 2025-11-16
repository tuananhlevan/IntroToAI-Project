import torchvision
from torchvision import transforms

c100_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
])
c10_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
])

c10_full_train_dataset = torchvision.datasets.CIFAR10(root="IntroToAI-Project/data/CIFAR10",
                                                      train=True,
                                                      download=True,
                                                      transform=c10_transform)
c10_test_dataset = torchvision.datasets.CIFAR10(root="IntroToAI-Project/data/CIFAR10",
                                                train=False,
                                                download=True,
                                                transform=c10_transform)

c100_full_train_dataset = torchvision.datasets.CIFAR100(root="IntroToAI-Project/data/CIFAR100",
                                                        train=True,
                                                        download=True,
                                                        transform=c100_transform)
c100_test_dataset = torchvision.datasets.CIFAR100(root="IntroToAI-Project/data/CIFAR100",
                                                  train=False,
                                                  download=True,
                                                  transform=c100_transform)