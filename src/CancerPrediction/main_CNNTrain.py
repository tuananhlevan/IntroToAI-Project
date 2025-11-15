import torch
from torch import optim, nn
from torch.nn import functional as F

from CNNModel import CNN
from CNNTrain import CNN_train

torch.manual_seed(42)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = CNN().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
EPOCHS = 10

CNN_train(model=model, optimizer=optimizer, criterion=criterion, epochs=EPOCHS, device=DEVICE)