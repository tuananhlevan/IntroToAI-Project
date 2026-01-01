import torch
from torch import optim, nn
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from src.Logger.Logger import Logger

def train(model, device, path_to_model, log_path, train_loader, val_loader):
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.1,  # Peak Learning Rate
        steps_per_epoch=len(train_loader),
        epochs=20,
        pct_start=0.3,  # Spend 30% of time raising LR, 70% lowering it
        anneal_strategy='cos',  # Cosine annealing
        div_factor=25.0,  # Initial LR = max_lr / 25
        final_div_factor=10000.0  # Final LR = Initial LR / 10000 (Almost zero)
    )
    criterion = nn.CrossEntropyLoss()
    epochs = 20

    logger = Logger(log_path)

    print("Starting Training...")
    for epoch in range(epochs):

        # --- TRAIN ---
        model.train()
        running_loss = 0.

        train_loop = tqdm(train_loader,
                          desc=f"Epoch {epoch + 1}/{epochs} [Training]",
                          leave=True)

        for i, data in enumerate(train_loop, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            pred = model(inputs)
            loss = criterion(pred, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        print(f"[Epoch {epoch + 1}] Training loss: {running_loss / len(train_loader):.3f}")

        # --- VALIDATE ---
        model.eval()
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
        correct = 0
        total = 0

        val_loop = tqdm(val_loader,
                        desc=f"Epoch {epoch + 1}/{epochs} [Validation]",
                        leave=True)

        with torch.no_grad():
            for data in val_loop:
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        logger.log(epoch=epoch + 1, loss=running_loss / len(val_loader), acc=acc)

        print(f"[Epoch {epoch + 1}] Validation Accuracy: {acc:.3f}%")
        print("-" * 30)

    print("Finished Training")
    logger.close()

    torch.save(model._orig_mod.state_dict(), path_to_model)
    print(f"Saved model at: {path_to_model}")