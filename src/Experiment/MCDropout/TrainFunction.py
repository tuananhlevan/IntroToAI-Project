import torch
from torch import optim, nn
from tqdm import tqdm

from src.Logger.Logger import Logger

def train(model, device, path_to_model, log_path, train_loader, val_loader, sample_times=20):
    optimizer = optim.Adam(model.parameters(), lr=1e-4, fused=True)
    criterion = nn.CrossEntropyLoss()
    epochs = 10
    best_acc = 0

    logger = Logger(log_path)

    print("Starting Training...")
    for epoch in range(epochs):

        # --- TRAIN ---
        model.train()
        model.features.eval()
        running_loss = 0.

        train_loop = tqdm(train_loader,
                          desc=f"Epoch {epoch + 1}/{epochs} [Training]",
                          leave=True)

        for i, data in enumerate(train_loop, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            pred = 0
            for _ in range(sample_times):
                pred += model(inputs)
            loss = criterion(pred / sample_times, labels)

            loss.backward()
            optimizer.step()

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

                outputs = 0
                for _ in range(sample_times):
                    outputs += model(images)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        logger.log(epoch=epoch + 1, loss=running_loss / len(val_loader), acc=acc)

        print(f"[Epoch {epoch + 1}] Validation Accuracy: {acc:.3f}%")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), path_to_model)
            print(f"Saved best model at accuracy: {acc:.3f}%")
        print("-" * 30)

    print("Finished Training")
    logger.close()