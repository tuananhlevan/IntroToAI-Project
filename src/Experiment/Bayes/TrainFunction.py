import torch
from torch import optim, nn
from tqdm import tqdm

from src.Logger.Logger import Logger

def train(model, device, path_to_model, log_path, train_loader, val_loader, sample_times=20):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    criterion = nn.CrossEntropyLoss()
    epochs = 20

    logger = Logger(log_path)

    print("Starting Training...")
    for epoch in range(epochs):
        warmup_epochs = 5
        if epoch < warmup_epochs:
            kl_weight = (epoch / warmup_epochs)
        else:
            kl_weight = 1.0

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

            loss = model.sample_elbo(
                inputs=inputs,
                labels=labels,
                criterion=criterion,
                sample_nbr=sample_times,
                complexity_cost_weight=kl_weight / len(train_loader)
            )

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        print(f"[Epoch {epoch + 1}] Training loss: {running_loss / len(train_loader):.3f}")

        # --- VALIDATE ---
        model.eval()
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

        print(f"[Epoch {epoch + 1}] Validation Accuracy: {acc:.3f} %")
        print("-" * 30)

    print("Finished Training")
    logger.close()

    torch.save(model._orig_mod.state_dict(), path_to_model)
    print(f"Saved model at: {path_to_model}")
