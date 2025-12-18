import torch
from torch import nn, optim
from torch.optim.lr_scheduler import OneCycleLR

from tqdm import tqdm

from src.Logger.Logger import Logger

def train(model, device, path_to_model, log_path, train_loader, val_loader):
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.1,  # Peak Learning Rate
        steps_per_epoch=len(train_loader),
        epochs=100,
        pct_start=0.3,  # Spend 30% of time raising LR, 70% lowering it
        anneal_strategy='cos',  # Cosine annealing
        div_factor=25.0,  # Initial LR = max_lr / 25
        final_div_factor=10000.0  # Final LR = Initial LR / 10000 (Almost zero)
    )
    criterion = nn.CrossEntropyLoss()
    epochs = 100

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

        logger.log(epoch + 1, running_loss / len(train_loader), 100 * correct / total)

        print(f"[Epoch {epoch + 1}] Validation Accuracy: {100 * correct / total:.3f} %")
        print("-" * 30)

    print("Finished Training")
    logger.close()
    
    torch.save(model.state_dict(), path_to_model)
    print(f"Model saved to path {path_to_model}")