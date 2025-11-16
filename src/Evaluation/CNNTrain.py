import torch

from tqdm.auto import tqdm

def CNN_train(model, optimizer, criterion, epochs, device, path_to_model, train_loader, val_loader):
    print("Starting Training...")
    for epoch in range(epochs):
        # --- TRAIN ---
        model.train()
        running_loss = 0.
        
        train_loop = tqdm(train_loader, 
                      desc=f"Epoch {epoch + 1}/{epochs} [Training]", 
                      leave=False)
        
        for i, data in enumerate(train_loop, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            pred = model(inputs)
            loss = criterion(pred, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"[Epoch {epoch + 1}] Training loss: {running_loss / len(train_loader):.3f}")
        
        # --- VALIDATE ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        val_loop = tqdm(val_loader, 
                        desc=f"Epoch {epoch + 1}/{epochs} [Validation]", 
                        leave=False)
        
        with torch.no_grad():
            for data in val_loop:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                val_loss += loss.item()

        print(f"[Epoch {epoch + 1}] Validation loss: {val_loss / len(val_loader):.3f}")
        print(f"[Epoch {epoch + 1}] Validation Accuracy: {100 * correct // total} %")
        print("-" * 30)

    print("Finished Training")
    
    torch.save(model.state_dict(), path_to_model)
    
    print(f"Model saved to path {path_to_model}")