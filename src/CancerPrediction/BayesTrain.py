import torch

from data import train_loader, test_loader, val_loader

from tqdm.auto import tqdm

def Bayes_train(model, optimizer, criterion, epochs, device, train_loader=train_loader, val_loader=val_loader):
    print("Starting Training...")
    for epoch in range(epochs):
        # --- TRAIN ---
        model.train()
        running_loss = 0.
        
        train_loop = tqdm(train_loader, 
                      desc=f"Epoch {epoch + 1}/{epochs} [Training]", 
                      leave=False) # leave=False removes bar on completion
        
        for i, data in enumerate(train_loop, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            loss = model.sample_elbo(
                inputs=inputs,
                labels=labels,
                criterion=criterion,
                sample_nbr=3)
            
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
        
        with torch.no_grad(): # No gradients needed for validation
            for data in val_loop:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                
                loss = model.sample_elbo(
                    inputs=images,
                    labels=labels,
                    criterion=criterion,
                    sample_nbr=3)
                
                val_loss += loss.item()
                
                outputs = model(images) 
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"[Epoch {epoch + 1}] Validation loss: {val_loss / len(val_loader):.3f}")
        print(f"[Epoch {epoch + 1}] Validation Accuracy: {100 * correct // total} %")
        print("-" * 30)

    print("Finished Training")
        