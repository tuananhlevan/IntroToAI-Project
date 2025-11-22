import torch.nn.functional as F

def train(model, optimizer, X_train, y_train, epochs=1000, num_samples=5):
    for epoch in range(epochs):
        optimizer.zero_grad()
        kl = 0
        BCE_loss = 0

        for _ in range(num_samples):
            y_pred = model(X_train, sample=True)
            BCE_loss += F.binary_cross_entropy(y_pred, y_train, reduction="sum")
            kl += model.kl_loss()

        loss = ((kl / (len(X_train))) + BCE_loss) / num_samples
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 200 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss = {loss}")