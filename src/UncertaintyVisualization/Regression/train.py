def train(model, optimizer, X_train, y_train, epochs=1000, num_samples=5):
    for epoch in range(epochs):
        optimizer.zero_grad()
        kl = 0
        log_likelihood = 0
        
        for _ in range(num_samples):
            y_pred = model(X_train, sample=True)
            log_likelihood += -0.5 * ((y_train - y_pred) ** 2).sum() / 0.1**2 # Gaussian log likelihood
            kl += model.kl_loss()
        
        loss = (kl / (len(X_train))) - (log_likelihood / num_samples)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 200 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss = {loss}")