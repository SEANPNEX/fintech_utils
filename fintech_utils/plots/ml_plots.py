import torch
import matplotlib.pyplot as plt
import os 

def train_and_plot(
    model, 
    train_loader, 
    valid_loader, 
    loss_fn, 
    optimizer, 
    device, 
    n_epochs=100, 
    print_every=10, 
    scheduler=None, 
    save_path='./best_model.pth',
    title='Loss over epochs'
):
    """Train a PyTorch model and plot training/validation loss over epochs.
    model: PyTorch model to be trained
    train_loader: DataLoader for training data
    valid_loader: DataLoader for validation data
    loss_fn: Loss function
    optimizer: Optimizer
    device: Device to run the training on ('cpu' or 'cuda')
    n_epochs: Number of epochs to train
    print_every: Frequency of printing loss
    scheduler: Learning rate scheduler (optional)
    save_path: Path to save the best model
    title: Title for the loss plot
    """
    model.to(device)
    train_losses, valid_losses = [], []
    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_hat = model(x_batch)
            loss = loss_fn(y_hat, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_batch.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # Validation
        valid_loss = 0.0
        with torch.no_grad():
            model.eval()
            for x_val, y_val in valid_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                y_hat = model(x_val)
                loss = loss_fn(y_hat, y_val)
                valid_loss += loss.item() * x_val.size(0)

        avg_valid_loss = valid_loss / len(valid_loader.dataset)
        valid_losses.append(avg_valid_loss)

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_valid_loss)
            else:
                scheduler.step()

        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            best_epoch = epoch
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)

        if epoch % print_every == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch:03d}: Train Loss = {avg_train_loss:.6f}, Valid Loss = {avg_valid_loss:.6f}")

    print(f"Best epoch: {best_epoch} with validation loss: {best_loss:.6f}")

    # Plotting
    plt.figure(figsize=(7, 5))
    plt.plot(train_losses, label='Training loss', color='blue')
    plt.plot(valid_losses, label='Validation loss', color='red')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    return train_losses, valid_losses, best_epoch, best_loss

def plot_predictions(y_test, y_test_pred, title='Test Predictions'):
    """
        Plot true vs predicted values
        y_test: true values
        y_test_pred: predicted values
        title: plot title
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='True')
    plt.plot(y_test_pred, label='Predicted')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_residuals(y_test, y_test_pred, title='Residuals'):
    """
        Plot residuals
        y_test: true values
        y_test_pred: predicted values
        title: plot title
    """
    residuals = y_test - y_test_pred
    plt.figure(figsize=(12, 6))
    plt.plot(residuals, label='Residuals')
    plt.axhline(0, color='red', linestyle='--')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

