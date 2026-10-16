import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from modelA import ModelA_StandardCNN
from dataset import get_dataloaders # Assuming you wrapped your data code in this function

# 1. Setup Device and Data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)

# 2. Reusable Training Function
def train_and_evaluate(model_class, optimizer_name, lr, momentum=0.0, epochs=20):
    print(f"\n--- Training with {optimizer_name} ---")
    model = model_class().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize the specific optimizer
    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == "SGD_Momentum":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training Phase
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        
    return train_losses, val_losses

# 3. Run the Ablation Study on Model A
if __name__ == "__main__":
    epochs = 20
    
    # Run all three optimizers
    train_loss_sgd, val_loss_sgd = train_and_evaluate(ModelA_StandardCNN, "SGD", lr=0.01, epochs=epochs)
    train_loss_mom, val_loss_mom = train_and_evaluate(ModelA_StandardCNN, "SGD_Momentum", lr=0.01, momentum=0.9, epochs=epochs)
    train_loss_adam, val_loss_adam = train_and_evaluate(ModelA_StandardCNN, "Adam", lr=0.001, epochs=epochs) # Adam usually uses a lower LR
    
    # 4. Plot the comparison for your report
    plt.figure(figsize=(10, 6))
    plt.plot(val_loss_sgd, label='SGD (Vanilla)')
    plt.plot(val_loss_mom, label='SGD with Momentum (0.9)')
    plt.plot(val_loss_adam, label='Adam')
    plt.title('Validation Loss Comparison: Optimizers on Model A')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('optimizer_comparison.png')
    plt.show()