import torch
import torch.nn as nn

class ModelA_StandardCNN(nn.Module):
    def __init__(self, num_classes=17): # 17 classes for the Jute Pest dataset
        super(ModelA_StandardCNN, self).__init__()
        
        # Block 1: Input (3, 64, 64) -> Output (16, 32, 32)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2: Input (16, 32, 32) -> Output (32, 16, 16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3: Input (32, 16, 16) -> Output (64, 8, 8)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Flatten layer
        self.flatten = nn.Flatten()
        
        # Fully Connected (Dense) Layers
        self.fc1 = nn.Linear(in_features=64 * 8 * 8, out_features=128)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x

# Test the model with a dummy tensor to ensure the dimensions work
if __name__ == "__main__":
    model = ModelA_StandardCNN()
    dummy_input = torch.randn(1, 3, 64, 64) # Batch size 1, 3 channels, 64x64 image
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") 
    # Should print: Output shape: torch.Size([1, 17])