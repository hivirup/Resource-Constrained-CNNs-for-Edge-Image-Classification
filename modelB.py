import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution:
    1. Depthwise convolution: spatial filtering per input channel.
    2. Pointwise convolution: combines information across channels.
    3. ReLU activation.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels,
            bias=False
        )

        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.relu(x)

        return x


class ModelB_LightweightCNN(nn.Module):
    """
    Lightweight CNN for the 17-class Jute Pest dataset.

    Input:
        RGB images of shape (batch_size, 3, 64, 64)

    Output:
        Raw class logits of shape (batch_size, num_classes)
    """

    def __init__(self, num_classes=17):
        super().__init__()

        # Block 1: (3, 64, 64) -> (16, 64, 64)
        self.conv1 = DepthwiseSeparableConv(
            in_channels=3,
            out_channels=16
        )
        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Block 2: (16, 32, 32) -> (32, 32, 32)
        self.conv2 = DepthwiseSeparableConv(
            in_channels=16,
            out_channels=32
        )
        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Block 3: (32, 16, 16) -> (64, 16, 16)
        self.conv3 = DepthwiseSeparableConv(
            in_channels=32,
            out_channels=64
        )
        self.pool3 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Reduce each of the 64 feature maps to one value.
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.flatten = nn.Flatten()

        # Lightweight classifier.
        self.fc = nn.Linear(
            in_features=64,
            out_features=num_classes
        )

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))

        x = self.global_avg_pool(x)
        x = self.flatten(x)

        x = self.fc(x)

        return x


def count_trainable_parameters(model):
    """Return the total number of trainable parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


if __name__ == "__main__":

    model = ModelB_LightweightCNN(num_classes=17)

    # Test with one RGB image.
    dummy_input = torch.randn(1, 3, 64, 64)

    model.eval()

    with torch.no_grad():
        output = model(dummy_input)

    total_params = count_trainable_parameters(model)

    print("Model B: Lightweight CNN")
    print("-" * 40)
    print("Input shape:", tuple(dummy_input.shape))
    print("Output shape:", tuple(output.shape))
    print(f"Trainable parameters: {total_params:,}")
    print(f"Parameter limit: 100,000")
    print("Meets parameter limit:", total_params < 100_000)

    assert output.shape == (1, 17), "Incorrect output shape!"
    assert total_params < 100_000, "Model exceeds parameter limit!"

    print("\nModel B tests passed.")