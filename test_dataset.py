import torch
from dataset import get_dataloaders

def test_dataset():
    print("=" * 60)
    print("TESTING JUTE PEST DATASET")
    print("=" * 60)

    # Load the datasets
    try:
        train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)
        print("\nDataset loaded successfully!")

    except Exception as e:
        print("\nERROR: Failed to load the dataset.")
        print("Details:", e)
        return

    # Display dataset sizes
    print("\n1. DATASET SPLIT SIZES")
    print("-" * 40)

    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    test_size = len(test_loader.dataset)
    total_size = train_size + val_size + test_size

    print(f"Training samples   : {train_size} ({train_size / total_size * 100:.2f}%)")
    print(f"Validation samples : {val_size} ({val_size / total_size * 100:.2f}%)")
    print(f"Testing samples    : {test_size} ({test_size / total_size * 100:.2f}%)")
    print(f"Total samples      : {total_size}")

    # Check one batch from each split
    print("\n2. BATCH VALIDATION")
    print("-" * 40)

    loaders = [
        ("Training", train_loader),
        ("Validation", val_loader),
        ("Testing", test_loader)
    ]

    for name, loader in loaders:
        try:
            images, labels = next(iter(loader))

            print(f"\n{name} set:")
            print(f"  Image shape       : {tuple(images.shape)}")
            print(f"  Label shape       : {tuple(labels.shape)}")
            print(f"  Image data type   : {images.dtype}")
            print(f"  Label data type   : {labels.dtype}")
            print(f"  Minimum label     : {labels.min().item()}")
            print(f"  Maximum label     : {labels.max().item()}")
            print(f"  Image values range: {images.min().item():.3f} to {images.max().item():.3f}")

            # Validate expected shapes and label types
            assert images.ndim == 4, "Images should have four dimensions."
            assert images.shape[1:] == (3, 64, 64), "Images should be RGB and resized to 64x64."
            assert labels.ndim == 1, "Labels should be one-dimensional."
            assert labels.dtype == torch.int64, "Labels should have dtype torch.int64."

            print("  Status            : PASSED")

        except Exception as e:
            print(f"\nERROR in {name} set: {e}")
            return

    print("\n" + "=" * 60)
    print("ALL DATASET TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_dataset()