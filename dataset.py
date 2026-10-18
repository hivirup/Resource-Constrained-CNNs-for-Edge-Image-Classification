# dataset.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset

def get_dataloaders(batch_size=32):
    # 1. Define transformations
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 2. Base dataset path
    base_path = r"C:\Users\chani\OneDrive\Documents\Uni\Semester 5\EN3150 - Pattern Recognition\Jute_Pest_Dataset"

    # 3. Load folders
    train_data = datasets.ImageFolder(root=f"{base_path}\\train", transform=transform)
    val_data   = datasets.ImageFolder(root=f"{base_path}\\val", transform=transform)
    test_data  = datasets.ImageFolder(root=f"{base_path}\\test", transform=transform)

    # 4. Combine and resplit
    full_dataset = ConcatDataset([train_data, val_data, test_data])
    total_size = len(full_dataset)
    train_size = int(0.70 * total_size)
    val_size   = int(0.15 * total_size)
    test_size  = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 5. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Classes found: {train_data.classes}")

    return train_loader, val_loader, test_loader
