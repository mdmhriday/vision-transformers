import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.vit import VisionTransformer
from training.train import Train 

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_size', type=int, default=224, help='Image size for ViT (default: 224)')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size for ViT (default: 16)')
    parser.add_argument('--embed_dim', type=int, default=768, help='Embedding dimension for ViT (default: 768)')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads for ViT (default: 8)')
    parser.add_argument('--num_layers', type=int, default=12, help='Number of transformer layers (default: 12)')
    parser.add_argument('--hidden_dim', type=int, default=3072, help='Hidden dimension (default: 3072)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (default: 0.1)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs (default: 10)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers (default: 4)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')

    return parser.parse_args()

def main():
    args = parse_args()

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)), 
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    
    train_dataset = datasets.CIFAR10(root="./Dataset", train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root="./Dataset", train=False, transform=transform, download=True)

    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    
    model = VisionTransformer(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_channels=3,
        num_classes=10,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    )

    trainer = Train(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        num_epoch=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        save_path=args.checkpoint_dir,
        weight_decay=args.weight_decay
    )

    print("Starting Training...")
    trainer.train()
    print("Training Completed.")

if __name__ == "__main__":
    main()
