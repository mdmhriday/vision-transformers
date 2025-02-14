import argparse
from models.vit import VisionTransformer
from datasets.dataloader import CustomDataset
from training.train import Train

def parse_args():
    parser = argparse.ArgumentParser()


    parser.add_argument('--img_size', type=int, default=224, help='Image size for ViT (default: 224)')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size for ViT (default: 16)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs (default: 10)')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads for ViT (default: 8)')
    parser.add_argument('--num_layers', type=int, default=12, help='Number of transformer layers (default: 12)')
    parser.add_argument('--dim', type=int, default=768, help='Embedding dimension for ViT (default: 768)')
    parser.add_argument('--hidden_dim', type=int, default=3072, help='Hidden dimension (default: 3072)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (default: 0.1)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers (default: 4)')
    parser.add_argument('--dataset_root', type=str, default='./Dataset', help='Root directory for dataset (default: ./Dataset)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save checkpoints (default: ./checkpoints)')
    parser.add_argument('--mixed_precision', type=bool, default=True, help='Use mixed precision training (default: True)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    data_loader = ImageDataLoader(
        dataset_name='CIFAR10', 
        batch_size=args.batch_size, 
        img_size=args.img_size, 
        num_workers=args.num_workers, 
        download=True, 
        root=args.dataset_root
    )

    
    model = VisionTransformer(
        img_size=args.img_size, 
        patch_size=args.patch_size, 
        in_channels=3, 
        num_classes=10, 
        dim=args.dim, 
        num_heads=args.num_heads, 
        num_layers=args.num_layers, 
        hidden_dim=args.hidden_dim, 
        dropout=args.dropout
    )


    trainer = Trainer(
        model=model, 
        train_loader=data_loader.train_loader, 
        val_loader=data_loader.val_loader, 
        num_epochs=args.num_epochs, 
        learning_rate=args.learning_rate, 
        batch_size=args.batch_size, 
        save_path=args.checkpoint_dir,
        mixed_precision=args.mixed_precision
    )
    print("Training start")

    trainer.train()

    Print("End Training")

if __name__ == "__main__":
    main()