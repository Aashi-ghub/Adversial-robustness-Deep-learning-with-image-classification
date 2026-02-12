"""
Main entry point for Adversarial Robustness Training and Evaluation
"""
import argparse
import torch
import os

from models.cnn import get_model
from training.data_loader import get_data_loaders
from training.train import train
from defense.adv_training import adversarial_train
from evaluation.robustness_eval import comprehensive_evaluation, print_evaluation_summary
from demo.app import run_demo


def main():
    parser = argparse.ArgumentParser(description='Adversarial Robustness for Deep Learning')
    
    # Mode
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'adv-train', 'evaluate', 'demo'],
                       help='Mode: train, adv-train, evaluate, or demo')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['mnist', 'cifar10', 'cifar100'],
                       help='Dataset to use')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    
    # Attack parameters
    parser.add_argument('--attack', type=str, default='pgd',
                       choices=['fgsm', 'pgd'],
                       help='Attack method for adversarial training')
    parser.add_argument('--epsilon', type=float, default=0.3,
                       help='Perturbation magnitude')
    
    # Model checkpoint
    parser.add_argument('--save-path', type=str, default='checkpoints/model.pth',
                       help='Path to save/load model')
    parser.add_argument('--load-path', type=str, default=None,
                       help='Path to load pretrained model')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create checkpoints directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)
    
    # Get data loaders
    print(f"\nLoading {args.dataset.upper()} dataset...")
    train_loader, test_loader = get_data_loaders(
        args.dataset, batch_size=args.batch_size
    )
    
    # Get model
    print(f"\nInitializing model...")
    model = get_model(args.dataset, device)
    
    # Load pretrained model if specified
    if args.load_path:
        print(f"Loading pretrained model from {args.load_path}")
        checkpoint = torch.load(args.load_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Execute based on mode
    if args.mode == 'train':
        print("\n" + "="*60)
        print("STANDARD TRAINING")
        print("="*60)
        model = train(
            model, train_loader, test_loader, 
            epochs=args.epochs, lr=args.lr, device=device,
            save_path=args.save_path
        )
        
    elif args.mode == 'adv-train':
        print("\n" + "="*60)
        print("ADVERSARIAL TRAINING")
        print("="*60)
        print(f"Attack Method: {args.attack.upper()}")
        print(f"Epsilon: {args.epsilon}")
        model = adversarial_train(
            model, train_loader, test_loader,
            epochs=args.epochs, lr=args.lr, device=device,
            attack_method=args.attack, epsilon=args.epsilon,
            save_path=args.save_path
        )
        
    elif args.mode == 'evaluate':
        print("\n" + "="*60)
        print("ROBUSTNESS EVALUATION")
        print("="*60)
        
        # Load model if not already loaded
        if not args.load_path:
            print(f"Loading model from {args.save_path}")
            checkpoint = torch.load(args.save_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Comprehensive evaluation
        results = comprehensive_evaluation(
            model, test_loader, device=device,
            epsilons=[0.0, 0.1, 0.2, 0.3]
        )
        
        # Print summary
        print_evaluation_summary(results)
        
    elif args.mode == 'demo':
        print("\n" + "="*60)
        print("DEMO MODE")
        print("="*60)
        
        model_path = args.load_path if args.load_path else args.save_path
        run_demo(model_path, args.dataset, device)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
