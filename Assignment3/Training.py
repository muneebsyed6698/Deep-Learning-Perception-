"""
Training script for DCGAN digit generation.
Generates handwritten digits using the MNIST dataset.
"""

import torch
import sys
from dcgan import DCGAN, plot_losses, plot_generated_images


def main():
    # Configuration
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_EPOCHS = 50
    BATCH_SIZE = 128
    LEARNING_RATE = 0.0002
    NZ = 100  # Latent vector size
    NGF = 64  # Number of generator filters
    NDF = 64  # Number of discriminator filters
    DATA_PATH = './data'
    CHECKPOINT_PATH = './checkpoints/dcgan_checkpoint.pth'
    
    print("="*60)
    print("DCGAN Handwritten Digit Generation")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print("="*60)
    
    # Initialize DCGAN
    dcgan = DCGAN(
        device=DEVICE,
        nz=NZ,
        ngf=NGF,
        ndf=NDF,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE
    )
    
    # Train
    try:
        dcgan.train(num_epochs=NUM_EPOCHS, data_path=DATA_PATH)
        
        # Save checkpoint
        dcgan.save_checkpoint(CHECKPOINT_PATH)
        
        # Plot training losses
        plot_losses(dcgan.g_losses, dcgan.d_losses, 
                   save_path='./outputs/losses.png')
        
        # Generate sample images
        print("\nGenerating 10 sample digits...")
        generated_images = dcgan.generate_samples(num_samples=10)
        plot_generated_images(generated_images, 
                            save_path='./outputs/generated_digits.png', 
                            num_images=10)
        
        print("\n" + "="*60)
        print("Training completed successfully!")
        print("="*60)
        print("Generated files:")
        print("  - checkpoints/dcgan_checkpoint.pth (model weights)")
        print("  - outputs/generated_digits.png (sample images)")
        print("  - outputs/losses.png (training curves)")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
