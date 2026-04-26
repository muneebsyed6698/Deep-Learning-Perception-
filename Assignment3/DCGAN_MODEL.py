"""
DCGAN Implementation for Handwritten Digit Generation
Based on: Unsupervised Representation Learning with Deep Convolutional GANs
"Radford, Alec; Metz, Luke; Chintala, Soumith"
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import os


class Generator(nn.Module):
    """Generator network that converts noise to images."""
    
    def __init__(self, nz=100, ngf=64, nc=1):
        """
        Args:
            nz (int): Size of latent vector
            ngf (int): Number of generator filters
            nc (int): Number of channels (1 for grayscale)
        """
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input: (batch_size, nz, 1, 1)
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # (batch_size, ngf*4, 4, 4)
            
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # (batch_size, ngf*2, 8, 8)
            
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # (batch_size, ngf, 16, 16)
            
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # (batch_size, nc, 32, 32) -> we'll resize to 28x28
        )
    
    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    """Discriminator network that classifies real vs generated images."""
    
    def __init__(self, ndf=64, nc=1):
        """
        Args:
            ndf (int): Number of discriminator filters
            nc (int): Number of channels (1 for grayscale)
        """
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input: (batch_size, nc, 28, 28)
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (batch_size, ndf, 14, 14)
            
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (batch_size, ndf*2, 7, 7)
            
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (batch_size, ndf*4, 3, 3)
            
            nn.Conv2d(ndf * 4, 1, 3, 1, 0, bias=False),
            nn.Sigmoid()
            # (batch_size, 1, 1, 1)
        )
    
    def forward(self, x):
        return self.main(x).view(-1)


class DCGAN:
    """DCGAN trainer for generating handwritten digits."""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu',
                 nz=100, ngf=64, ndf=64, batch_size=128, lr=0.0002, beta1=0.5):
        """
        Initialize DCGAN components.
        
        Args:
            device (str): 'cuda' or 'cpu'
            nz (int): Size of latent vector
            ngf (int): Number of generator filters
            ndf (int): Number of discriminator filters
            batch_size (int): Batch size for training
            lr (float): Learning rate
            beta1 (float): Beta1 for Adam optimizer
        """
        self.device = device
        self.nz = nz
        self.batch_size = batch_size
        
        # Initialize networks
        self.generator = Generator(nz=nz, ngf=ngf, nc=1).to(device)
        self.discriminator = Discriminator(ndf=ndf, nc=1).to(device)
        
        # Initialize weights
        self._init_weights()
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Optimizers
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
        
        # History
        self.g_losses = []
        self.d_losses = []
    
    def _init_weights(self):
        """Initialize network weights following DCGAN guidelines."""
        for module in [self.generator, self.discriminator]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, 0.0, 0.02)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, 0.0, 0.02)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.normal_(m.weight, 1.0, 0.02)
                    nn.init.constant_(m.bias, 0)
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch."""
        for batch_idx, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(self.device)
            
            batch_size = real_images.size(0)
            real_label = torch.ones(batch_size, device=self.device)
            fake_label = torch.zeros(batch_size, device=self.device)
            
            # Train Discriminator
            self.optimizer_d.zero_grad()
            
            # Real images
            output_real = self.discriminator(real_images)
            loss_d_real = self.criterion(output_real, real_label)
            
            # Fake images
            z = torch.randn(batch_size, self.nz, 1, 1, device=self.device)
            fake_images = self.generator(z)
            output_fake = self.discriminator(fake_images.detach())
            loss_d_fake = self.criterion(output_fake, fake_label)
            
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            self.optimizer_d.step()
            
            # Train Generator
            self.optimizer_g.zero_grad()
            
            z = torch.randn(batch_size, self.nz, 1, 1, device=self.device)
            fake_images = self.generator(z)
            output = self.discriminator(fake_images)
            loss_g = self.criterion(output, real_label)
            
            loss_g.backward()
            self.optimizer_g.step()
            
            # Store losses
            self.g_losses.append(loss_g.item())
            self.d_losses.append(loss_d.item())
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                      f"Loss_D: {loss_d.item():.4f}, Loss_G: {loss_g.item():.4f}")
    
    def train(self, num_epochs=50, data_path='./data'):
        """
        Train the DCGAN.
        
        Args:
            num_epochs (int): Number of training epochs
            data_path (str): Path to store MNIST data
        """
        # Data loading
        transform = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
        ])
        
        mnist_dataset = datasets.MNIST(root=data_path, train=True, 
                                       transform=transform, download=True)
        dataloader = DataLoader(mnist_dataset, batch_size=self.batch_size, 
                               shuffle=True, num_workers=0)
        
        print(f"Training on {self.device}")
        print(f"Total batches per epoch: {len(dataloader)}")
        
        for epoch in range(num_epochs):
            self.train_epoch(dataloader, epoch)
            print(f"Epoch {epoch} completed")
    
    def generate_samples(self, num_samples=10):
        """
        Generate fake images.
        
        Args:
            num_samples (int): Number of images to generate
            
        Returns:
            Tensor of generated images
        """
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.nz, 1, 1, device=self.device)
            fake_images = self.generator(z)
        self.generator.train()
        return fake_images
    
    def save_checkpoint(self, path):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        print(f"Checkpoint loaded from {path}")


def plot_losses(g_losses, d_losses, save_path='losses.png'):
    """Plot training losses."""
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('DCGAN Training Losses')
    plt.savefig(save_path)
    plt.close()
    print(f"Loss plot saved to {save_path}")


def plot_generated_images(images, save_path='generated_digits.png', num_images=10):
    """
    Plot and save generated images.
    
    Args:
        images (Tensor): Generated images
        save_path (str): Path to save the plot
        num_images (int): Number of images to display
    """
    fig = plt.figure(figsize=(15, 3))
    
    for i in range(min(num_images, len(images))):
        ax = fig.add_subplot(2, 5, i + 1)
        img = images[i].cpu().detach().squeeze().numpy()
        # Denormalize from [-1, 1] to [0, 1]
        img = (img + 1) / 2
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Generated images saved to {save_path}")
