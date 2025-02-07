import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

# -------------------------------
# 1. Define the Generator
# -------------------------------
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input is the latent vector Z, going into a convolution
            nn.ConvTranspose2d(100, 64 * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            # State: (64*8) x 4 x 4
            nn.ConvTranspose2d(64 * 8, 64 * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            # State: (64*4) x 8 x 8
            nn.ConvTranspose2d(64 * 4, 64 * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            # State: (64*2) x 16 x 16
            nn.ConvTranspose2d(64 * 2, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # State: (64) x 32 x 32
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()  # Output is (3) x 64 x 64, values in [-1, 1]
        )

    def forward(self, input):
        return self.main(input)

# -------------------------------
# 2. Define the Discriminator
# -------------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input is (3) x 64 x 64
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (64) x 32 x 32
            nn.Conv2d(64, 64 * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (64*2) x 16 x 16
            nn.Conv2d(64 * 2, 64 * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (64*4) x 8 x 8
            nn.Conv2d(64 * 4, 64 * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (64*8) x 4 x 4
            nn.Conv2d(64 * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()  # Output is a probability
        )

    def forward(self, input):
        return self.main(input).view(-1)

# -------------------------------
# 3. Custom Weights Initialization
# -------------------------------
def weights_init(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# -------------------------------
# 4. Main Training Function
# -------------------------------
def main():
    # Set random seed for reproducibility
    manualSeed = 999
    torch.manual_seed(manualSeed)

    # Hyperparameters
    image_size = 64           # Image dimensions (64x64)
    batch_size = 64           # Batch size during training
    num_epochs = 100          # Number of epochs
    lr = 0.0002               # Learning rate
    beta1 = 0.5               # Beta1 hyperparameter for Adam optimizer
    nz = 100                  # Size of latent vector (input noise)
    ngf = 64                  # Generator feature map size (not used explicitly here)
    ndf = 64                  # Discriminator feature map size (not used explicitly here)
    nc = 3                    # Number of channels in the training images (3 for RGB)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset path (adjust the path as needed)
    # For ImageFolder, ensure that your images are in a subdirectory (e.g., a dummy class folder)
    dataroot = "./data/egyptian_hieroglyphs_initial/train"

    # Create output directory for generated images
    os.makedirs("output", exist_ok=True)

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        # Normalize to [-1, 1] as expected by the Tanh in the Generator
        transforms.Normalize([0.5] * nc, [0.5] * nc)
    ])

    # Create the dataset and dataloader.
    # Note: ImageFolder expects the directory structure to be:
    #   dataroot/class_name/image.jpg
    # If your images are not organized into class subfolders,
    # either reorganize the data or use a custom Dataset.
    dataset = datasets.ImageFolder(root=dataroot, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Initialize the Generator and Discriminator
    netG = Generator().to(device)
    netD = Discriminator().to(device)

    # Apply the custom weights initialization
    netG.apply(weights_init)
    netD.apply(weights_init)

    # Loss function and optimizers
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Labels for real and fake images
    real_label = 1.
    fake_label = 0.

    print("Starting Training Loop...")
    # Training Loop
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            ############################
            netD.zero_grad()
            # Get real images
            real_images = data[0].to(device)
            b_size = real_images.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            # Forward pass real batch through D
            output = netD(real_images)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Generate fake images batch with Generator
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake_images = netG(noise)
            label.fill_(fake_label)

            # Classify all fake batch with Discriminator
            output = netD(fake_images.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            # Compute total discriminator loss
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update Generator: maximize log(D(G(z)))
            ############################
            netG.zero_grad()
            # Since we just updated D, we want to fool it.
            label.fill_(real_label)
            output = netD(fake_images)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # Print training stats periodically
            if i % 50 == 0:
                print(f"[Epoch {epoch+1}/{num_epochs}][Batch {i}/{len(dataloader)}] "
                      f"Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} "
                      f"D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}")

        # Save generated images after each epoch
        with torch.no_grad():
            fixed_noise = torch.randn(64, nz, 1, 1, device=device)
            fake = netG(fixed_noise).detach().cpu()
        utils.save_image(fake, f"output/fake_samples_epoch_{epoch+1:03d}.png", normalize=True)

    print("Training complete!")

# -------------------------------
# 5. Multiprocessing-Safe Entry Point
# -------------------------------
if __name__ == "__main__":
    main()
