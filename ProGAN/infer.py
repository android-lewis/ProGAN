from typing import Generator
from model import (
    Generator
)
import torch
import torch.optim as optim
from torchvision.utils import save_image
from scipy.stats import truncnorm
import config
from math import log2

checkpoint = "./generator.pth"
gen = Generator(config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG).to(config.DEVICE)
opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
lr = config.LEARNING_RATE
n = 100

def load_checkpoint(checkpoint_file, model, optimizer, lr, n):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    
    gen.eval()
    alpha = 1.0
    for i in range(n):
        with torch.no_grad():
            noise = torch.randn(8, config.Z_DIM, 1, 1).to(config.DEVICE)
            steps = int(log2(config.START_TRAIN_AT_IMG_SIZE / 4))
            img = gen(noise, alpha, steps)
            save_image(img*0.5+0.5, f"saved_examples/img_{i}.png")

if __name__ == "__main__":
    load_checkpoint(checkpoint, gen, opt_gen, lr, n)

