import torch
import model
from model import Generator
import numpy as np
from PIL import Image
from torchvision.utils import save_image
import albumentations as Alb
from albumentations.pytorch import ToTensorV2



if (torch.cuda.is_available()):
    DEVICE="cuda"
else:
    DEVICE = "cpu"
LEARNING_RATE =1e-4
CHECKPOINT_GEN = "genf.pth.tar"


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

if __name__=="__main__":
    gen = Generator(in_channels=3).to(DEVICE)
    opt_gen = torch.optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
    load_checkpoint(
    CHECKPOINT_GEN,
    gen,
    opt_gen,
    LEARNING_RATE,
    )

    test_transform = Alb.Compose(
        [
            Alb.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
            ToTensorV2(),
        ]
    )






    image = Image.open("LR/baboon.png")


    with torch.no_grad():
        upscaled_img = gen(
            test_transform(image=np.asarray(image))["image"]
                .unsqueeze(0)
                .to(DEVICE)
        )
    save_image(upscaled_img, f"saved/boboo17.png")
