import random
import pygame
import os
from generator.creator import Creator, ScreenFiller
from generator.noiser import RotationNoiser
from generator.placer import Placer
from generator.sampler import NumberSampler, BarcodeSampler
import pandas as pd
import os
import argparse
from tqdm import tqdm

os.environ["SDL_VIDEODRIVER"] = "dummy"

def generate_barcodes(size=10, folder="../data/"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    pygame.init()
    screen = pygame.display.set_mode((256, 32))
    generator = Creator()
    noiser = RotationNoiser()
    placer = Placer()
    filler = ScreenFiller()

    images = []
    for i, sequence in enumerate(tqdm(BarcodeSampler(size))):
        screen = filler(screen)
        img = generator(sequence)
        #img = noiser(img)
        screen = placer(screen, img)
        filename = f"img{i}.jpg"
        pygame.image.save(screen, os.path.join(folder, filename))
        images.append({"sequence": sequence, "image": filename})
    df = pd.DataFrame(images)
    df.to_csv(os.path.join(folder, "data.csv"), index=False)

def generate_numbers(size=10, folder="../data/"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    pygame.init()
    screen = pygame.display.set_mode((256, 32))
    generator = Creator(font_size=20, font_name="Arial")
    noiser = RotationNoiser()
    placer = Placer()
    filler = ScreenFiller()

    images = []
    for i, sequence in enumerate(tqdm(NumberSampler(size))):
        screen = filler(screen)
        img = generator(sequence)
        #img = noiser(img)
        screen = placer(screen, img)
        filename = f"img{i}.jpg"
        pygame.image.save(screen, os.path.join(folder, filename))
        images.append({"sequence": sequence, "image": filename})
    df = pd.DataFrame(images)
    df.to_csv(os.path.join(folder, "data.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate IMb barcodes images')
    parser.add_argument('--size', default=10, type=int, help='Number of images to generate')
    parser.add_argument('--folder', default='data/', type=str, help='Number of images to generate')

    args = parser.parse_args()
    generate_barcodes(size=args.size, folder=args.folder)