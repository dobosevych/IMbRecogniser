import random
import pygame
import os
from generator.creator import Creator, ScreenFiller
from generator.noiser import RotationNoiser
from generator.placer import Placer
from generator.sampler import Sampler
import pandas as pd

def generate(size=10, in_folder="../data/"):
    pygame.init()
    screen = pygame.display.set_mode((256, 32))
    generator = Creator()
    noiser = RotationNoiser()
    placer = Placer()
    filler = ScreenFiller()

    images = []
    for i, sequence in enumerate(Sampler(size)):
        screen = filler(screen)
        img = generator(sequence)
        img = noiser(img)
        screen = placer(screen, img)
        filename = f"img{i}.jpg"
        pygame.image.save(screen, os.path.join(in_folder, filename))
        images.append({"sequence": sequence, "image": filename})
    df = pd.DataFrame(images)
    df.to_csv(os.path.join(in_folder, "data.csv"))


if __name__ == "__main__":
    generate(size=10)