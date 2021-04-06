import random

import pygame


class RotationNoiser:
    def __init__(self, angle_distortion=3):
        self.angle_distortion = angle_distortion

    def __call__(self, img):
        angle = random.uniform(-self.angle_distortion, self.angle_distortion)
        rotated_image = pygame.transform.rotate(img, angle)
        return rotated_image

class SaltAndPepperNoiser:

    def __init__(self, width=256, height=32):
        self.width = width
        self.height = height

    def __call__(self, img):
        # randomly add white pixels
        number_of_pixels = random.randint(200, 1000)
        for i in range(number_of_pixels):
            y = random.randint(0, self.height - 1)
            x = random.randint(0, self.width - 1)
            img.set_at((x, y), (255, 255, 255))

        # randomly add black pixels
        number_of_pixels = random.randint(200, 1000)
        for i in range(number_of_pixels):
            y = random.randint(0, self.height - 1)
            x = random.randint(0, self.width - 1)
            img.set_at((x, y), (0, 0, 0))

        return img
