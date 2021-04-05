import random

import pygame


class RotationNoiser:
    def __init__(self, angle_distortion=3):
        self.angle_distortion = angle_distortion

    def __call__(self, img):
        angle = random.uniform(-self.angle_distortion, self.angle_distortion)
        rotated_image = pygame.transform.rotate(img, angle)
        return rotated_image