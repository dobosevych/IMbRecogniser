import random

import pygame


def color_cutter(color):
    return [min(max(0, color_channel), 255) for color_channel in color]


class ScreenFiller:
    def __init__(self, color_range=(240, 255), color_distortion=0.03, ):
        self.color_range = color_range
        self.color_distortion = color_distortion

    def __call__(self, screen):
        rand_color = random.randint(*self.color_range)
        color = [rand_color + random.randint(int(-255 * self.color_distortion), int(255 * self.color_distortion))
                 for _ in range(3)]
        color = color_cutter(color)
        screen.fill(color)
        return screen


class Creator:
    FONT_NAME = "USPSIMBStandard"
    FONT_SIZE = 15

    def __init__(self, color_range=(0, 127), color_distortion=0.03, font_distortion=0.15):
        self.color_range = color_range
        self.color_distortion = color_distortion
        self.font_distortion = font_distortion

    def __call__(self, sequence):
        font_size = self.FONT_SIZE + random.randint(int(self.FONT_SIZE * self.font_distortion),
                                                    int(self.FONT_SIZE * self.font_distortion))
        font = pygame.font.SysFont(self.FONT_NAME, font_size)
        rand_color = random.randint(*self.color_range)
        color = [rand_color + random.randint(int(-255 * self.color_distortion), int(255 * self.color_distortion))
                 for _ in range(3)]
        color = color_cutter(color)
        img = font.render(sequence, True, color)
        return img
