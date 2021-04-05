import random
import pygame


class Placer:
    def __call__(self, screen, img):
        new_width, new_height = img.get_size()
        if screen.get_size()[0] < img.get_size()[0]:
            new_width = screen.get_size()[0]
        if screen.get_size()[1] < img.get_size()[1]:
            new_height = screen.get_size()[1]
        rotated_image = pygame.transform.scale(img, (new_width, new_height))
        pos_x = screen.get_size()[0] - new_width
        pos_y = screen.get_size()[1] - new_height
        screen.blit(rotated_image, (random.randint(0, pos_x), random.randint(0, pos_y)))
        return screen

