import random
import pygame


class Placer:
    def __call__(self, screen, img):
        new_width, new_height = img.get_size()
        if screen.get_size()[0] < img.get_size()[0]:
            new_width = screen.get_size()[0] - 6
        if screen.get_size()[1] < img.get_size()[1]:
            new_height = screen.get_size()[1] - 6

        rotated_image = pygame.transform.scale(img, (new_width, new_height))
        pos_x = screen.get_size()[0] - new_width
        pos_y = screen.get_size()[1] - new_height
        screen.blit(rotated_image, (random.randint(1, pos_x - 1), random.randint(1, pos_y - 1)))
        return screen

