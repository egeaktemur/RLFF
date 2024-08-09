import math
import pygame

class LunarLanderDisplay:
    def __init__(self, lander, width=800, height=600):
        self.lander = lander
        self.width = width
        self.height = height
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Lunar Lander')
        self.background_color = (0, 0, 0)  # Black background
        self.lander_color = (255, 255, 255)  # White lander
        self.base_color = (0, 255, 0)  # Green base
        self.surface_color = (128, 128, 128)  # Gray surface
        self.thruster_color = (255, 0, 0)  # Red thruster fire

    def render(self, action):
        self.screen.fill(self.background_color)

        # Draw the surface
        surface_rect = pygame.Rect(0, self.height * (1 - self.lander.surface_height), self.width, self.lander.surface_height * self.height)
        pygame.draw.rect(self.screen, self.surface_color, surface_rect)

        # Draw the base on the surface
        base_rect = pygame.Rect((self.width * (1 + self.lander.base_x)/2 - 20, self.height * (1 - self.lander.surface_height)), (40, 20))
        pygame.draw.rect(self.screen, self.base_color, base_rect)

        # Calculate lander position
        lander_x = self.width * (1 + self.lander.x)/2
        lander_y = self.height * (1 - self.lander.y)

        # Create the lander surface
        lander_surface = pygame.Surface((20, 20), pygame.SRCALPHA)
        lander_surface.fill(self.lander_color)

        # Rotate the lander surface by the angle of the lander
        rotated_lander = pygame.transform.rotate(lander_surface, -math.degrees(self.lander.angle))

        # Get the new rect for the rotated surface
        rotated_rect = rotated_lander.get_rect(center=(lander_x, lander_y))

        # Draw the rotated lander on the screen
        self.screen.blit(rotated_lander, rotated_rect.topleft)

        # Draw the thruster fire based on the action
        if action == 1:  # Main engine fire
            thruster_rect = pygame.Rect(rotated_rect.centerx - 2, rotated_rect.bottom, 4, 10)
            pygame.draw.rect(self.screen, self.thruster_color, thruster_rect)
        if action == 2:  # Left-side engine fire
            thruster_rect = pygame.Rect(rotated_rect.left - 10, rotated_rect.centery - 2, 10, 4)
            pygame.draw.rect(self.screen, self.thruster_color, thruster_rect)
        if action == 3:  # Right-side engine fire
            thruster_rect = pygame.Rect(rotated_rect.right, rotated_rect.centery - 2, 10, 4)
            pygame.draw.rect(self.screen, self.thruster_color, thruster_rect)

        pygame.display.flip()
        pygame.time.wait(100)

    def close(self):
        pygame.time.wait(1000)
        pygame.quit()
