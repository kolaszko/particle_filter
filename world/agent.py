import numpy as np


class Agent:
    def __init__(self, heading, velocity, x_max, y_max):
        self.x = x_max * np.random.rand()
        self.y = y_max * np.random.rand()
        self.heading = np.radians(heading)
        self.velocity = velocity

        self.x_max = x_max
        self.y_max = y_max

    def move(self):
        turn = np.random.randint(-40, 40)
        turn = np.radians(turn)

        dx = np.sin(self.heading + turn) * self.velocity
        dy = np.cos(self.heading + turn) * self.velocity

        # If collision with edges turn around
        if not 0 < self.x + dx < self.x_max or not 0 < self.y + dy < self.y_max:
            print('Detected collision')
            turn = np.pi

        dx = np.sin(self.heading + turn) * self.velocity
        dy = np.cos(self.heading + turn) * self.velocity

        self.x += dx
        self.y += dy
        self.heading += turn

        return turn
