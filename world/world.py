import numpy as np
import matplotlib.pyplot as plt

from .agent import Agent
from .map import Map


class World:
    def __init__(self, width, height, landmarks):
        self.width = width
        self.height = height

        self.map = Map(width, height, num_landmarks=landmarks)
        self.agent = Agent(heading=0, velocity=10, x_max=self.width, y_max=self.height)

        self.path_gt = []
        self.path_estimated = []

        self.figure = plt.figure(figsize=(self.width / 100, self.height / 100), dpi=100)
        self.ax = self.figure.gca()

    def show(self, particles, pos_e):
        # Set figure limits
        self.ax.set_xlim((0, self.width))
        self.ax.set_ylim((0, self.height))

        # Show landmarks
        self.ax.scatter(self.map.landmarks[:, 0], self.map.landmarks[:, 1], marker='*', color='k', label='landmarks')

        # Show particles
        self.ax.scatter(particles[:, 0], particles[:, 1], marker='.', color='b', alpha=0.5, label='particles')

        # Show actual position
        self.ax.scatter(self.agent.x, self.agent.y, marker='o', color='g', label='gt position')

        # Show ground truth path
        p = np.asarray(self.path_gt)
        self.ax.plot(p[:, 0], p[:, 1], color='g', label='gt path')

        # Append estimated pose
        self.path_estimated.append(pos_e)

        # Show estimated pose
        self.ax.scatter(pos_e[0], pos_e[1], marker='o', color='m', alpha=0.5, label='predicted position')

        # Show estimated path
        p_hat = np.asarray(self.path_estimated)
        self.ax.plot(p_hat[:, 0], p_hat[:, 1], color='m', label='predicted path')

        self.ax.legend(loc='upper right')

        plt.pause(.5)
        plt.cla()

    def tick(self):
        direction = self.agent.move()
        self.path_gt.append([self.agent.x, self.agent.y])
        return direction, self.agent.velocity

    def sense(self):
        d = np.linalg.norm(self.map.landmarks - [self.agent.x, self.agent.y], axis=-1)

        # Add noise to sensor
        d += np.random.normal(0, 0.2, d.shape)
        return d
