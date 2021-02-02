import numpy as np
from scipy.special import softmax

from .particle_filter import ParticleFilter


class LandmarkParticleFilter(ParticleFilter):

    def __init__(self, num_particles, width, height, landmarks, std_position, std_heading):

        self.width = width
        self.height = height

        self.num_particles = num_particles
        self.particles = np.random.rand(num_particles, 2)
        self.headings = np.radians(np.random.rand(num_particles) * 360)
        self.particle_indices = np.arange(self.num_particles)
        self.weights = []

        self._init_sample()

        self.landmarks = landmarks
        self.std_position = std_position
        self.std_heading = std_heading


    def _init_sample(self):
        self.particles *= np.array([self.width, self.height])

    def resample(self):
        resample_indices = np.random.choice(self.particle_indices, self.num_particles, p=self.weights)
        self.particles = self.particles[resample_indices, :]
        self.headings = self.headings[resample_indices]

    def predict(self, direction, velocity):
        self.headings += direction + np.random.normal(0, self.std_heading)
        self.particles[:, 0] += velocity * np.sin(self.headings) + np.random.normal(0, self.std_position, (self.num_particles))
        self.particles[:, 1] += velocity * np.cos(self.headings) + np.random.normal(0, self.std_position, (self.num_particles))


    def update(self, measurement):
        w_n = []
        for p in self.particles:
            w = np.linalg.norm(p - self.landmarks, axis=-1)
            w = np.linalg.norm(measurement - w, axis=-1).squeeze()
            w_n.append(w)

        w_n /= np.sum(w_n)
        self.weights = np.clip(w_n, 1e-6, 1)
        self.weights = -np.log(self.weights)
        self.weights = softmax(self.weights)

    def mean_position(self):
        return np.mean(self.particles, axis=0)