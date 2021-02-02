import numpy as np


class Map:
    def __init__(self, width, height, num_landmarks=10):
        self.width = width
        self.height = height
        self.landmarks = self._generate_landmarks(num_landmarks)
        # self.agent = Agent()

    def _generate_landmarks(self, num_landmarks):
        landmarks = np.random.rand(num_landmarks, 2)
        landmarks *= np.array([self.width, self.height])

        return landmarks
