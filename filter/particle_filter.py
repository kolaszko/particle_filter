import abc


class ParticleFilter(abc.ABC):

    @abc.abstractmethod
    def resample(self):
        pass

    @abc.abstractmethod
    def predict(self):
        pass

    @abc.abstractmethod
    def update(self):
        pass
