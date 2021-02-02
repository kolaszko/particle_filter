import numpy as np
import argparse

from world import World
from filter import LandmarkParticleFilter



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--landmarks', type=int, default=5)
    parser.add_argument('--particles', type=int, default=500)
    parser.add_argument('--std_position', type=float, default=5.0)
    parser.add_argument('--std_heading', type=float, default=0.05)
    parser.add_argument('--iterations', type=int, default=50)

    args, _ = parser.parse_known_args()

    w = World(args.width, args.height, args.landmarks)
    f = LandmarkParticleFilter(args.particles, args.width, args.height, w.map.landmarks, args.std_position, args.std_heading)

    for i in range(args.iterations):
        dir, vel = w.tick()
        measurement = w.sense()

        f.predict(dir, vel)
        f.update(measurement)
        f.resample()

        w.show(particles=f.particles, pos_e=f.mean_position())
