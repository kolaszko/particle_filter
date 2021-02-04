import argparse

from world import World
from filter import LandmarkParticleFilter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=640, help='width of map')
    parser.add_argument('--height', type=int, default=480, help='height of map')
    parser.add_argument('--landmarks', type=int, default=6, help='number of landmarks')
    parser.add_argument('--particles', type=int, default=500, help='number of particles')
    parser.add_argument('--std_position', type=float, default=5.0, help='position Gaussian noise standard deviation')
    parser.add_argument('--std_heading', type=float, default=0.05, help='heading Gaussian noise standard deviation')
    parser.add_argument('--steps', type=int, default=100, help='steps to go')

    args, _ = parser.parse_known_args()

    w = World(args.width, args.height, args.landmarks)
    f = LandmarkParticleFilter(args.particles, args.width, args.height, w.map.landmarks, args.std_position,
                               args.std_heading)

    for i in range(args.steps):
        dir, vel = w.tick()
        measurement = w.sense()

        f.predict(dir, vel)
        f.update(measurement)
        f.resample()

        w.show(particles=f.particles, pos_e=f.mean_position())
