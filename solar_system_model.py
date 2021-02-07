import numpy as np
import matplotlib.pyplot as plt


class Planet:
    bodies = []  # list of bodies

    def __init__(self, x, y, vx, vy, m, orbital_period, name, color, steps):
        self.bodies.append(self)  # adding each body to the body list
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.m = m
        self.dt = orbital_period * 24 * 3600 / steps  # calculating appropriate dt for each body
        self.name = name
        self.color = color
        self.steps = steps
        self.x_array = np.zeros(steps + 1)  # creating array for x coordinates
        self.y_array = np.zeros(steps + 1)  # creating array for y coordinates
        self.x_array[0] = x  # populating x array with initial position
        self.y_array[0] = y  # populating y array with initial position
        self.vx_array = np.zeros(steps + 1)  # creating array for x velocities
        self.vy_array = np.zeros(steps + 1)  # creating array for y velocities
        self.vx_array[0] = vx  # populating x array with initial speed
        self.vy_array[0] = vy  # populating y array with initial speed

    def orbit(self):
        G, M = 6.67408E-11, 1.989E30

        for i in range(self.steps):
            fx = -G * M * self.m * (self.x / ((self.x ** 2 + self.y ** 2) ** (3 / 2)))  # force in x direction
            fy = -G * M * self.m * (self.y / ((self.x ** 2 + self.y ** 2) ** (3 / 2)))  # force in x direction
            self.x += self.vx * self.dt  # updating x position
            self.y += self.vy * self.dt  # updating y position
            self.vx += (fx / self.m) * self.dt  # updating x speed
            self.vy += (fy / self.m) * self.dt  # updating y speed
            self.x_array[i + 1] = self.x  # populating x position array
            self.y_array[i + 1] = self.y  # populating y position array
            self.vx_array[i + 1] = self.vx  # populating x speed array
            self.vy_array[i + 1] = self.vy  # populating y speed array

    def plot_orbits(self):
        plt.plot(self.x_array, self.y_array, ':', color=self.color)  # plotting the orbits of the bodies
        plt.xlabel('x/m')
        plt.ylabel('y/m')
        plt.title('Trajectories of the planets')

    def plot_origin(self):
        plt.plot(0, 0, 'o', color='darkorange')  # plotting the initial positions of the bodies
        plt.plot(self.x_array[0], self.y_array[0], '.', color=self.color)


steps = 100000  # number of steps to be taken by each planet (higher number, higher accuracy)

mercury = Planet(46E9, 0, 0, 58.98E3, 0.33011E24, 87.97, 'Mercury', 'grey', steps)
venus = Planet(107.48E9, 0, 0, 35.26E3, 4.8675E24, 224.70, 'Venus', 'red', steps)
earth = Planet(147.09E9, 0, 0, 30.29E3, 5.9724E24, 365.25, 'Earth', 'blue', steps)
mars = Planet(206.62E9, 0, 0, 26.50E3, 0.64171E24, 686.98, 'Mars', 'crimson', steps)

mercury.orbit()
venus.orbit()
earth.orbit()
mars.orbit()

mercury.plot_orbits()
venus.plot_orbits()
earth.plot_orbits()
mars.plot_orbits()

plt.legend(['Mercury', 'Venus', 'Earth', 'Mars'])

mercury.plot_origin()
venus.plot_origin()
earth.plot_origin()
mars.plot_origin()

plt.show()

for body in Planet.bodies:  # for loop for each instance of the class Planet
    orbital_rad = np.sqrt(body.x_array ** 2 + body.y_array ** 2)  # creating a list of radii at each step
    orbital_vel = np.sqrt(body.vx_array ** 2 + body.vy_array ** 2)  # creating a list of speeds at each step

    print(f'{body.name}\'s maximum distance from the Sun = {max(orbital_rad)} m')
    print(f'{body.name}\'s minimum distance from the Sun = {min(orbital_rad)} m')
    print(f'{body.name}\'s mean distance from the Sun = {sum(orbital_rad) / len(orbital_rad)} m')

    print(f'{body.name}\'s maximum speed = {max(orbital_vel)} m/s')
    print(f'{body.name}\'s minimum speed = {min(orbital_vel)} m/s')
    print(f'{body.name}\'s mean speed = {sum(orbital_vel) / len(orbital_vel)} m/s\n')
