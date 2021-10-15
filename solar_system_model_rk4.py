"""
# Object Oriented Solar System Model using the Runge-Kutta Method
# 3rd year Project
# James Kavanagh-Cranston
# 07/10/2021
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits import mplot3d
import time
import sys
import os
import pickle

class Coordinate:

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class Body:
    bodies = []

    def __init__(self, position, mass, velocity, name, colour):
        self.position = position
        self.mass = mass
        self.velocity = velocity
        self.name = name
        self.colour = colour
    
    def returnVector(self):
        return [self.position.x, self.position.y, self.position.z, self.velocity.x, self.velocity.y, self.velocity.z]
    
    def returnMass(self):
        return self.mass

    def returnName(self):
        return self.name


class Simulation:

    def __init__(self, bodies):
        self.bodies = bodies
        self.N_bodies = len(bodies)
        self.concatVector = np.zeros(0)
        for i in bodies:
            for element in i.returnVector():
                self.concatVector = np.append(self.concatVector, element)
        self.massList = np.array([i.returnMass() for i in self.bodies])
        self.nameList = [i.returnName() for i in self.bodies]

    def rk4(self, t, dt):
        a = dt * calcDiffEqs(t, self.concatVector, self.massList)
        b = dt * calcDiffEqs(t + 0.5*dt, self.concatVector + a/2, self.massList)
        c = dt * calcDiffEqs(t + 0.5*dt, self.concatVector + b/2, self.massList)
        d = dt * calcDiffEqs(t + dt, self.concatVector + b, self.massList)

        yNew = self.concatVector + ((a + 2*b + 2*c + d) / 6.0)
        return yNew
    
    def run(self, T, dt):
        self.path = [self.concatVector]
        clock_time = 0
        nsteps = int(T / dt)
        start_time = time.time()
        for step in range(nsteps):
            sys.stdout.flush()
            sys.stdout.write(f"Integrating: step = {step} / {nsteps} | Simulation time = {round(clock_time, 3)} Percentage = {round(100 * step / nsteps, 1)}%\r")
            yNew = self.rk4(0, dt)
            self.path.append(yNew)
            self.concatVector = yNew
            clock_time += dt
        runtime = time.time() - start_time
        print(f"\nSimulation completed in {runtime} seconds!")
        self.path = np.array(self.path)

    
def calcDiffEqs(t, y, masses):
    G = 6.67e-11 #m^3 kg^-1 s^-2
    N_bodies = int(len(y) / 6)
    solvedVector = np.zeros(y.size)
    for i in range(N_bodies):
        ioffset = i * 6
        for j in range(N_bodies):
            joffset = j * 6
            solvedVector[ioffset] = y[ioffset + 3]
            solvedVector[ioffset + 1] = y[ioffset + 4]
            solvedVector[ioffset + 2] = y[ioffset + 5]
            if i!= j:
                dx = y[ioffset] - y[joffset]
                dy = y[ioffset + 1] - y[joffset + 1]
                dz = y[ioffset + 2] - y[joffset + 2]
                r = math.sqrt(dx**2 + dy**2 + dz**2)
                ax = (-G * masses[j] / r**3) * dx
                ay = (-G * masses[j] / r**3) * dy
                az = (-G * masses[j] / r**3) * dz
                solvedVector[ioffset + 3] += ax
                solvedVector[ioffset + 4] += ay
                solvedVector[ioffset + 5] += az
    
    return solvedVector


##########  Body initial positions  ##########

'''
#   initial positions are held in dictionaries.
#   the below data are representative of the corresponding bodies as they were on 24th Nov 2019
#   data was obtained from NASA JPL at:
#   https://ssd.jpl.nasa.gov/horizons/app.html#/
'''

sun = {"position":Coordinate(0,0,0), "mass":1.989e30, "velocity":Coordinate(0,0,0), "colour":'darkorange', "name":'Sun'} #darkorange  pink
mercury = {"position":Coordinate(-2.754973475923117E+10, 3.971482635075326E+10, 5.772553348387497E+09), "mass":3.302e23, "velocity":Coordinate(-4.985503186396708E+04, -2.587115609586964E+04, 2.459423100025674E+03), "colour":'slategrey', "name":'Mercury'}
venus = {"position":Coordinate(6.071824980347975E+10, -9.031478095293820E+10, -4.743119158717781E+09), "mass":48.685e23, "velocity":Coordinate(2.882992914024795E+04, 1.941822077492687E+04, -1.397248807063850E+03), "colour":'red', "name":'Venus'}
earth = {"position":Coordinate(7.081801535330121E+10, 1.304740594736121E+11, -4.347298831932247E+06), "mass":5.972e24, "velocity":Coordinate(-2.659141632959534E+04, 1.428195558990953E+04, 1.506587338520049E-01), "colour":'dodgerblue', "name":'Earth'}
mars = {"position":Coordinate(-2.338256705323077E+11, -6.744910716399051E+10, 4.323713396453075E+09), "mass":6.417e23, "velocity":Coordinate(7.618913418166695E+03, -2.120844917567340E+04, -6.313535649528479E+02), "colour":'orangered', "name":'Mars'}
jupiter = {"position":Coordinate(3.640886585245620E+10, -7.832464736633219E+11, 2.438628231032491E+09), "mass":1.898e27, "velocity":Coordinate(1.290779089733536E+04, 1.225548372134438E+03, -2.938400770294290E+01), "colour":'peru', "name":'Jupiter'}
saturn = {"position":Coordinate(5.402930559845881E+11, -1.401101262550307E+12, 2.852732020323873E+09), "mass":5.683e26, "velocity":Coordinate(8.493464320782032E+03, 3.448137125239667E+03, -3.974048933366207E+01), "colour":'sandybrown', "name":'Saturn'}
uranus = {"position":Coordinate(2.440195048138449E+12, 1.684964594605102E+12, -2.534637639068711E+10), "mass":86.813e24, "velocity":Coordinate(-3.907126929953547E+03, 5.287630406417824E+03, 7.019651631039658E+00), "colour":'lightblue', "name":'Uranus'}
neptune = {"position":Coordinate(4.370905565958221E+12, -9.700637945288652E+11, -8.076862923992699E+10), "mass":102.409e24, "velocity":Coordinate(1.155455528773291E+03, 5.342194379391015E+03, -1.363189398672890E+01), "colour":'turquoise', "name":'Neptune'}
pluto = {"position":Coordinate(1.924334541511769E+12, -4.695917904493715E+12, -5.411176029132771E+10), "mass":1.307e22, "velocity":Coordinate(5.165981607247869E+03, 9.197384040335644E+01, -1.566614784409940E+03), "colour":'rosybrown', "name":'Pluto'}
moon = {"position":Coordinate(7.048971389599609E+10, 1.303131748973128E+11, 2.721761216115206E+07), "mass":7.349e22, "velocity":Coordinate(-2.612964553516142E+04, 1.331468891451992E+04, -2.786657700313278E+01), "colour":'pink', "name":'Moon'}#slategrey

# bodyNames = [sun, mercury, venus, earth, mars, jupiter, saturn, uranus, neptune, pluto, moon]
# bodyNames = [sun, mercury, venus, earth, moon, mars, jupiter, saturn, uranus, neptune, pluto]
bodyNames = [sun, mercury, venus, earth, moon, mars, jupiter, saturn, uranus, neptune]
# bodyNames = [sun, mercury, venus, earth, moon, mars, jupiter, saturn, uranus]
# bodyNames = [sun, mercury, venus, earth, moon, mars, jupiter, saturn]
# bodyNames = [sun, mercury, venus, earth, moon, mars, jupiter]
# bodyNames = [sun, mercury, venus, earth, moon, mars]
# bodyNames = [sun, mercury, venus, earth, moon]
# bodyNames = [sun, mercury, venus, earth]
# bodyNames = [sun, mercury, venus]
# bodyNames = [sun, mercury]
# bodyNames = [sun, earth]

#   for varifying resultants
'''
resultant = 'position'
for body in bodyNames:
    print(math.sqrt(body[resultant].x**2 + body[resultant].y**2 + body[resultant].z**2) / 1E12)
'''

bodyObjects = []
bodyObjects = [Body( body['position'], body['mass'], body['velocity'], body['name'], body['colour'] ) for body in bodyNames]

simulation = Simulation(bodyObjects)

#   time period in years
T = 1
#   dt in hours
dt = 10
#   outer planet
planetTo = bodyObjects[-1].name

filename = f'T={T}__dt={dt}hr__to_planet={planetTo}.data'

if os.path.isfile(filename) == False:
    simulation.run(T*365*24*60*60, dt*60*60)
    path = simulation.path

    with open(filename, 'wb') as f:
        pickle.dump(path, f)

elif os.path.isfile(filename) == True:
    with open(filename, 'rb') as f:
        path = pickle.load(f)

for index, i in enumerate(range(0, len(bodyObjects) * 6, 6)):
    plt.plot(path[-1,i], path[-1,i+1], 'o', color=bodyObjects[index].colour)

for index, i in enumerate(range(0, len(bodyObjects) * 6, 6)):
    plt.plot(path[:,i], path[:,i+1], label=bodyObjects[index].name, color=bodyObjects[index].colour)

plt.legend()
plt.axis('equal')
plt.show()

# simulation.showConcat()

#   plotting the initial positions of the planets
testerPlot = False

if testerPlot == True:
    for index, body in enumerate(bodyNames):
        if index > 9:
            print("break")
            break
        plt.plot(body['position'].x, body['position'].y, 'o', color=body['colour'], label=body['name'])
    plt.axis('equal')
    plt.legend()
    plt.show()
    plt.savefig("solarSystem.png")
