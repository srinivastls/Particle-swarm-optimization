from sympy import *
import random
import matplotlib.pyplot as plt
import numpy as np


class Particle:
    def __init__(self, dim, min, max, f):
        self.dim = dim
        self.pos = np.array([0.0 for i in range(dim)])
        self.vel = np.array([0.0 for i in range(dim)])

        for i in range(dim):
            self.pos[i] = min + (max-min)*random.random()
            self.vel[i] = min + (max-min)*random.random()

        self.fitness = sub(f, self.pos)
        self.pbest_val = self.fitness
        self.pbest = np.copy(self.pos)


def sub(f, x):
    val = []
    for i in range(len(x)):
        val.append(('x'+str(i+1), x[i]))
    return f.subs(val).evalf()


def plot2d(f, min, max, gbest_arr):
    x1 = []
    x2 = []
    for gbest in gbest_arr:
        x1.append(gbest[0])
        x2.append(gbest[1])

    x = np.arange(min, max, (max-min)/100)
    y = np.arange(min, max, (max-min)/100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            Z[i][j] = sub(f, (X[i][j], Y[i][j]))

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(x1, x2)
    ax.contourf(X, Y, Z)
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_surface(X, Y, Z)
    plt.show()


def plot1d(f, min, max, gbest_arr):
    x = np.arange(min, max, (max-min)/100)
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = sub(f, (x[i],))

    plt.plot(x, y)
    plt.plot(gbest_arr, [0 for i in range(len(gbest_arr))], 'o')
    plt.show()


def print_table(p):
    width = 10
    prec = 3
    for k in range(dim):
        print(f'x{k+1}'.rjust(width), end="")
    for k in range(dim):
        print(f'v{k+1}'.rjust(width), end="")
    for k in range(dim):
        print(f'pb{k+1}'.rjust(width), end="")
    print('Best'.rjust(width), end="")
    print('Fitness'.rjust(width))
    for i in range(n):
        for k in range(dim):
            print(f"{p[i].pos[k]:{width}.{prec}f}", end="")
        for k in range(dim):
            print(f"{p[i].vel[k]:{width}.{prec}f}", end="")
        for k in range(dim):
            print(f"{p[i].pbest[k]:{width}.{prec}f}", end="")
        print(f"{p[i].pbest_val:{width}.{prec}f}", end="")
        print(f"{p[i].fitness:{width}.{prec}f}")
    print('\n')


def pso(n, dim, min, max, f, max_iter):
    w = float(input("Inertia : "))
    c1 = float(input("c1 : "))
    c2 = float(input("c2 : "))

    p = np.array([Particle(dim, min, max, f) for i in range(n)])
    gbest_val_arr = []
    gbest_arr = []
    gbest = np.copy(p[0].pos)
    gbest_val = p[0].fitness

    for i in range(1, n):
        if p[i].fitness < gbest_val:
            gbest_val = p[i].fitness
            gbest = np.copy(p[i].pos)

    gbest_arr.append(gbest)
    gbest_val_arr.append(gbest_val)
    print_table(p)
    iter = 0

    while iter < max_iter:
        for i in range(n):
            r1 = random.random()
            r2 = random.random()
            new_vel = (
                (w*p[i].vel) +
                (c1*r1*(p[i].pbest-p[i].pos)) +
                (c2*r2*(gbest-p[i].pos))
            )

            for k in range(dim):
                if new_vel[k] > min and new_vel[k] < max:
                    p[i].vel[k] = new_vel[k]
                if p[i].pos[k] + p[i].vel[k] > min and p[i].pos[k] + p[i].vel[k] < max:
                    p[i].pos[k] = p[i].pos[k] + p[i].vel[k]

            p[i].fitness = sub(f, p[i].pos)

            if p[i].fitness < p[i].pbest_val:
                p[i].pbest_val = p[i].fitness
                p[i].pbest = np.copy(p[i].pos)

            if p[i].fitness < gbest_val:
                gbest_val = p[i].fitness
                gbest = np.copy(p[i].pos)

        iter += 1
        gbest_arr.append(gbest)
        gbest_val_arr.append(gbest_val)
        if(iter ==1):
            print_table(p)


    print(f"Best value : {gbest_val.round(4)}")
    print(f"Best position {gbest.round(3)}")

    return gbest_arr, gbest_val_arr


dim = int(input("Enter the Dimension : "))
var = ''
for i in range(dim):
    var += 'x'+str(i+1)+' '
x = symbols(var)
func = input("Enter the Function : ")
f = sympify(func)
min = float(input("Min : "))
max = float(input("Max : "))
n = int(input("Enter No.Of Particles : "))
max_iter = int(input("Enter Max Iterations : "))

gbest_arr, gbest_val_arr = pso(n, dim, min, max, f, max_iter)

if dim == 2:
    plot2d(f, min, max, gbest_arr)
if dim == 1:
    plot1d(f, min, max, gbest_arr)

plt.plot(gbest_val_arr)
plt.xlabel("NO OF ITERATIONS ")
plt.ylabel("FITNESS VALUE")
plt.show()
