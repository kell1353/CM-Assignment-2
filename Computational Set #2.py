from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import random
import sys



def runga_kutta_4(f, t, q, work,  n_points):
    global time_array; global q_array

    h = (t[-1] - t[0])/n_points
    t_sol, q_sol = 0, q
    
    time_array = np.zeros(n)
    q_array = []

    for i in range(0, n):
        k1 = h*f(q_sol, t_sol, work)
        k2 = h*f(q_sol + 0.5*k1, t_sol + 0.5*h, work)
        k3 = h*f(q_sol + 0.5*k2, t_sol + 0.5*h, work)
        k4 = h*f(q_sol + k3, t_sol + h, work)
  
        t_sol = t_sol + h
        time_array[i] = t_sol
        
        q_sol = q_sol + ((k1 + 2*k2 + 2*k3 + k4)/6)
        q_array.append(q_sol)


def parse(q):
    global x_t; global y_t; global z_t
    x_t, y_t, z_t = [], [], []
    
    for i in range(0, n-1): x_t.append(q_array[i][0])
    for i in range(0, n-1): y_t.append(q_array[i][1])
    for i in range(0, n-1): z_t.append(q_array[i][2])


def graph3D(x_t, y_t, z_t, title):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(x_t, y_t, z_t, lw=0.5)
    ax.set_title(title)
    
    plt.show()


# TESTING FUNCTION
##def func(y, t, work):
##    q_sol = y - t**2 +1
##    return q_sol
##
##t= np.linspace(0, 2, 4)
##initial_conditions = np.array([.5, .5])
##runga_kutta_4(func, t, initial_conditions, work, 4)
##print(q_array)

'COMPLETED'
######################################## PROBLEM 11.6 ########################################
def lorenz(q, t, work):
    ' q[0] = x, q[1] = y, q[2] = z '
    ' work[0] = sigma. work[1] = b, work[2] = r '
    
    dx_dt =  work[0]*(q[1] - q[0])
    dy_dt =  work[2]*q[0] - q[1] - q[0]*q[2]
    dz_dt =  q[0]*q[1] - work[1]*q[2]

    q_sol = np.array([dx_dt, dy_dt, dz_dt])
    return q_sol


'Time points'
t_max = 20 # In seconds
n = 2000
t = np.linspace(0, t_max, n)

#################### PART a.) ##################
sigma = 10
b = 8/3
    
' Initial Conditions '
# Interesting initial conditions: x=1, y=1, z=1, r = 28, 0 < t < 100, n = 10000
x = 2; y = 2; z = 5
'Build Arrays'
initial_conditions = np.array([x, y, z])
#work = [sigma, b, r]

' r = 0 '
r = 0
work = [sigma, b, r]
' Run the 4th Order Runga Kutta algorithm to solve the differential equation '
runga_kutta_4(lorenz, t, initial_conditions,  work,  n)
' Build Arrays to Plot '
parse(q_array)
' Plot the the ouput'
graph3D(x_t, y_t, z_t, 'r = ' + str(r))


' r = 10 '
r = 10
work = [sigma, b, r]
' Run the 4th Order Runga Kutta algorithm to solve the differential equation '
runga_kutta_4(lorenz, t, initial_conditions,  work,  n)
' Build Arrays to Plot '
parse(q_array)
' Plot the the ouput'
graph3D(x_t, y_t, z_t, 'r = ' + str(r))


' r = 20 '
r = 20
work = [sigma, b, r]
' Run the 4th Order Runga Kutta algorithm to solve the differential equation '
runga_kutta_4(lorenz, t, initial_conditions,  work,  n)
' Build Arrays to Plot '
parse(q_array)
' Plot the the ouput'
graph3D(x_t, y_t, z_t, 'r = ' + str(r))




################## PART b.) ##################
' r = 28 '
r = 28
work = [sigma, b, r]
' Run the 4th Order Runga Kutta algorithm to solve the differential equation '
runga_kutta_4(lorenz, t, initial_conditions,  work,  n)
' Build Arrays to Plot '
parse(q_array)
' Plot the the ouput'
graph3D(x_t, y_t, z_t, 'r = ' + str(r))




######################################## PROBLEM 11.7 ########################################
def simple_lorenz(q, t, work):
    ' q[0] = x, q[1] = y, q[2] = z '
    ' work[0] = a. work[1] = b, work[2] = c'
    
    dx_dt =  -(q[1] + q[2])
    dy_dt =  q[0] + work[0]*q[1]
    dz_dt =  work[1] + q[2]*(q[0] - work[2])

    q_sol = np.array([dx_dt, dy_dt, dz_dt])
    return q_sol

'Time points'
t_max = 200 # In seconds
n = 20000
t = np.linspace(0, t_max, n)



################## PART a.) ##################
' Initial Conditions varying the value of c'
x = -1; y = 0; z = 0
a = .2; b = .2
'c points around 5.7'
c_min = 5.0
c_max = 6.4 
c_points = 10
c = np.linspace(c_min, c_max, c_points)


for i in c:
    'Build Arrays'
    initial_conditions = np.array([x, y, z])
    work = [a, b, i]

    ' Run the 4th Order Runga Kutta algorithm to solve the differential equation '
    runga_kutta_4(simple_lorenz, t, initial_conditions,  work,  n)

    ' Build Arrays to Plot '
    parse(q_array)
    
    ' Plot the the ouput'
    graph3D(x_t, y_t, z_t, 'c = ' + str(i))



################## PART b.) ##################
' Initial Conditions varying the initial x(0), y(0), z(0) '
initial_values = [[-1, 0, 0], [0, 1, 1], [-1, -1, -1], [2, -1, 3], [-5, -1, 4]]
a = .2; b = .2; c = 5.7

for i in initial_values:
    'Build Arrays'
    x_0 = i[0]
    y_0 = i[1] 
    z_0 = i[2]
    
    initial_conditions = np.array([x_0, y_0, z_0])
    work = [a, b, c]

    ' Run the 4th Order Runga Kutta algorithm to solve the differential equation '
    runga_kutta_4(simple_lorenz, t, initial_conditions,  work,  n)

    ' Build Arrays to Plot '
    parse(q_array)
    ' Plot the the ouput'
    graph3D(x_t, y_t, z_t, 'x = ' + str(x_0) + ' y = ' + str(y_0) +  ' z = ' + str(z_0))


    
######################################## PROBLEM 11.8 ########################################
def duffing_oscillator(q, t, work):
    ' q[0] = x_t '
    ' work[0] = alpha. work[1] = beta, work[2] = gamma,  work[3] = F , work[4] = w'
    cos = np.cos

    x = q[0]
    dx_dt = q[1]
    alpha = work[0];  beta = work[1]; gamma = work[2]; F = work[3];  w = work[4]
    
    dx_dt =  dx_dt
    dv_dt =  F*cos(w*t) - beta*(x**3) - alpha*x - 2*gamma*dx_dt

    q_sol = np.array([dx_dt, dv_dt])
    return q_sol

'Time points'
t_max = 100 # In seconds
n = 1000
t = np.linspace(0, t_max, n)

#################### PART a.) ##################
' Initial Conditions '
dx_dt = 0
x_0 = 0
alpha = 1; beta = .2; gamma = 0; F = 1.0#; w = 0
'Build Arrays'
initial_conditions = np.array([x_0, dx_dt])

w_points = np.linspace(0,6, 100)
A = []; w_values = []

for j in w_points:
    w = j
    work = [alpha, beta, gamma, F, w]
    ' Run the 4th Order Runga Kutta algorithm to solve the differential equation '
    runga_kutta_4(duffing_oscillator, t, initial_conditions,  work,  n)

    ' Build Arrays to Plot '
    x_t = []
    for i in range(0, n):
        x_t.append(q_array[i][1])
        
    A.append(max(x_t))
    w_values.append(w)
    #print(max(x_t))

' Plot the the ouput'
#ax, fig = plt.figure()
plt.plot(w_values, A, lw=0.5)
plt.xlabel('\u03C9')
plt.ylabel('Amplitude')
plt.show()


#################### PART b.) ##################
' Initial Conditions '
dx_dt = 0
x_0 = 0
alpha = 1; beta = .2; gamma = 0; w = 1.6
'Build Arrays'
initial_conditions = np.array([x_0, dx_dt])

F_points = np.linspace(0, 100, 100)
A = []; F_values = []

for j in F_points:
    F = j
    work = [alpha, beta, gamma, F, w]
    ' Run the 4th Order Runga Kutta algorithm to solve the differential equation '
    runga_kutta_4(duffing_oscillator, t, initial_conditions,  work,  n)

    ' Build Arrays to Plot '
    x_t = []
    for i in range(0, n):
        x_t.append(q_array[i][1])
        
    A.append(max(x_t))
    F_values.append(F)
    #print(max(x_t))

'Plot the the ouput'
#ax, fig = plt.figure()
plt.plot(w_values, A, lw=0.5)
plt.title('\u03C9 = ' + str(w))
plt.xlabel('Force')
plt.ylabel('Amplitude')
plt.show()


######################################## PROBLEM 11.12 ########################################

def graph2D(center, x, y, p_x, p_y, title):
    fig, ax = plt.subplots(1)
    ax.plot(x, y, linewidth = .5)
    ax.scatter(p_x, p_y, s=1)

    ax.set_aspect(1)
    ax.set_xlim([center[0] - 10, center[0] + 10])
    ax.set_ylim([center[1] - 10, center[1] + 10])
    ax.set_title(title)
    plt.show()

def dist(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

################## PART a.) ##################
' Setting the center point in the grid '
a = 1


#################### PART b.) ##################
'Set the central particle  in x, y coordinates' 
particles_x = [50]
particles_y = [50]

check = 're-roll'
' Generate a particle at a random position that is no the center '
while check == 're-roll':
    rand_x = random.randint(0, 100) # 'replace with nx'
    rand_y = random.randint(0, 100) # 'replace with ny'

    if ((rand_x in particles_x) and (rand_y in particles_y)):
        check = 're-roll'
    else:
        check = 'continue'

particles_x.append(rand_x)
particles_y.append(rand_y)

particles_x[1] = 51
particles_y[1] = 50
print(particles_x[1], particles_y[1])

'Allow the particle to randomly move until adjacent to center or out of the grid'
while True: # (particles_x[1] <= 100 and particles_y[1] <= 100) and (particles_x[1] != 50 + a or particles_y[1] != 50 + a):

    'The case the particle exits the grid'
    if (particles_x[1] < 0 or particles_x[1] > 100 or particles_y[1] < 0 or particles_y[1] > 100):
        print(particles_x[1], particles_y[1])
        sys.exit('Particle exited the grid')

    'Check if the particle becomes adjacent to the central particle'
    if ((particles_x[1] == 50 + a or particles_x[1] == 50 - a) and particles_y[1] == 50) or ((particles_y[1] == 50 + a or particles_y[1] == 50 - a) and particles_x[1] == 50):

        'Plot particles'
        max_x = max(particles_x); min_x = min(particles_x)
        max_y = max(particles_y); min_y = min(particles_y)

        center = [(max_x + min_x)/2, (max_y + min_y)/2]
        print(center)

        'Calculate r_min'
        distance = []
        for j in range(1, len(particles_x)):
            distance.append(np.sqrt((particles_x[j] - particles_x[0])**2 + (particles_y[j] - particles_y[0])**2))
        r_min = max(distance)
        print(r_min)

        'Plot circle around the particles and calculate R_min'
        # theta goes from 0 to 2pi
        theta = np.linspace(0, 2*np.pi, 100)
        # the radius of the circle
        r = r_min
        # compute x1 and x2
        x1 = r*np.cos(theta) + center[0]
        x2 = r*np.sin(theta) + center[1]
        # create the circle
        graph2D(center, x1, x2, particles_x, particles_y, 'x = ' + str(particles_x[1]) + ', y = ' + str(particles_y[1]))
        
        print('R_min: ' + str(r))
        
        'Exit the program'
        sys.exit('x = ' + str(particles_x[1]) + ', y = ' + str(particles_y[1]))


    'Performing the random movement'
    # movement in the x or y direction 
    rand_move = random.randint(0, 1)
    # movement is positive or negative direction 
    rand_dir = random.randint(0, 1)

    'Determining which way the particle will move' 
    if rand_move == 0:
        if rand_dir == 0: particles_x[1] = particles_x[1] - 1
        else: particles_x[1] = particles_x[1] + 1
        
    if rand_move == 1:
        if rand_dir == 0: particles_y[1] = particles_y[1] - 1
        else: particles_y[1] = particles_y[1] + 1




################## PART c.) ##################
'Particles is written in x, y coordinates' 
particles_x = [51, 50, 50]
particles_y = [50, 51, 49]

'The spacing of the lattice points'
a = 1

' Set the number of particles to be generated '
N = 25

for i in range(1, N):
    check = 're-roll'
    'Generate a particle at a unique random position'
    while check == 're-roll':
        rand_x = random.randint(0, 100) # 'replace with nx'
        rand_y = random.randint(0, 100) # 'replace with ny'

        if ((rand_x in particles_x) and (rand_y in particles_y)):
            check = 're-roll'
        else:
            check = 'continue'

    particles_x.append(rand_x)
    particles_y.append(rand_y)


'Create two lists to set the particles that leave the grid or become adjacent to cluster'
exited_particles = []
stopped_particles = [(50,50)]

'Allow the particle to randomly move until adjacent to center or out of the grid'
while (len(particles_x) != 0 and len(particles_y) != 0) :
    size = len(particles_x)
    'Check if the particle becomes adjacent to any of the stopped particles'
    for i in range(0, size):
        for j in range(len(stopped_particles)):
            if dist(particles_x[i], particles_y[i], stopped_particles[j][0], stopped_particles[j][1]) == a:
                print(particles_x[i], particles_y[i])
                stopped_particles.append((particles_x[i], particles_y[i]))
                particles_x[i] = 'del'
                particles_y[i] = 'del'
                break

    for i in range(0, size):
        if (particles_x[i] != 'del' and particles_y[i] != 'del'):
            'Performing the random movement of the particles'
            # movement in the x or y direction 
            rand_move = random.randint(0, 1)
            # movement is positive or negative direction 
            rand_dir = random.randint(0, 1)

            'Setting which way the particle will move' 
            if rand_move == 0:
                if rand_dir == 0: particles_x[i] = particles_x[i] - 1
                else: particles_x[i] = particles_x[i] + 1
                
            if rand_move == 1:
                if rand_dir == 0: particles_y[i] = particles_y[i] - 1
                else: particles_y[i] = particles_y[i] + 1

    'Check if the case the particle exits the grid'
    for i in range(0, size):
        if (particles_x[i] != 'del' and particles_y[i] != 'del'):
            if (particles_x[i] < 0 or particles_x[i] > 100 or particles_y[i] < 0 or particles_y[i] > 100):
                exited_particles.append((particles_x[i], particles_y[i]))
                particles_x[i] = 'del'
                particles_y[i] = 'del'

    'Remove particles that have stopped or exited the grid'
    if ('del' in particles_x and 'del' in particles_y):
        particles_x = [p for p in particles_x if p != 'del']
        particles_y = [p for p in particles_y if p != 'del']


print(particles_x, particles_y)
print(exited_particles)
print(stopped_particles)



p_x =[]; p_y = []
for i in range(0,len(stopped_particles)):
    p_x.append(stopped_particles[i][0])
    p_y.append(stopped_particles[i][1])
    
'Get the maximum and minimum of x, y values of the stopped particles'
max_x = max(p_x); min_x = min(p_x)
max_y = max(p_y); min_y = min(p_y)

'Calculate center of cluster'
center = [(max_x + min_x)/2, (max_y + min_y)/2]
print(center)

'Calculate r_min'
distance = []
for j in range(1, len(stopped_particles)):
    distance.append(dist(p_x[j], p_y[j], p_x[0], p_y[0]))
r_min = max(distance)/2


'Plot circle around the particles and calculate R_min'
# theta goes from 0 to 2pi
theta = np.linspace(0, 2*np.pi, 100)
# the radius of the circle
r = r_min
# compute x1 and x2
x1 = r*np.cos(theta) + center[0]
x2 = r*np.sin(theta) + center[1]
# create the circle
graph2D(center, x1, x2, p_x, p_y, 'Multiple Particles')
        
print('R_min: ' + str(r))            
        



#################### PART d.) ##################
' Calculate Fractal Dimension using R_min from part c.) '
ln = np.log
D = ln(N)/ln(R_min)


######################################## PROBLEM 11.5 ########################################
def recursive_func(b, y):
    return 1 - b*y**2

fig, ax = plt.subplots()

'Simulate system for 10000 values of a linearly spaced between 0 and 2'
n = 10000
b = np.linspace(0, 2, n)

'initial condition of system'
y = 1e-5 * np.ones(n)
'number of iterations of system'
iterations = 1000
'we are going to keep last 100 iterations'
last = 100

for i in range(iterations):
    y = recursive_func(b, y)
    # We display the bifurcation diagram.
    if i >= (iterations - last):
        ax.plot(b, y, ',k', alpha=.25)

'Plot vertical lines at each of the bifurcation points'
ax.axvline(x=.75, linewidth = .75)   
ax.axvline(x=1.25, linewidth = .75)
ax.axvline(x=1.360, linewidth = .75)

ax.set_xlabel('b')
ax.set_ylabel('y')
ax.set_title('Bifurcation Diagram for the system: ' + 'yₙ₊₁ = 1 - bₙyₙ²')
plt.show()    


########## for the logistic equation ########## 
def logistic(a, x):
    return a*x*(1 - x)

##fig, ax = plt.subplots()
##
##'Simulate system for 10000 values of a linearly spaced between 2 and 4
##n = 10000
##a = np.linspace(2, 4, n)
##
##'initial condition of system'
##x = 1e-5 * np.ones(n)
##'number of iterations of system'
##iterations = 1000
##'we are going to keep last 100 iterations'
##last = 100
##
##for i in range(iterations):
##    x = logistic(a, x)
##    # We display the bifurcation diagram.
##    if i >= (iterations - last):
##        ax.plot(a, x, ',k', alpha=.25)
##
##plt.show()    




