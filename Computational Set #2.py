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

##def lorenz(q, t, work):
##    ' q[0] = x, q[1] = y, q[2] = z '
##    ' work[0] = sigma. work[1] = b, work[2] = r '
##    
##    dx_dt =  work[0]*(q[1] - q[0])
##    dy_dt =  work[2]*q[0] - q[1] - q[0]*q[2]
##    dz_dt =  q[0]*q[1] - work[1]*q[2]
##
##    q_sol = np.array([dx_dt, dy_dt, dz_dt])
##    return q_sol
##
##
##'Time points'
##t_max = 20 # In seconds
##n = 2000
##t = np.linspace(0, t_max, n)
##
#################### PART a.) ##################
##sigma = 10
##b = 8/3
##    
##' Initial Conditions '
### Interesting initial conditions: x=1, y=1, z=1, r = 28, 0 < t < 100, n = 10000
##x = 2; y = 2; z = 5
##'Build Arrays'
##initial_conditions = np.array([x, y, z])
###work = [sigma, b, r]
##
##' r = 0 '
##r = 0
##work = [sigma, b, r]
##' Run the 4th Order Runga Kutta algorithm to solve the differential equation '
##runga_kutta_4(lorenz, t, initial_conditions,  work,  n)
##' Build Arrays to Plot '
##parse(q_array)
##' Plot the the ouput'
##graph3D(x_t, y_t, z_t, 'r = ' + str(r))
##
##
##' r = 10 '
##r = 10
##work = [sigma, b, r]
##' Run the 4th Order Runga Kutta algorithm to solve the differential equation '
##runga_kutta_4(lorenz, t, initial_conditions,  work,  n)
##' Build Arrays to Plot '
##parse(q_array)
##' Plot the the ouput'
##graph3D(x_t, y_t, z_t, 'r = ' + str(r))
##
##
##' r = 20 '
##r = 20
##work = [sigma, b, r]
##' Run the 4th Order Runga Kutta algorithm to solve the differential equation '
##runga_kutta_4(lorenz, t, initial_conditions,  work,  n)
##' Build Arrays to Plot '
##parse(q_array)
##' Plot the the ouput'
##graph3D(x_t, y_t, z_t, 'r = ' + str(r))
##
##
##
##
#################### PART b.) ##################
##' r = 28 '
##r = 28
##work = [sigma, b, r]
##' Run the 4th Order Runga Kutta algorithm to solve the differential equation '
##runga_kutta_4(lorenz, t, initial_conditions,  work,  n)
##' Build Arrays to Plot '
##parse(q_array)
##' Plot the the ouput'
##graph3D(x_t, y_t, z_t, 'r = ' + str(r))






'COMPLETED'
######################################## PROBLEM 11.7 ########################################
##def simple_lorenz(q, t, work):
##    ' q[0] = x, q[1] = y, q[2] = z '
##    ' work[0] = a. work[1] = b, work[2] = c'
##    
##    dx_dt =  -(q[1] + q[2])
##    dy_dt =  q[0] + work[0]*q[1]
##    dz_dt =  work[1] + q[2]*(q[0] - work[2])
##
##    q_sol = np.array([dx_dt, dy_dt, dz_dt])
##    return q_sol
##
##'Time points'
##t_max = 500 # In seconds
##n = 15000
##t = np.linspace(0, t_max, n)



################## PART a.) ##################
##' Initial Conditions varying the value of c'
##x = -1; y = 0; z = 0
##a = .2; b = .2
##'c points around 5.7'
##c_min = 5.0
##c_max = 6.4 
##c_points = 10
##c = np.linspace(c_min, c_max, c_points)
##
##
##for i in c:
##    'Build Arrays'
##    initial_conditions = np.array([x, y, z])
##    work = [a, b, i]
##
##    ' Run the 4th Order Runga Kutta algorithm to solve the differential equation '
##    runga_kutta_4(simple_lorenz, t, initial_conditions,  work,  n)
##
##    ' Build Arrays to Plot '
##    parse(q_array)
##    ' Plot the the ouput'
##    graph3D(x_t, y_t, z_t, 'c = ' + str(i))



################## PART b.) ##################
##' Initial Conditions varying the initial x(0), y(0), z(0) '
##initial_values = [[-1, 0, 0], [0, 1, 1], [-1, -1, -1], [2, -1, 3], [-5, -1, 4]]
##a = .2; b = .2; c = 5.7
##
##for i in initial_values:
##    'Build Arrays'
##    x_0 = i[0]
##    y_0 = i[1] 
##    z_0 = i[2]
##    
##    initial_conditions = np.array([x_0, y_0, z_0])
##    work = [a, b, c]
##
##    ' Run the 4th Order Runga Kutta algorithm to solve the differential equation '
##    runga_kutta_4(simple_lorenz, t, initial_conditions,  work,  n)
##
##    ' Build Arrays to Plot '
##    parse(q_array)
##    ' Plot the the ouput'
##    graph3D(x_t, y_t, z_t, 'x = ' + str(x_0) + ' y = ' + str(y_0) +  ' z = ' + str(z_0))


    
'NOT DONE (UNFINISHED)'
######################################## PROBLEM 11.8 ########################################
##def duffing_oscillator(q, t, work):
##    ' q[0] = x_t '
##    ' work[0] = alpha. work[1] = beta, work[2] = gamma,  work[3] = F , work(4) = w'
##    cos = np.cos
##    
##    dx_dt =  0
##    dv_dt =  F*cos(w*t) - beta*(x**3) - alpha*x - 2*gamma*dx_dt
##
##    q_sol = np.array([dx_dt, dv_dt])
##    return q_sol
##
##'Time points'
##t_max = 100 # In seconds
##n = 10000
##t = np.linspace(0, t_max, n)
##
##' Initial Conditions '
##x_t = 0
##alpha = 1; beta = .2; gamma = 0; F = 4.0; w = 30
##'Build Arrays'
##initial_conditions = np.array([x])
##work = [alpha, beta, gamma, F, w]
##' Run the 4th Order Runga Kutta algorithm to solve the differential equation '
##runga_kutta_4(duffing_oscillator, t, initial_conditions,  work,  n)
##
##
##' Build Arrays to Plot '
##x_t = []
##for i in range(0, n): x_t.append(q_array[i][1])
##
##' Plot the the ouput'
###ax, fig = plt.figure()
##plt.plot(x_t, t, lw=0.5)
##plt.show()


######################################## PROBLEM 11.12 ########################################

################## PART a.) ##################
' Setting the center point in the grid '
##center_point_pos_x = 50
##center_point_pos_y = 50
##
##' Creating the grid ' 
##nx, ny = 10, 10
a = 1
##x_limit = a*nx
##y_limit = a*ny
##
##' Setting it up in python '
###x = np.linspace(-x_limit, x_limit, nx)
###y = np.linspace(-y_limit, y_limit, ny)
##
##x = np.zeros(x_limit)
##y = np.zeros(y_limit)
##X, Y = np.meshgrid(x, y)
##
##print(X)

################## PART b.) ##################
' Particles is written in x, y coordinates ' 
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

print(particles_x[1], particles_y[1])

' Allow the particle to randomly move until adjacent to center or out of the grid '
while (particles_x[1] <= 100 and particles_y[1] <= 100) and (particles_x[1] != 50 + a or particles_y[1] != 50 + a):
    ' movement in the x or y direction '
    rand_move = random.randint(0, 1)
    ' movement is positive or negative direction ' 
    rand_dir = random.randint(0, 1)

    ' Determining which way the particle will move ' 
    if rand_move == 0:
        if rand_dir == 0: particles_x[1] = particles_x[1] - 1
        else: particles_x[1] = particles_x[1] + 1
        
    if rand_move == 1:
        if rand_dir == 0: particles_y[1] = particles_y[1] - 1
        else: particles_y[1] = particles_y[1] + 1

    ' The case the particle exits the grid '
    if (particles_x[1] < 0 or particles_x[1] > 100 or particles_y[1] < 0 or particles_y[1] > 100):
        print(particles_x[1], particles_y[1])
        #print( 'Particle exited the grid')
        sys.exit('Particle exited the grid')

    ' The particle becomes adjacent '
    if (particles_x[1] == 50 + a or particles_y[1] == 50 + a): 
        #print(particles_x[1], particles_y[1])
        'Calculate circle around the particle and call it R_min'

        'Exit the program'
        sys.exit('x = ' + str(particles_x[1]) + ', y = ' + str(particles_y[1]))

    #print(particles_x[1], particles_y[1])    

######################################## PROBLEM 11.5 ########################################













