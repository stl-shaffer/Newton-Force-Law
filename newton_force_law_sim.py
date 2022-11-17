"""
Scott Shaffer
Physics 270
Final Project

This is a code that uses arrays, Newton's second law, and Euler's method to animate the position of 6 masses connected in series (in the shape of a hexagon). The primary method of this code is to use the relative positions of two masses on a spring to calculate a force vector that points from one mass to the equilibrium position of the spring.

Fundamental features of the system:
    - the system is in 2 dimensions
    - springs never cross through other springs (the no broken springs rule)
    - collisions between masses and springs are perfectly elastic
"""
import math as m
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random as ran #not used in this code, to be implemented later
"""
abbreviations as follows:
x,y: <reference to coordinate axes, when attached to another abbreviation (e.g. px1) it denotes the portion of that quantity in the x or y direction>
t: <time>
p1,p2,etc.: <p denotes position, when followed by a number it denotes the position of a mass>
v: <denoting the velocity of a given mass>
dist: <abbreviation for distance>
dp,dy,etc.: <d denotes the derivative of a quantity>
f: <denotes the force supplied by some spring. The force array differs from the position and velocity arrays. In the position array px[1] is the x position of the first mass px[2] the x-position of the second et cetera. This also applied to the velocity and derivative arrays. There are 12 forces, so the force array doesn't share this convenient symmetry>
s12,s23,etc.: <s denotes a spring and the two numbers that follow denote the masses connected by the spring>
len: <length>
len_0: <the current length of a given spring>
eq: <equilibrium>
fcon: <force constant>
pl and pr: <in the abbreviation eq_x_pr/eq_y_pl pr and pl denote the equilibrium positions of a given spring. In one dimension one of these are to the left and one to the right - so l and r seemed appropriate>
"""
t = 0.0
dt = 0.1
endtime = 30 #conservation of energy breaks around 12 seconds at dt = 0.01
num_of_p_and_1 = 7 #total number of masses (6) + 1. I made a seven element array because I wanted to ignore element zero for the sake of readability
num_of_frames = int(endtime/dt+2.0) #there's a frame at each time, frame being one distinct window in my animation, so the total number of frames = total time/timestep
"""properties of the system / energy array"""
m_p1 = 1.0 #mass of a point
m_p2 = 1.0
m_p3 = 1.0
m_p4 = 1.0
m_p5 = 1.0
m_p6 = 1.0
s12_fcon = 1.0 #force constant
s12_eq_len = 1.0 #equilibrium length
s23_fcon = 1.0
s23_eq_len = 1.0
s34_fcon = 1.0
s34_eq_len = 1.0
s45_fcon = 1.0
s45_eq_len = 1.0
s56_fcon = 1.0
s56_eq_len = 1.0
s61_fcon = 1.0
s61_eq_len = 1.0
energy_in_the_system = np.zeros(num_of_frames)
"""x and y position arrays""" #I made four arrays (px,py,vx,vy) which are together the equivalent of the dy we've been using
p1x = np.zeros(num_of_frames) #holds the position of the first mass for every frame (for the animation)
p2x = np.zeros(num_of_frames)
p3x = np.zeros(num_of_frames)
p4x = np.zeros(num_of_frames)
p5x = np.zeros(num_of_frames)
p6x = np.zeros(num_of_frames)
px = np.zeros(num_of_p_and_1) #px has 6 elements (the position of each mass)
dpxdt = np.zeros(num_of_p_and_1) #along with the four arrays which are together dy, I made four arrays (dpxdt,dpydt,dvxdt,and dvydt) which are together dydt
p1y = np.zeros(num_of_frames) #holds p1y at all times t (for the animation)
p2y = np.zeros(num_of_frames)
p3y = np.zeros(num_of_frames)
p4y = np.zeros(num_of_frames)
p5y = np.zeros(num_of_frames)
p6y = np.zeros(num_of_frames)
py = np.zeros(num_of_p_and_1) #the y-position of each coordinate
dpydt = np.zeros(num_of_p_and_1)
px[1] = 1.0 #px[1] is the x-position of the first mass
py[1] = 0.0
px[2] = 0.7
py[2] = 0.7
px[3] = -0.7
py[3] = 0.7
px[4] = -1.0
py[4] = 0
px[5] = -0.7
py[5] = -0.6
px[6] = 1.2
py[6] = -1.2
"""x and y velocity arrays"""
vx = np.zeros(num_of_p_and_1)
dvxdt = np.zeros(num_of_p_and_1)
vy = np.zeros(num_of_p_and_1)
dvydt = np.zeros(num_of_p_and_1)
vx[1] = 0.0
vy[1] = 0.0
vx[2] = 0.0
vy[2] = 0.0
vx[3] = 0.0
vy[3] = 0.0
vx[4] = 0.0
vy[4] = 0.0
vx[5] = 0.0
vy[5] = 0.0
vx[6] = 0.0
vy[6] = 0.0
"""x and y force array"""
fx = np.zeros(2*num_of_p_and_1) #there are twice as many forces as there are masses
fy = np.zeros(2*num_of_p_and_1)

"""x positions"""
def derivs1(px,dpxdt,vx):
    dpxdt[1] = vx[1]
    dpxdt[2] = vx[2]
    dpxdt[3] = vx[3]
    dpxdt[4] = vx[4]
    dpxdt[5] = vx[5]
    dpxdt[6] = vx[6]
"""y positions"""
def derivs2(py,dpydt,vy):
    dpydt[1] = vy[1]
    dpydt[2] = vy[2]
    dpydt[3] = vy[3]
    dpydt[4] = vy[4]
    dpydt[5] = vy[5]
    dpydt[6] = vy[6]
"""x velocities"""
def derivs3(vx,dvxdt,fx):
    dvxdt[1] = (fx[1]+fx[12])/m_p1
    dvxdt[2] = (fx[2]+fx[3])/m_p2
    dvxdt[3] = (fx[4]+fx[5])/m_p3
    dvxdt[4] = (fx[6]+fx[7])/m_p4
    dvxdt[5] = (fx[8]+fx[9])/m_p5
    dvxdt[6] = (fx[10]+fx[11])/m_p6
"""y velocities"""
def derivs4(vy,dvydt,fy):
    dvydt[1] = (fy[1]+fy[12])/m_p1
    dvydt[2] = (fy[2]+fy[3])/m_p2
    dvydt[3] = (fy[4]+fy[5])/m_p3
    dvydt[4] = (fy[6]+fy[7])/m_p4
    dvydt[5] = (fy[8]+fy[9])/m_p5
    dvydt[6] = (fy[10]+fy[11])/m_p6

"""
integrator:
this integrator employs the fourth-order runge kutta technique
"""
def dy(px,dpxdt,vx,dvxdt,fx,py,dpydt,vy,dvydt,fy):
    n = 0
    """x positions"""
    orange1 = np.zeros(num_of_p_and_1)
    orange2 = np.zeros(num_of_p_and_1)
    orange3 = np.zeros(num_of_p_and_1)
    orange4 = np.zeros(num_of_p_and_1)
    bermuda2 = np.zeros(num_of_p_and_1)
    bermuda3 = np.zeros(num_of_p_and_1)
    bermuda4 = np.zeros(num_of_p_and_1)
    derivs1(px,orange1,vx)
    for n in range(0,num_of_p_and_1):
        bermuda2[n] = px[n]+orange1[n]*(dt/2)
    derivs1(bermuda2,orange2,vx)
    for n in range(0,num_of_p_and_1):
        bermuda3[n] = px[n]+orange2[n]*(dt/2)
    derivs1(bermuda3,orange3,vx)
    for n in range(0,num_of_p_and_1):
        bermuda4[n] = px[n]+orange3[n]*dt
    derivs1(bermuda4,orange4,vx)
    for n in range(0,num_of_p_and_1):
        dpxdt[n] = (1./6.)*(orange1[n]+2*orange2[n]+2*orange3[n]+orange4[n])
    """y positions"""
    green1 = np.zeros(num_of_p_and_1)
    green2 = np.zeros(num_of_p_and_1)
    green3 = np.zeros(num_of_p_and_1)
    green4 = np.zeros(num_of_p_and_1)
    bahama2 = np.zeros(num_of_p_and_1)
    bahama3 = np.zeros(num_of_p_and_1)
    bahama4 = np.zeros(num_of_p_and_1)
    derivs2(py,green1,vy)
    for n in range(0,num_of_p_and_1):
        bahama2[n] = py[n]+green1[n]*(dt/2)
    derivs2(bahama2,green2,vy)
    for n in range(0,num_of_p_and_1):
        bahama3[n] = py[n]+green2[n]*(dt/2)
    derivs2(bahama3,green3,vy)
    for n in range(0,num_of_p_and_1):
        bahama4[n] = py[n]+green3[n]*dt
    derivs2(bahama4,green4,vy)
    for n in range(0,num_of_p_and_1):
        dpydt[n] = (1./6.)*(green1[n]+2*green2[n]+2*green3[n]+green4[n])
    """x velocities"""
    yellow1 = np.zeros(num_of_p_and_1)
    yellow2 = np.zeros(num_of_p_and_1)
    yellow3 = np.zeros(num_of_p_and_1)
    yellow4 = np.zeros(num_of_p_and_1)
    aruba2 = np.zeros(num_of_p_and_1)
    aruba3 = np.zeros(num_of_p_and_1)
    aruba4 = np.zeros(num_of_p_and_1)
    derivs3(vx,yellow1,fx)
    for n in range(0,num_of_p_and_1):
        aruba2[n] = vx[n]+yellow1[n]*(dt/2)
    derivs3(aruba2,yellow2,fx)
    for n in range(0,num_of_p_and_1):
        aruba3[n] = vx[n]+yellow2[n]*(dt/2)
    derivs3(aruba3,yellow3,fx)
    for n in range(0,num_of_p_and_1):
        aruba4[n] = vx[n]+yellow3[n]*dt
    derivs3(aruba4,yellow4,fx)
    for n in range(0,num_of_p_and_1):
        dvxdt[n] = (1./6.)*(yellow1[n]+2*yellow2[n]+2*yellow3[n]+yellow4[n])
    """y velocities"""
    red1 = np.zeros(num_of_p_and_1)
    red2 = np.zeros(num_of_p_and_1)
    red3 = np.zeros(num_of_p_and_1)
    red4 = np.zeros(num_of_p_and_1)
    jamaica2 = np.zeros(num_of_p_and_1)
    jamaica3 = np.zeros(num_of_p_and_1)
    jamaica4 = np.zeros(num_of_p_and_1)
    derivs4(vy,red1,fy)
    for n in range(0,num_of_p_and_1):
        jamaica2[n] = vy[n]+red1[n]*(dt/2)
    derivs4(jamaica2,red2,fy)
    for n in range(0,num_of_p_and_1):
        jamaica3[n] = vy[n]+red2[n]*(dt/2)
    derivs4(jamaica3,red3,fy)
    for n in range(0,num_of_p_and_1):
        jamaica4[n] = vy[n]+red3[n]*dt
    derivs4(jamaica4,red4,fy)
    for n in range(0,num_of_p_and_1):
        dvydt[n] = (1./6.)*(red1[n]+2*red2[n]+2*red3[n]+red4[n])
    """updating each position and velocity"""
    w = 0
    for w in range(1,6):
        px[w] += dpxdt[w]*dt
        py[w] += dpydt[w]*dt
        vx[w] += dvxdt[w]*dt
        vy[w] += dvydt[w]*dt

k=0
while(t<endtime):
    """spring 12""" #the spring from 1 to 2
    dx_p_12 = m.fabs(px[1]-px[2]) #distance from p1 to p2 along the x-axis
    dy_p_12 = m.fabs(py[1]-py[2])
    s12_len_0 = pow(dx_p_12**2+dy_p_12**2,1/2) #current spring length
    s12_eq_len_x = s12_eq_len*(dx_p_12/s12_len_0) #the amount of the equilibrium length parallel to the x-axis
    s12_eq_len_y = s12_eq_len*(dy_p_12/s12_len_0)
    s12_eq_x_pl = (px[1]+px[2]-s12_eq_len_x)/2
    s12_eq_x_pr = (px[1]+px[2]+s12_eq_len_x)/2
    s12_eq_y_pl = (py[1]+py[2]-s12_eq_len_y)/2
    s12_eq_y_pr = (py[1]+py[2]+s12_eq_len_y)/2
    if(m.fabs(s12_eq_x_pl - px[1]) < m.fabs(s12_eq_x_pr - px[1])):
        fx[1] = s12_fcon*(s12_eq_x_pl - px[1])
        fx[2] = s12_fcon*(s12_eq_x_pr - px[2])
    else:
        fx[1] = s12_fcon*(s12_eq_x_pr - px[1])
        fx[2] = s12_fcon*(s12_eq_x_pl - px[2])
    if(m.fabs(s12_eq_y_pl - py[1]) < m.fabs(s12_eq_y_pr - py[1])):
        fy[1] = s12_fcon*(s12_eq_y_pl - py[1])
        fy[2] = s12_fcon*(s12_eq_y_pr - py[2])
    else:
        fy[1] = s12_fcon*(s12_eq_y_pr - py[1])
        fy[2] = s12_fcon*(s12_eq_y_pl - py[2])
    """spring 23"""
    dx_p_23 = m.fabs(px[3]-px[2])
    dy_p_23 = m.fabs(py[3]-py[2])
    s23_len_0 = pow(dx_p_23**2+dy_p_23**2,1/2)
    s23_eq_len_x = s23_eq_len*(dx_p_23/s23_len_0)
    s23_eq_len_y = s23_eq_len*(dy_p_23/s23_len_0)
    s23_eq_x_pl = (px[3]+px[2]-s23_eq_len_x)/2
    s23_eq_x_pr = (px[3]+px[2]+s23_eq_len_x)/2
    s23_eq_y_pl = (py[3]+py[2]-s23_eq_len_y)/2
    s23_eq_y_pr = (py[3]+py[2]+s23_eq_len_y)/2
    if(m.fabs(s23_eq_x_pl - px[2]) < m.fabs(s23_eq_x_pr - px[2])):
        fx[3] = s23_fcon*(s23_eq_x_pl - px[2])
        fx[4] = s23_fcon*(s23_eq_x_pr - px[3])
    else:
        fx[3] = s23_fcon*(s23_eq_x_pr - px[2])
        fx[4] = s23_fcon*(s23_eq_x_pl - px[3])
    if(m.fabs(s23_eq_y_pl - py[2]) < m.fabs(s23_eq_y_pr - py[2])):
        fy[3] = s23_fcon*(s23_eq_y_pl - py[2])
        fy[4] = s23_fcon*(s23_eq_y_pr - py[3])
    else:
        fy[3] = s23_fcon*(s23_eq_y_pr - py[2])
        fy[4] = s23_fcon*(s23_eq_y_pl - py[3])
    """spring 34"""
    dx_p_34 = m.fabs(px[3]-px[4])
    dy_p_34 = m.fabs(py[3]-py[4])
    s34_len_0 = pow(dx_p_34**2+dy_p_34**2,1/2)
    s34_eq_len_x = s34_eq_len*(dx_p_34/s34_len_0)
    s34_eq_len_y = s34_eq_len*(dy_p_34/s34_len_0)
    s34_eq_x_pl = (px[3]+px[4]-s34_eq_len_x)/2
    s34_eq_x_pr = (px[3]+px[4]+s34_eq_len_x)/2
    s34_eq_y_pl = (py[3]+py[4]-s34_eq_len_y)/2
    s34_eq_y_pr = (py[3]+py[4]+s34_eq_len_y)/2
    if(m.fabs(s34_eq_x_pl - px[3]) < m.fabs(s34_eq_x_pr - px[3])):
        fx[5] = s34_fcon*(s34_eq_x_pl - px[3])
        fx[6] = s34_fcon*(s34_eq_x_pr - px[4])
    else:
        fx[5] = s34_fcon*(s34_eq_x_pr - px[3])
        fx[6] = s34_fcon*(s34_eq_x_pl - px[4])
    if(m.fabs(s34_eq_y_pl - py[3]) < m.fabs(s34_eq_y_pr - py[3])):
        fy[5] = s34_fcon*(s34_eq_y_pl - py[3])
        fy[6] = s34_fcon*(s34_eq_y_pr - py[4])
    else:
        fy[5] = s34_fcon*(s34_eq_y_pr - py[3])
        fy[6] = s34_fcon*(s34_eq_y_pl - py[4])
    """spring 45"""
    dx_p_45 = m.fabs(px[4]-px[5])
    dy_p_45 = m.fabs(py[4]-py[5])
    s45_len_0 = pow(dx_p_45**2+dy_p_45**2,1/2)
    s45_eq_len_x = s45_eq_len*(dx_p_45/s45_len_0)
    s45_eq_len_y = s45_eq_len*(dy_p_45/s45_len_0)
    s45_eq_x_pl = (px[5]+px[4]-s45_eq_len_x)/2
    s45_eq_x_pr = (px[5]+px[4]+s45_eq_len_x)/2
    s45_eq_y_pl = (py[5]+py[4]-s45_eq_len_y)/2
    s45_eq_y_pr = (py[5]+py[4]+s45_eq_len_y)/2
    if(m.fabs(s45_eq_x_pl - px[4]) < m.fabs(s45_eq_x_pr - px[4])):
        fx[7] = s45_fcon*(s45_eq_x_pl - px[4])
        fx[8] = s45_fcon*(s45_eq_x_pr - px[5])
    else:
        fx[7] = s45_fcon*(s45_eq_x_pr - px[4])
        fx[8] = s45_fcon*(s45_eq_x_pl - px[5])
    if(m.fabs(s45_eq_y_pl - py[4]) < m.fabs(s45_eq_y_pr - py[4])):
        fy[7] = s45_fcon*(s45_eq_y_pl - py[4])
        fy[8] = s45_fcon*(s45_eq_y_pr - py[5])
    else:
        fy[7] = s45_fcon*(s45_eq_y_pr - py[4])
        fy[8] = s45_fcon*(s45_eq_y_pl - py[5])
    """spring 56"""
    dx_p_56 = m.fabs(px[5]-px[6])
    dy_p_56 = m.fabs(py[5]-py[6])
    s56_len_0 = pow(dx_p_56**2+dy_p_56**2,1/2)
    s56_eq_len_x = s56_eq_len*(dx_p_56/s56_len_0)
    s56_eq_len_y = s56_eq_len*(dy_p_56/s56_len_0)
    s56_eq_x_pl = (px[5]+px[6]-s56_eq_len_x)/2
    s56_eq_x_pr = (px[5]+px[6]+s56_eq_len_x)/2
    s56_eq_y_pl = (py[5]+py[6]-s56_eq_len_y)/2
    s56_eq_y_pr = (py[5]+py[6]+s56_eq_len_y)/2
    if(m.fabs(s56_eq_x_pl - px[5]) < m.fabs(s56_eq_x_pr - px[5])):
        fx[9] = s56_fcon*(s56_eq_x_pl - px[5])
        fx[10] = s56_fcon*(s56_eq_x_pr - px[6])
    else:
        fx[9] = s56_fcon*(s56_eq_x_pr - px[5])
        fx[10] = s56_fcon*(s56_eq_x_pl - px[6])
    if(m.fabs(s56_eq_y_pl - py[5]) < m.fabs(s56_eq_y_pr - py[5])):
        fy[9] = s56_fcon*(s56_eq_y_pl - py[5])
        fy[10] = s56_fcon*(s56_eq_y_pr - py[6])
    else:
        fy[9] = s56_fcon*(s56_eq_y_pr - py[5])
        fy[10] = s56_fcon*(s56_eq_y_pl - py[6])
    """spring 61"""
    dx_p_61 = m.fabs(px[6]-px[1])
    dy_p_61 = m.fabs(py[6]-py[1])
    s61_len_0 = pow(dx_p_61**2+dy_p_61**2,1/2)
    s61_eq_len_x = s61_eq_len*(dx_p_61/s61_len_0)
    s61_eq_len_y = s61_eq_len*(dy_p_61/s61_len_0)
    s61_eq_x_pl = (px[6]+px[1]-s61_eq_len_x)/2
    s61_eq_x_pr = (px[6]+px[1]+s61_eq_len_x)/2
    s61_eq_y_pl = (py[6]+py[1]-s61_eq_len_y)/2
    s61_eq_y_pr = (py[6]+py[1]+s61_eq_len_y)/2
    if(m.fabs(s61_eq_x_pl - px[6]) < m.fabs(s61_eq_x_pr - px[6])):
        fx[11] = s61_fcon*(s61_eq_x_pl - px[6])
        fx[12] = s61_fcon*(s61_eq_x_pr - px[1])
    else:
        fx[11] = s61_fcon*(s61_eq_x_pr - px[6])
        fx[12] = s61_fcon*(s61_eq_x_pl - px[1])
    if(m.fabs(s61_eq_y_pl - py[6]) < m.fabs(s61_eq_y_pr - py[6])):
        fy[11] = s61_fcon*(s61_eq_y_pl - py[6])
        fy[12] = s61_fcon*(s61_eq_y_pr - py[1])
    else:
        fy[11] = s61_fcon*(s61_eq_y_pr - py[6])
        fy[12] = s61_fcon*(s61_eq_y_pl - py[1])
    """no broken springs"""
    """point 1"""
    dist_s23_p1 = m.fabs((py[2]-py[3])*px[1]-(px[2]-px[3])*py[1]+px[2]*py[3]-py[2]*px[3])/pow(pow(py[2]-py[3],2)+pow(px[2]-px[3],2),1/2)
    dist_s34_p1 = m.fabs((py[3]-py[4])*px[1]-(px[3]-px[4])*py[1]+px[3]*py[4]-py[3]*px[4])/pow(pow(py[3]-py[4],2)+pow(px[3]-px[4],2),1/2)
    dist_s45_p1 = m.fabs((py[4]-py[5])*px[1]-(px[4]-px[5])*py[1]+px[4]*py[5]-py[4]*px[5])/pow(pow(py[4]-py[5],2)+pow(px[4]-px[5],2),1/2)
    dist_s56_p1 = m.fabs((py[5]-py[6])*px[1]-(px[5]-px[6])*py[1]+px[5]*py[6]-py[5]*px[6])/pow(pow(py[5]-py[6],2)+pow(px[5]-px[6],2),1/2)
    if(dist_s23_p1 < 0.01):
        vx[1] *= -1
        vy[1] *= -1
        vx[2] *= -1
        vy[2] *= -1
        vx[3] *= -1
        vy[3] *= -1
    if(dist_s34_p1 < 0.01):
        vx[1] *= -1
        vy[1] *= -1
        vx[3] *= -1
        vy[3] *= -1
        vx[4] *= -1
        vy[4] *= -1
    if(dist_s45_p1 < 0.01):
        vx[1] *= -1
        vy[1] *= -1
        vx[4] *= -1
        vy[4] *= -1
        vx[5] *= -1
        vy[5] *= -1
    if(dist_s56_p1 < 0.01):
        vx[1] *= -1
        vy[1] *= -1
        vx[5] *= -1
        vy[5] *= -1
        vx[6] *= -1
        vy[6] *= -1
    """point 2"""
    dist_s34_p2 = m.fabs((py[3]-py[4])*px[2]-(px[3]-px[4])*py[2]+px[3]*py[4]-py[3]*px[4])/pow(pow(py[3]-py[4],2)+pow(px[3]-px[4],2),1/2)
    dist_s45_p2 = m.fabs((py[4]-py[5])*px[2]-(px[4]-px[5])*py[2]+px[4]*py[5]-py[4]*px[5])/pow(pow(py[4]-py[5],2)+pow(px[4]-px[5],2),1/2)
    dist_s56_p2 = m.fabs((py[5]-py[6])*px[2]-(px[5]-px[6])*py[2]+px[5]*py[6]-py[5]*px[6])/pow(pow(py[5]-py[6],2)+pow(px[5]-px[6],2),1/2)
    dist_s61_p2 = m.fabs((py[6]-py[1])*px[2]-(px[6]-px[1])*py[2]+px[6]*py[1]-py[6]*px[1])/pow(pow(py[6]-py[1],2)+pow(px[6]-px[1],2),1/2)
    if(dist_s34_p2 < 0.01):
        vx[2] *= -1
        vy[2] *= -1
        vx[3] *= -1
        vy[3] *= -1
        vx[4] *= -1
        vy[4] *= -1
    if(dist_s45_p2 < 0.01):
        vx[2] *= -1
        vy[2] *= -1
        vx[4] *= -1
        vy[4] *= -1
        vx[5] *= -1
        vy[5] *= -1
    if(dist_s56_p2 < 0.01):
        vx[2] *= -1
        vy[2] *= -1
        vx[5] *= -1
        vy[5] *= -1
        vx[6] *= -1
        vy[6] *= -1
    if(dist_s61_p2 < 0.01):
        vx[2] *= -1
        vy[2] *= -1
        vx[6] *= -1
        vy[6] *= -1
        vx[1] *= -1
        vy[1] *= -1
    """point 3"""
    dist_s45_p3 = m.fabs((py[4]-py[5])*px[3]-(px[4]-px[5])*py[3]+px[4]*py[5]-py[4]*px[5])/pow(pow(py[4]-py[5],2)+pow(px[4]-px[5],2),1/2)
    dist_s56_p3 = m.fabs((py[5]-py[6])*px[3]-(px[5]-px[6])*py[3]+px[5]*py[6]-py[5]*px[6])/pow(pow(py[5]-py[6],2)+pow(px[5]-px[6],2),1/2)
    dist_s61_p3 = m.fabs((py[6]-py[1])*px[3]-(px[6]-px[1])*py[3]+px[6]*py[1]-py[6]*px[1])/pow(pow(py[6]-py[1],2)+pow(px[6]-px[1],2),1/2)
    dist_s12_p3 = m.fabs((py[1]-py[2])*px[3]-(px[1]-px[2])*py[3]+px[1]*py[2]-py[1]*px[2])/pow(pow(py[1]-py[2],2)+pow(px[1]-px[2],2),1/2)
    if(dist_s45_p3 < 0.01):
        vx[3] *= -1
        vy[3] *= -1
        vx[4] *= -1
        vy[4] *= -1
        vx[5] *= -1
        vy[5] *= -1
    if(dist_s56_p3 < 0.01):
        vx[3] *= -1
        vy[3] *= -1
        vx[5] *= -1
        vy[5] *= -1
        vx[6] *= -1
        vy[6] *= -1
    if(dist_s61_p3 < 0.01):
        vx[3] *= -1
        vy[3] *= -1
        vx[6] *= -1
        vy[6] *= -1
        vx[1] *= -1
        vy[1] *= -1
    if(dist_s12_p3 < 0.01):
        vx[3] *= -1
        vy[3] *= -1
        vx[6] *= -1
        vy[6] *= -1
        vx[1] *= -1
        vy[1] *= -1
    """point 4"""
    dist_s56_p4 = m.fabs((py[5]-py[6])*px[4]-(px[5]-px[6])*py[4]+px[5]*py[6]-py[5]*px[6])/pow(pow(py[5]-py[6],2)+pow(px[5]-px[6],2),1/2)
    dist_s61_p4 = m.fabs((py[6]-py[1])*px[4]-(px[6]-px[1])*py[4]+px[6]*py[1]-py[6]*px[1])/pow(pow(py[6]-py[1],2)+pow(px[6]-px[1],2),1/2)
    dist_s12_p4 = m.fabs((py[1]-py[2])*px[4]-(px[1]-px[2])*py[4]+px[1]*py[2]-py[1]*px[2])/pow(pow(py[1]-py[2],2)+pow(px[1]-px[2],2),1/2)
    dist_s23_p4 = m.fabs((py[2]-py[3])*px[4]-(px[2]-px[3])*py[4]+px[2]*py[3]-py[2]*px[1])/pow(pow(py[2]-py[3],2)+pow(px[2]-px[3],2),1/2)
    if(dist_s56_p4 < 0.01):
        vx[4] *= -1
        vy[4] *= -1
        vx[5] *= -1
        vy[5] *= -1
        vx[6] *= -1
        vy[6] *= -1
    if(dist_s61_p4 < 0.01):
        vx[4] *= -1
        vy[4] *= -1
        vx[6] *= -1
        vy[6] *= -1
        vx[1] *= -1
        vy[1] *= -1
    if(dist_s12_p4 < 0.01):
        vx[4] *= -1
        vy[4] *= -1
        vx[1] *= -1
        vy[1] *= -1
        vx[2] *= -1
        vy[2] *= -1
    if(dist_s23_p4 < 0.01):
        vx[4] *= -1
        vy[4] *= -1
        vx[2] *= -1
        vy[2] *= -1
        vx[3] *= -1
        vy[3] *= -1
    """point 5"""
    dist_s61_p5 = m.fabs((py[6]-py[1])*px[5]-(px[6]-px[1])*py[5]+px[6]*py[1]-py[6]*px[1])/pow(pow(py[6]-py[1],2)+pow(px[6]-px[1],2),1/2)
    dist_s12_p5 = m.fabs((py[1]-py[2])*px[5]-(px[1]-px[2])*py[5]+px[1]*py[2]-py[1]*px[2])/pow(pow(py[1]-py[2],2)+pow(px[1]-px[2],2),1/2)
    dist_s23_p5 = m.fabs((py[2]-py[3])*px[5]-(px[2]-px[3])*py[5]+px[2]*py[3]-py[2]*px[3])/pow(pow(py[2]-py[3],2)+pow(px[2]-px[3],2),1/2)
    dist_s34_p5 = m.fabs((py[3]-py[4])*px[5]-(px[3]-px[4])*py[5]+px[3]*py[4]-py[3]*px[4])/pow(pow(py[3]-py[4],2)+pow(px[3]-px[4],2),1/2)
    if(dist_s61_p5 < 0.01):
        vx[5] *= -1
        vy[5] *= -1
        vx[6] *= -1
        vy[6] *= -1
        vx[1] *= -1
        vy[1] *= -1
    if(dist_s12_p5 < 0.01):
        vx[5] *= -1
        vy[5] *= -1
        vx[1] *= -1
        vy[1] *= -1
        vx[2] *= -1
        vy[2] *= -1
    if(dist_s23_p5 < 0.01):
        vx[5] *= -1
        vy[5] *= -1
        vx[2] *= -1
        vy[2] *= -1
        vx[3] *= -1
        vy[3] *= -1
    if(dist_s34_p5 < 0.01):
        vx[5] *= -1
        vy[5] *= -1
        vx[3] *= -1
        vy[3] *= -1
        vx[4] *= -1
        vy[4] *= -1
    """point 6"""
    dist_s12_p6 = m.fabs((py[1]-py[2])*px[6]-(px[1]-px[2])*py[6]+px[1]*py[2]-py[1]*px[2])/pow(pow(py[1]-py[2],2)+pow(px[1]-px[2],2),1/2)
    dist_s23_p6 = m.fabs((py[2]-py[3])*px[6]-(px[2]-px[3])*py[6]+px[2]*py[3]-py[2]*px[3])/pow(pow(py[2]-py[3],2)+pow(px[2]-px[3],2),1/2)
    dist_s34_p6 = m.fabs((py[3]-py[4])*px[6]-(px[3]-px[4])*py[6]+px[3]*py[4]-py[3]*px[4])/pow(pow(py[3]-py[4],2)+pow(px[3]-px[4],2),1/2)
    dist_s45_p6 = m.fabs((py[4]-py[5])*px[6]-(px[4]-px[5])*py[6]+px[4]*py[5]-py[4]*px[5])/pow(pow(py[4]-py[5],2)+pow(px[4]-px[5],2),1/2)
    if(dist_s12_p6 < 0.01):
        vx[6] *= -1
        vy[6] *= -1
        vx[1] *= -1
        vy[1] *= -1
        vx[2] *= -1
        vy[2] *= -1
    if(dist_s23_p6 < 0.01):
        vx[6] *= -1
        vy[6] *= -1
        vx[2] *= -1
        vy[2] *= -1
        vx[3] *= -1
        vy[3] *= -1
    if(dist_s34_p6 < 0.01):
        vx[6] *= -1
        vy[6] *= -1
        vx[3] *= -1
        vy[3] *= -1
        vx[4] *= -1
        vy[4] *= -1
    if(dist_s45_p6 < 0.01):
        vx[6] *= -1
        vy[6] *= -1
        vx[4] *= -1
        vy[4] *= -1
        vx[5] *= -1
        vy[5] *= -1
    """function call for differentiating"""
    dy(px,dpxdt,vx,dvxdt,fx,py,dpydt,vy,dvydt,fy)
    #print("%d. f1: %.2f f2: %.2f len12: %.2f len23: %.2f len34: %.2f dx: %.2f dy: %.2f  v1: %.2f v2: %.2f acc_v1: %.2f acc_v2: %.2f"%(k,fx[1],fx[2],s12_len_0,s23_len_0,s34_len_0,dx_p_12,dy_p_12,dpxdt[1],dpxdt[2],dvxdt[1],dvxdt[2]))
    """calculating the energy in the system"""
    energy_p1 = 1/2*m_p1*(pow(vx[1],2)+pow(vy[1],2))
    energy_p2 = 1/2*m_p2*(pow(vx[2],2)+pow(vy[2],2))
    energy_p3 = 1/2*m_p3*(pow(vx[3],2)+pow(vy[3],2))
    energy_p4 = 1/2*m_p4*(pow(vx[4],2)+pow(vy[4],2))
    energy_p5 = 1/2*m_p1*(pow(vx[5],2)+pow(vy[5],2))
    energy_p6 = 1/2*m_p1*(pow(vx[6],2)+pow(vy[6],2))
    total_energy = energy_p1 + energy_p2 + energy_p3 + energy_p4 + energy_p5 + energy_p6
    energy_in_the_system[k] = total_energy
    """storing the x and y positions of each coordinate (for the animation)"""
    p1x[k] = px[1]
    p2x[k] = px[2]
    p3x[k] = px[3]
    p4x[k] = px[4]
    p5x[k] = px[5]
    p6x[k] = px[6]
    p1y[k] = py[1]
    p2y[k] = py[2]
    p3y[k] = py[3]
    p4y[k] = py[4]
    p5y[k] = py[5]
    p6y[k] = py[6]
    k += 1
    t += dt
"""
Making the animation
"""
fig, ax = plt.subplots()
x1data, x2data, x3data, x4data, x5data, x6data, y1data, y2data, y3data, y4data, y5data, y6data = p1x, p2x, p3x, p4x, p5x, p6x, p1y, p2y, p3y, p4y, p5y, p6y
ln, = plt.plot(p1x, p1y, animated=True)
def init():
    ax.set_xlim(-5, 5) #the width of each frame in our animation
    ax.set_ylim(-5, 5) #the height of each frame in our animation
    return ln,
def animate(frame):
    ln.set_data([x1data[frame],x2data[frame],x3data[frame],x4data[frame],x5data[frame],x6data[frame],x1data[frame]],[y1data[frame],y2data[frame],y3data[frame],y4data[frame],y5data[frame],y6data[frame],y1data[frame]])
    print(energy_in_the_system[frame])
    return ln,
ani = animation.FuncAnimation(fig, animate, frames=num_of_frames,init_func=init, blit=True)
plt.show()
"""
going forward there are three major upgrades I want to make to this code. I want to find a Python library with better animation tools for this task (e.g. I want to adjust the fps rate of the animation). And I want to include another data structure corresponding to a "thermal pool" which will continually provide momentum to the system - of random magnitude and direction (within some interval)
"""
