## EN.530.603 Applied Optimal Control
# Final Project

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from functools import partial
from trajopt_sqp import trajopt_sqp

# import userinput packages
import tkinter as tk
from tkinter import simpledialog
from tkinter import messagebox

## -- Min Distance b/w Vehicle and Rectangular Obstacles
# Checking if a point is in line shadow
def point_in_line_shadow(p1: np.ndarray, p2: np.ndarray, q: np.ndarray) -> bool:
    segment_length = np.linalg.norm(p2 - p1)
    segment_dir = (p2 - p1) / segment_length
    projection = np.dot(q - p1, segment_dir)

    return 0 < projection < segment_length

# Getting the shortest distance
def get_min_distance_to_segment(p1: np.ndarray, p2: np.ndarray, q: np.ndarray) -> float:
    return np.linalg.norm(np.cross(
        (p2 - p1) / np.linalg.norm(p2 - p1),
        q - p1
    )) if point_in_line_shadow(p1, p2, q) else min(
        np.linalg.norm(q - p1), np.linalg.norm(q - p2)
    )

# Doing this for all points with all sides
def get_rectangle_sides(vertices: list) -> list:
    return list(map(
        lambda i: (vertices[i], vertices[(i + 1) % 4]),
        range(4)
    ))

# calculate the minimum distance from the point to all sides of the other rectangle
def get_min_distance_point_rectangle(rect_sides: list, q: np.ndarray) -> float:
    return min(map(
        lambda side: get_min_distance_to_segment(*side, q),
        rect_sides
    ))

# Altogether
def get_min_distance_rectangles(r1: list, r2: list) -> float:
    r1 = list(map(np.asarray, r1))
    r2 = list(map(np.asarray, r2))

    min_r1_to_r2 = min(map(
        partial(
            get_min_distance_point_rectangle,
            get_rectangle_sides(r2)
        ),
        r1
    ))

    min_r2_to_r1 = min(map(
        partial(
            get_min_distance_point_rectangle,
            get_rectangle_sides(r1)
        ),
        r2
    ))

    return min(min_r1_to_r2, min_r2_to_r1)



class Direct_Collocation:

  def __init__(self, parking_slot, x0, xf, os_c):

    # time horizon and segments
    self.parking_slot = parking_slot
    self.tf = 20.0
    self.N = 8
    self.dt = self.tf / self.N
    self.ylim = 23
    self.xlim = 15
    # cost function parameters
    self.Q = np.diag([0, 0, 0, 0, 0])
    self.R = np.diag([1, 5])
    self.Qf = np.diag([5, 5, 100, 1, 1])



    ## Modified: add parking vehicle obstacles
    # demonstration of 2 obstacles
    # self.os_p = np.array([[3.0, 4.0],[3.0, 14.0]])
    # self.os_r = np.array([[.5],[.5]])
    self.os_w = 2.0  # Width in x
    self.os_l = 4.0  # Length in y
    self.os_c = os_c
    self.Nobst = len(self.os_c)
    self.Nwall = 4

        # initial state
    self.x0 = x0

    self.xf = xf
    # self.wall_con = np.array
    # Corners Coordinates
    self.obsts = np.zeros((4, 2, self.Nobst))
    for i in range(self.Nobst):
      self.obsts[:,:,i] = np.array([[self.os_c[i, 0] - self.os_w/2, self.os_c[i, 1] - self.os_l/2],
                                    [self.os_c[i, 0] + self.os_w/2, self.os_c[i, 1] - self.os_l/2],
                                    [self.os_c[i, 0] + self.os_w/2, self.os_c[i, 1] + self.os_l/2],
                                    [self.os_c[i, 0] - self.os_w/2, self.os_c[i, 1] + self.os_l/2]])
    
    # # Car
    # heading = self.x0[2] + self.dt * self.x0[4] # x[2] + dt * w
    # self.carcoord = np.array([[self.x0[0] - (self.os_w/2)*np.cos(heading) + (self.os_l/2)*np.sin(heading), 
    #                            self.x0[1] - (self.os_w/2)*np.sin(heading) - (self.os_l/2)*np.cos(heading)],
    #                           [self.x0[0] + (self.os_w/2)*np.cos(heading) + (self.os_l/2)*np.sin(heading), 
    #                            self.x0[1] + (self.os_w/2)*np.sin(heading) - (self.os_l/2)*np.cos(heading)],
    #                           [self.x0[0] + (self.os_w/2)*np.cos(heading) - (self.os_l/2)*np.sin(heading), 
    #                            self.x0[1] + (self.os_w/2)*np.sin(heading) + (self.os_l/2)*np.cos(heading)],
    #                           [self.x0[0] - (self.os_w/2)*np.cos(heading) - (self.os_l/2)*np.sin(heading), 
    #                            self.x0[1] - (self.os_w/2)*np.sin(heading) + (self.os_l/2)*np.cos(heading)]])

  def f(self, k, x, u):
    # car dynamics and jacobians

    dt = self.dt
    c = np.cos(x[2])
    s = np.sin(x[2])
    v = x[3]
    w = x[4]

    A = np.array([[1, 0, -dt * s * v, dt * c, 0],
                  [0, 1,  dt * c * v, dt * s, 0],
                  [0, 0,  1,          0,    dt],
                  [0, 0,  0,          1,    0],
                  [0, 0,  0,          0,    1]])

    B = np.array([[0, 0],
                  [0, 0],
                  [0, 0],
                  [dt, 0],
                  [0, dt]])

    x = np.array([x[0] + dt * c * v, 
                  x[1] + dt * s * v, 
                  x[2] + dt * w, 
                  v + dt * u[0],
                  w + dt * u[1]])

    return x, A, B

  def L(self, k, x, u):
    # car cost (just standard quadratic cost)
    L = self.dt * 0.5 * (np.transpose(x-self.xf) @ self.Q @ (x-self.xf) +
                        np.transpose(u) @ self.R @ u)
    Lx = self.dt * self.Q @ (x- self.xf)
    Lxx = self.dt * self.Q
    Lu = self.dt * self.R @ u
    Luu = self.dt * self.R

    return L, Lx, Lxx, Lu, Luu

  def Lf(self, x):
    # car final cost (just standard quadratic cost)
    L = np.transpose(x-self.xf) @ self.Qf @ (x-self.xf) * 0.5
    Lx = self.Qf @ (x-self.xf)
    Lxx = self.Qf

    return L, Lx, Lxx

  ## Modified: constraints -- similarly to HW5 P3
  def con(self, k, x, u):

    # Car
    heading = x[2] + self.dt * x[4] # x[2] + dt * w
    c = np.cos(heading)
    s = np.sin(heading)
    self.carcoord = np.array([[x[0] - (self.os_w/2)*c + (self.os_l/2)*s, 
                          x[1] - (self.os_w/2)*s - (self.os_l/2)*c],
                         [x[0] + (self.os_w/2)*c + (self.os_l/2)*s, 
                          x[1] + (self.os_w/2)*s - (self.os_l/2)*c],
                         [x[0] + (self.os_w/2)*c - (self.os_l/2)*s, 
                          x[1] + (self.os_w/2)*s + (self.os_l/2)*c],
                         [x[0] - (self.os_w/2)*c - (self.os_l/2)*s, 
                          x[1] - (self.os_w/2)*s + (self.os_l/2)*c]])

    cs = np.zeros(self.Nobst+self.Nwall)
    for i in [x for x in range(self.Nobst) if x != self.parking_slot]:
      minDist = get_min_distance_rectangles(self.carcoord, self.obsts[:,:,i])
      cs[i] = -minDist

    cs[self.Nobst] = -min(self.carcoord[0,:]-0)
    cs[self.Nobst+1] = -min(self.xlim-self.carcoord[0,:])
    cs[self.Nobst+2] = -min(self.carcoord[1,:]-0)
    cs[self.Nobst+3] = -min(self.ylim-self.carcoord[1,:])


    return cs

  ## Modified: constraints -- similarly to HW5 P3
  def conf(self, k, x):
    # (radius of disk obstacles)^2 - dist(x,center of disk obstacles)^2
    # i.e.: os_r^2 - dist(x,os_p)^2

    cs = np.zeros(self.Nobst)
    for i in range(self.Nobst):
      cs[i] = self.os_r[i]**2 - ((x[0] - self.os_p[i][0])**2 + (x[1] - self.os_p[i][1])**2)
    
    return cs

  def traj(self, us):

    N = us.shape[1]

    xs = np.zeros((5, N + 1))
    xs[:, 0] = self.x0
    for k in range(N):
      xs[:, k + 1], _, _ = self.f(k, xs[:, k], us[:, k])

    return xs

  def plot_traj(self, xs, us):

    # plot state trajectory
    # self.axs[0].plot(xs[0, :], xs[1, :], '-b')
    # self.axs[0].axis('equal')
    # self.axs[0].set_xlabel('x')
    # self.axs[0].set_ylabel('y')
    self.axs_1.plot(xs[0, :], xs[1, :], '-b', linewidth=1)
    self.axs_1.axis('equal')
    self.axs_1.set_xlabel('x')
    self.axs_1.set_ylabel('y')

    # plot control trajectory
    self.axs[0].lines.clear()
    self.axs[1].lines.clear()
    self.axs[0].grid()
    self.axs[1].grid()
    self.axs[0].plot(np.arange(0, self.tf, self.dt), us[0, :], '-b')
    self.axs[0].relim()
    self.axs[0].autoscale_view()
    self.axs[1].plot(np.arange(0, self.tf, self.dt), us[1, :], '-r')
    self.axs[1].relim()
    self.axs[1].autoscale_view()

    # drawing updated values
    fig.canvas.draw()

    # This will run the GUI event
    # loop until all UI events
    # currently waiting have been processed
    fig.canvas.flush_events()



if __name__ == '__main__':
  """modified"""
  waypoint = np.array([[6.0, 15.0],[9.0,15.0],
                       [6.0, 10.0],[9.0, 10.0],
                       [6.0, 13.0],[9.0, 13.0],
                       [6.0, 8.0],[9.0, 8.0]]) # center

  os_c = np.array([[3.0, 19.0],[12.0,19.0],
                   [3.0, 14.0],[12.0, 14.0],
                   [3.0, 9.0],[12.0, 9.0],
                   [3.0, 4.0],[12.0, 4.0]]) # center

  ROOT = tk.Tk()
  ROOT.withdraw()
  USER_INP = simpledialog.askstring(title="Autonomous Parallel Parking",
                                  prompt="Parking Slot Selection (0~7):")
  
  """modified"""
  # 1st stage
  if(int(USER_INP) == 4 or int(USER_INP) == 5 or int(USER_INP) == 6 or int(USER_INP) == 7):

    heading = np.pi/2
    x0_1 = np.array([7.5, 2.5, heading, 0, 0])
    xf_1 = np.array([waypoint[int(USER_INP)][0], waypoint[int(USER_INP)][1], heading, 0, 0])
    xf_2 = np.array([os_c[int(USER_INP)][0], os_c[int(USER_INP)][1], heading, 0, 0])


  elif(int(USER_INP) == 0 or int(USER_INP) == 1 or int(USER_INP) == 2 or int(USER_INP) == 3):

    heading = -np.pi/2
    x0_1 = np.array([7.5, 20.5, heading, 0, 0])
    xf_1 = np.array([waypoint[int(USER_INP)][0], waypoint[int(USER_INP)][1], heading, 0, 0])
    xf_2 = np.array([os_c[int(USER_INP)][0], os_c[int(USER_INP)][1], heading, 0, 0])
    

  else:
    tk.messagebox.showwarning(title="Autonomous Parallel Parking",message="Invalid Parking Slot")
  
  prob = Direct_Collocation(int(USER_INP), x0_1, xf_1, os_c)
  prob_parallel = Direct_Collocation(int(USER_INP), xf_1, xf_2, os_c)
  
  # initial control sequence
  us = np.concatenate((np.ones((2, prob.N // 2)) * 0.02,
                       np.ones((2, prob.N // 2)) * -0.02), axis=1)

  xs = prob.traj(us)

  us_1 = np.concatenate((np.ones((2, prob_parallel.N // 2)) * 0.02,
                       np.ones((2, prob_parallel.N // 2)) * -0.02), axis=1)

  xs_1 = prob_parallel.traj(us_1)

  plt.ion()
  plt.rcParams["figure.figsize"] = [15, 7.5]
  plt.rcParams["figure.autolayout"] = True
  fig, axs = plt.subplots(1, 2)
  fig.set_facecolor("slategray")
  axs[0].set_facecolor("lightgray")
  axs[1].set_facecolor("lightgray")
  
  prob.axs = axs
  plt.suptitle(r'$Direct\ Collocation$',fontsize=16)
  axs[1].legend([r'$u_1:\ acceleration$', r'$u_2:\ steering\ angle\ rate$'])
  axs[1].set_xlabel(r'$sec.$')
  # plot vehicles (obstacles)
  plt.rcParams["figure.figsize"] = [5, 23/3]
  plt.rcParams["figure.autolayout"] = True

  fig1, axs1 = plt.subplots(1, 1)
  fig1.set_facecolor("slategray")
  axs1.set_facecolor("lightgray")

  prob.fig = fig1
  prob.axs_1 = axs1
  for j in range(prob.Nobst):
    # rect = plt.Rectangle((prob.os_p[j][0]-prob.os_w/2, prob.os_p[j][1]-prob.os_l/2), 
    #                       prob.os_w, prob.os_l, color='cyan')
    if (j == prob.parking_slot):
      rect = plt.Rectangle((prob.obsts[:,:,j][0,0], prob.obsts[:,:,j][0,1]), prob.os_w, prob.os_l, color='black', linewidth=2, fill=False)

    else:
      rect = plt.Rectangle((prob.obsts[:,:,j][0,0], prob.obsts[:,:,j][0,1]), prob.os_w, prob.os_l, color='black')
    # circle = plt.Circle(prob.os_p[j], prob.os_r[j], color='r', linewidth=2, fill=False)
    axs1.add_patch(rect)
    # axs[0].add_patch(circle)
    # Rectangle((prob.os_p[j][0]-prob.os_w/2, prob.os_p[j][1]-prob.os_l/2), prob.os_w, prob.os_l)
  axs1.axis('equal')
  axs1.set_xlabel(r'$x$')
  axs1.set_ylabel(r'$y$')

  # plot initial trajectory
  prob.plot_traj(xs, us)
  xs, us, cost = trajopt_sqp(xs, us, prob)

  # ??????
  # carshapes = np.zeros((5, 2, prob.N+1))
  # for i in range(prob.N+1):
  #   heading = xs[2,i] + prob.dt * xs[4,i] # x[2] + dt * w
  #   c = np.cos(heading)
  #   s = np.sin(heading)
  #   carcoord = np.array([[xs[0,i] - (prob.os_w/2)*c + (prob.os_l/2)*s, 
  #                         xs[1,i] - (prob.os_w/2)*s - (prob.os_l/2)*c],
  #                        [xs[0,i] + (prob.os_w/2)*c + (prob.os_l/2)*s, 
  #                         xs[1,i] + (prob.os_w/2)*s - (prob.os_l/2)*c],
  #                        [xs[0,i] + (prob.os_w/2)*c - (prob.os_l/2)*s, 
  #                         xs[1,i] + (prob.os_w/2)*s + (prob.os_l/2)*c],
  #                        [xs[0,i] - (prob.os_w/2)*c - (prob.os_l/2)*s, 
  #                         xs[1,i] - (prob.os_w/2)*s + (prob.os_l/2)*c]])
  #   carshapes[:,:,i] = np.vstack([carcoord, carcoord[0]])
  # print(carshapes.shape, carshapes)
  # print(carshapes[:,:,0])
  

  plt.ioff()
  prob.plot_traj(xs, us)
  axs1.set_xlabel(r'$x$')
  axs1.set_ylabel(r'$y$')

  axs1.set_aspect('auto')
  axs1.set_xlim([0, 15])
  # axs1.set_xbound(lower=0.0, upper=15.0)
  axs1.set_ylim([0, 23])
  # axs1.grid()
  # axs1.set_ybound(lower=0.0, upper=23.0)
  for i in range(xs.shape[1]):
    axs1.add_patch(Rectangle((0, 0),
                      prob.xlim, prob.ylim,
                      fc='none',
                      color ='midnightblue',
                      lw = 7) )
    axs1.plot(xs[0, 0:i+1], xs[1, 0:i+1], color="lime", linewidth=3)
    plt.pause(prob.dt)
  # axs1.axis('equal')
  print(xs[:,-1])

  plt.show()