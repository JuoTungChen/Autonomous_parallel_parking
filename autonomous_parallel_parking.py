"""
EN.530.603 Applied Optimal Control
Final Project - Autonomous Parallel Parking
Team members: Juo-Tung Chen, Iou-Sheng Chang
"""

# ----------importing Packages---------------
# general
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from functools import partial
from trajopt_sqp import trajopt_sqp
import cv2

# userinput packages
import tkinter as tk
from tkinter import simpledialog
from tkinter import messagebox

np.set_printoptions(suppress=True)    # suppress scientific notations in the text on the plot

## ------Min Distance b/w Vehicle and Rectangular Obstacles---------
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


# ------- class for implementing Direct Collocation ----------

class Direct_Collocation:

  def __init__(self, parking_slot, x0, xf, tf, os_c, bound_xlim, bound_ylim, part):

    # time horizon and segments
    self.parking_slot = parking_slot
    self.tf = int(tf)
    self.N = 32
    self.dt = self.tf / self.N

    # cost function parameters
    self.Q = np.diag([0, 0, 0, 0, 0])
    self.R = np.diag([1, 5])
    self.Qf = np.diag([5, 5, 30, 10, 3])     

    # add constraints (vehicle obstacles and walls)
    self.os_w = 2.0               # Width in x
    self.os_l = 4.0               # Length in y
    self.os_c = os_c
    self.Nobst = len(self.os_c)   # NUM obstacle (vehicles)
    self.Nwall = 4                # NUM walls
    self.Ncontrol = 4             # NUM control constraints
    self.u_bound = np.array([[-1.5, 1.5], [-0.5, 0.5]])     # boundaries for control inputs
    self.buffer = 0.25            # buffer distance between the obstacles and the vehicle
    self.xlim = bound_xlim        # boundary for x
    self.ylim = bound_ylim        # boundary for y

    self.x0 = x0                  # initial state
    self.xf = xf                  # derised final parking state
    self.part = part              # which trajectory (1 or 2)
    self.iteration = 0            # count the numbers of iterations 
    self.min_dist = 100           # store the min distance between the obstacles and the vehicle throughout the trajectory

    # vehicle obstacles corners coordinates
    self.obsts = np.zeros((4, 2, self.Nobst))
    for i in range(self.Nobst):
      self.obsts[:,:,i] = np.array([[self.os_c[i, 0] - self.os_w/2, self.os_c[i, 1] - self.os_l/2],
                                    [self.os_c[i, 0] + self.os_w/2, self.os_c[i, 1] - self.os_l/2],
                                    [self.os_c[i, 0] + self.os_w/2, self.os_c[i, 1] + self.os_l/2],
                                    [self.os_c[i, 0] - self.os_w/2, self.os_c[i, 1] + self.os_l/2]])
    

  def f(self, k, x, u):
    # car dynamics and jacobians

    dt = self.dt
    c = np.cos(x[2])
    s = np.sin(x[2])
    v = x[3]
    w = x[4]

    A = np.array([[1, 0, -dt * s * v, dt * c,  0],
                  [0, 1,  dt * c * v, dt * s,  0],
                  [0, 0,           1,      0, dt],
                  [0, 0,           0,      1,  0],
                  [0, 0,           0,      0,  1]])

    B = np.array([[ 0,  0],
                  [ 0,  0],
                  [ 0,  0],
                  [dt,  0],
                  [ 0, dt]])

    x = np.array([x[0] + dt * c * v, 
                  x[1] + dt * s * v, 
                  x[2] + dt * w, 
                  v + dt * u[0],
                  w + dt * u[1]])

    return x, A, B

  def L(self, k, x, u):
    # car cost
    # x - xf to take desired final vehicel state into consideration
    L = self.dt * 0.5 * (np.transpose(x-self.xf) @ self.Q @ (x-self.xf) +
                         np.transpose(u) @ self.R @ u)
    Lx = self.dt * self.Q @ (x - self.xf)
    Lxx = self.dt * self.Q
    Lu = self.dt * self.R @ u
    Luu = self.dt * self.R

    return L, Lx, Lxx, Lu, Luu
    

  def Lf(self, x):
    # car final cost (just standard quadratic cost)
    # x - xf to take desired final vehicel state into consideration
    L = np.transpose(x-self.xf) @ self.Qf @ (x-self.xf) * 0.5
    Lx = self.Qf @ (x-self.xf)
    Lxx = self.Qf

    return L, Lx, Lxx

  def calculate_ineq(self, u, bound, UorL):
    if UorL > 0:
      return [max(u - bound, 0), 1]        # c and u_grad
    elif UorL < 0:
      return [max(bound - u, 0), -1]       # c and u_grad


  def con(self, k, x, u):
    # Constraints
    # Parked vehicles constraints
    heading = x[2] + self.dt * x[4] # x[2] + dt * w
    c = np.cos(heading)
    s = np.sin(heading)
    self.carcoord = np.array([[x[0] - (self.os_l/2)*c + (self.os_w/2)*s, 
                            x[1] - (self.os_l/2)*s - (self.os_w/2)*c],
                            [x[0] + (self.os_l/2)*c + (self.os_w/2)*s, 
                            x[1] + (self.os_l/2)*s - (self.os_w/2)*c],
                            [x[0] + (self.os_l/2)*c - (self.os_w/2)*s, 
                            x[1] + (self.os_l/2)*s + (self.os_w/2)*c],
                            [x[0] - (self.os_l/2)*c - (self.os_w/2)*s, 
                            x[1] - (self.os_l/2)*s + (self.os_w/2)*c]])

    cs = np.zeros(self.Nobst+self.Nwall)
    for i in [x for x in range(self.Nobst) if x != self.parking_slot]:
      minDist = get_min_distance_rectangles(self.carcoord, self.obsts[:,:,i])
      self.min_dist =  min(self.min_dist, minDist)        # store the min distance throughout the whole trajectory
      cs[i] = -minDist + self.buffer

    # Walls constraints
    cs[self.Nobst]   = -min(self.carcoord[0,:]-0)
    cs[self.Nobst+1] = -min(self.xlim-self.carcoord[0,:])
    cs[self.Nobst+2] = -min(self.carcoord[1,:]-0)
    cs[self.Nobst+3] = -min(self.ylim-self.carcoord[1,:])

    return cs

  def conf(self, k, x):
    # Constraints

    # Parked vehicles constraints
    heading = x[2] + self.dt * x[4] # x[2] + dt * w
    c = np.cos(heading)
    s = np.sin(heading)
    self.carcoord = np.array([[x[0] - (self.os_l/2)*c + (self.os_w/2)*s, 
                            x[1] - (self.os_l/2)*s - (self.os_w/2)*c],
                            [x[0] + (self.os_l/2)*c + (self.os_w/2)*s, 
                            x[1] + (self.os_l/2)*s - (self.os_w/2)*c],
                            [x[0] + (self.os_l/2)*c - (self.os_w/2)*s, 
                            x[1] + (self.os_l/2)*s + (self.os_w/2)*c],
                            [x[0] - (self.os_l/2)*c - (self.os_w/2)*s, 
                            x[1] - (self.os_l/2)*s + (self.os_w/2)*c]])

    cs = np.zeros(self.Nobst+self.Nwall+self.Ncontrol)
    for i in [x for x in range(self.Nobst) if x != self.parking_slot]:
      minDist = get_min_distance_rectangles(self.carcoord, self.obsts[:,:,i])
      self.min_dist =  min(self.min_dist, minDist)          # store the min distance throughout the whole trajectory
      cs[i] = -minDist

    # Walls constraints
    cs[self.Nobst]   = -min(self.carcoord[0,:]-0)
    cs[self.Nobst+1] = -min(self.xlim-self.carcoord[0,:])
    cs[self.Nobst+2] = -min(self.carcoord[1,:]-0)
    cs[self.Nobst+3] = -min(self.ylim-self.carcoord[1,:])

    return cs

  def traj(self, us):
    # calculate the state trajectory from the control sequence
    N = us.shape[1]
    xs = np.zeros((5, N + 1))
    xs[:, 0] = self.x0
    for k in range(N):
      xs[:, k + 1], _, _ = self.f(k, xs[:, k], us[:, k])

    return xs

  def plot_traj(self, xs, us):
 
    # plot 1st trajectory
    if (self.part == 1):
      if (self.iteration == 0):
        print("\n1st part of the trajectory:")
      print("iteration (1st) = ", self.iteration)
      self.iteration = self.iteration + 1

      self.axs2.plot(xs[0, :], xs[1, :], '-', c='b', linewidth=1)
      self.axs2.set_xlabel('x')
      self.axs2.set_ylabel('y')
      self.axs2.set_xlim([0, bound_xlim])
      self.axs2.set_ylim([0, bound_ylim])

      # u_1 = acceleration
      self.axs1[0][0].lines.clear()
      self.axs1[0][0].plot(np.arange(0, self.tf, self.dt), us[0, :], '-b')
      self.axs1[0][0].relim()
      self.axs1[0][0].autoscale_view()

      # u_2 = steering angle rate
      self.axs1[0][1].lines.clear()
      self.axs1[0][1].plot(np.arange(0, self.tf, self.dt), us[1, :], '-r')
      self.axs1[0][1].relim()
      self.axs1[0][1].autoscale_view()

      # xs[3] = velocity
      self.axs3[0][0].lines.clear()
      self.axs3[0][0].plot(np.arange(0, self.tf+self.dt, self.dt), xs[3, :], '-b')
      self.axs3[0][0].relim()
      self.axs3[0][0].autoscale_view()

      # xs[4] = steering angle
      self.axs3[0][1].lines.clear() 
      self.axs3[0][1].plot(np.arange(0, self.tf+self.dt, self.dt), xs[4, :], '-r')
      self.axs3[0][1].relim()
      self.axs3[0][1].autoscale_view()



    # plot 2nd trajectory
    if (self.part == 2):
      if (self.iteration == 0):
        print("\n2nd part of the trajectory:")
      print("iteration (2nd) = ", self.iteration)
      self.iteration = self.iteration + 1
      self.axs2.plot(xs[0, :], xs[1, :], '-', c='royalblue', linewidth=1)
      self.axs2.set_xlabel('x')
      self.axs2.set_ylabel('y')
      self.axs2.set_xlim([0, bound_xlim])
      self.axs2.set_ylim([0, bound_ylim])

      self.axs1[1][0].lines.clear()
      self.axs1[1][0].plot(np.arange(0, self.tf, self.dt), us[0, :], '-b')
      self.axs1[1][0].relim()
      self.axs1[1][0].autoscale_view()

      self.axs1[1][1].lines.clear()
      self.axs1[1][1].plot(np.arange(0, self.tf, self.dt), us[1, :], '-r')
      self.axs1[1][1].relim()
      self.axs1[1][1].autoscale_view()

      # xs[3] = velocity
      self.axs3[1][0].lines.clear()
      self.axs3[1][0].plot(np.arange(0, self.tf+self.dt, self.dt), xs[3, :], '-b')
      self.axs3[1][0].relim()
      self.axs3[1][0].autoscale_view()

      # xs[4] = steering angle
      self.axs3[1][1].lines.clear()
      self.axs3[1][1].plot(np.arange(0, self.tf+self.dt, self.dt), xs[4, :], '-r')
      self.axs3[1][1].relim()
      self.axs3[1][1].autoscale_view()

    # drawing updated values
    fig2.canvas.draw()

    # This will run the GUI event
    # loop until all UI events
    # currently waiting have been processed
    fig2.canvas.flush_events()


# ------- class for drawing the vehicle ----------
class Draw_vehicle:
  def __init__(self) -> None:
    self.margin = 0.05
    self.car_length = 4
    self.car_width = 2
    self.wheel_length = 0.75
    self.wheel_width = 7/20
    self.wheel_positions = self.margin * np.array([[25,15],[25,-15],[-25,15],[-25,-15]])
    self.part = 0
    
    self.car_color = "#069AF3"
    self.wheel_color = 'k'

    self.car_struct = np.array([[+self.car_length/2, +self.car_width/2],
                                [+self.car_length/2, -self.car_width/2],  
                                [-self.car_length/2, -self.car_width/2],
                                [-self.car_length/2, +self.car_width/2]], 
                                )
    
    self.wheel_struct = np.array([[+self.wheel_length/2, +self.wheel_width/2],
                                [+self.wheel_length/2, -self.wheel_width/2],  
                                [-self.wheel_length/2, -self.wheel_width/2],
                                [-self.wheel_length/2, +self.wheel_width/2]]
                                )

  def rotate_car(self, pts, angle=0):
    R = np.array([[np.cos(angle), -np.sin(angle)],
                [np.sin(angle),  np.cos(angle)]])
    return ((R @ pts.T).T)

  def render(self, x, y, psi, delta, plot):

    # adding car body
    rotated_struct = self.rotate_car(self.car_struct, angle=psi)
    rotated_struct += np.array([x,y])
    plot.fill(rotated_struct[:,0], rotated_struct[:,1], color=self.car_color)
    rotated_wheel_center = self.rotate_car(self.wheel_positions, angle=psi)

    for i, wheel in enumerate(rotated_wheel_center):
        # rotate front wheels (steering angle + heading)
        if i < 2:
          if self.part == 1:
              rotated_wheel = self.rotate_car(self.wheel_struct, angle=delta+psi)
          if self.part == 2:
              rotated_wheel = self.rotate_car(self.wheel_struct, angle=-delta+psi)
        # rotate back wheels (heading)
        else:
            rotated_wheel = self.rotate_car(self.wheel_struct, angle=psi)
        rotated_wheel += np.array([x,y]) + wheel 
        plot.fill(rotated_wheel[:,0], rotated_wheel[:,1], color=self.wheel_color)


# --------- main function -----------
if __name__ == '__main__':

  # Parking Slots (center)
  car_distance = 1.5
  left_distance = 3
  right_distance = 3
  top_distance = 5
  bottom_distance = 3
  bound_xlim = 15
  bound_ylim = 4*4 + 3*car_distance + top_distance + bottom_distance

  os_c = np.array([[left_distance, bound_ylim-top_distance-2], [bound_xlim-right_distance, bound_ylim-top_distance-2],
                   [left_distance, bound_ylim-top_distance-2-car_distance-4], [bound_xlim-right_distance, bound_ylim-top_distance-2-car_distance-4],
                   [left_distance,  bound_ylim-top_distance-2-2*(car_distance+4)], [bound_xlim-right_distance,  bound_ylim-top_distance-2-2*(car_distance+4)],
                   [left_distance,  bound_ylim-top_distance-2-3*(car_distance+4)], [bound_xlim-right_distance,  bound_ylim-top_distance-2-3*(car_distance+4)]])


  # 2 stage intermediate goal parameters
  x_offset = 3
  y_offset = 4
  waitpoints = np.array([[left_distance + x_offset, os_c[0][1]+y_offset],[bound_xlim-right_distance-x_offset, os_c[0][1]+y_offset],
                       [left_distance + x_offset, os_c[2][1]+y_offset],[bound_xlim-right_distance-x_offset, os_c[2][1]+y_offset],
                       [left_distance + x_offset, os_c[4][1]+y_offset],[bound_xlim-right_distance-x_offset, os_c[4][1]+y_offset],
                       [left_distance + x_offset, os_c[6][1]+y_offset],[bound_xlim-right_distance-x_offset, os_c[6][1]+y_offset]]) 

  # User Input to select desired parking slot ([0~7])
  ROOT = tk.Tk()
  ROOT.withdraw()
  USER_INP = simpledialog.askstring(title="Autonomous Parallel Parking",
                                  prompt="Parking Slot Selection (0~7):")
  
  # Determine x0 and xf based on user input
  if(int(USER_INP) >= 0 or int(USER_INP) < 8):
    print("Selected Parking Slot:", USER_INP)
    heading = np.pi/2
    x0_1 = np.array([bound_xlim/2, bottom_distance+1, heading, 0, 0])
    xf_1 = np.array([waitpoints[int(USER_INP)][0], waitpoints[int(USER_INP)][1], heading, 0, 0])
    xf_2 = np.array([os_c[int(USER_INP)][0], os_c[int(USER_INP)][1], heading, 0, 0])

  else:             # invalud input
    tk.messagebox.showwarning(title="Autonomous Parallel Parking",message="Invalid Parking Slot")

  if (int(USER_INP) >= 4 and int(USER_INP) < 8):
    t_1 = xf_1[1] / 2
  else:
    t_1 = 1.5* xf_1[1] - 2

  print("final time for the 1st trajectory = ",t_1)
  prob = Direct_Collocation(int(USER_INP), x0_1, xf_1, t_1, os_c, bound_xlim, bound_ylim, 1)


  ## ------------ Plots --------------
  plt.ion()

  # Control Input Plot
  plt.rcParams["figure.figsize"] = [15, 7.5]
  plt.rcParams["figure.autolayout"] = True
  fig1, axs1 = plt.subplots(2, 2)
  fig1.set_facecolor("slategray")
  axs1[0][0].set_facecolor("lightgray")
  axs1[0][1].set_facecolor("lightgray")
  axs1[1][0].set_facecolor("lightgray")
  axs1[1][1].set_facecolor("lightgray")
  prob.axs1 = axs1


  plt.suptitle(r'$Control\ Inputs$', fontsize=16, c='midnightblue')
  axs1[0][0].set_title(r'$Acceleration\ (1st\ trajectory):\ u_1\ versus\ t$')
  axs1[0][0].set_xlabel(r'$t\ [s]$')
  axs1[0][0].set_ylabel(r'$[\frac{m}{s^2}]$')
  axs1[0][1].set_title(r'$Steering\ Angle\ Rate\ (1st\ trajectory):\ u_2\ versus\ t$')
  axs1[0][1].set_xlabel(r'$t\ [s]$')
  axs1[0][1].set_ylabel(r'$[\frac{rad}{s}]$')
  axs1[1][0].set_title(r'$Acceleration\ (2st\ trajectory):\ u_1\ versus\ t$')
  axs1[1][0].set_xlabel(r'$t\ [s]$')
  axs1[1][0].set_ylabel(r'$[\frac{m}{s^2}]$')
  axs1[1][1].set_title(r'$Steering\ Angle\ Rate\ (2st\ trajectory):\ u_2\ versus\ t$')
  axs1[1][1].set_xlabel(r'$t\ [s]$')
  axs1[1][1].set_ylabel(r'$[\frac{rad}{s}]$')
  

  # Velocity and Steering Angle Plot
  plt.rcParams["figure.figsize"] = [15, 7.5]
  plt.rcParams["figure.autolayout"] = True
  fig3, axs3 = plt.subplots(2, 2)
  fig3.set_facecolor("slategray")
  axs3[0][0].set_facecolor("lightgray")
  axs3[0][1].set_facecolor("lightgray")
  axs3[1][0].set_facecolor("lightgray")
  axs3[1][1].set_facecolor("lightgray")
  prob.axs3 = axs3


  plt.suptitle(r'$Velocity\ and\ Steering\ angle$', fontsize=16, c='midnightblue')
  axs3[0][0].set_title(r'$Velocity\ (1st\ trajectory):\ v\ versus\ t$')
  axs3[0][0].set_xlabel(r'$t\ [s]$')
  axs3[0][0].set_ylabel(r'$[\frac{m}{s}]$')
  axs3[0][1].set_title(r'$Steering\ Angle (1st\ trajectory):\delta \ versus\ t$')
  axs3[0][1].set_xlabel(r'$t\ [s]$')
  axs3[0][1].set_ylabel(r'$[\frac{rad}{s}]$')
  axs3[1][0].set_title(r'$Velocity\ (2st\ trajectory):\ v\ versus\ t$')
  axs3[1][0].set_xlabel(r'$t\ [s]$')
  axs3[1][0].set_ylabel(r'$[\frac{m}{s}]$')
  axs3[1][1].set_title(r'$Steering\ Angle\ (2st\ trajectory):\delta \ versus\ t$')
  axs3[1][1].set_xlabel(r'$t\ [s]$')
  axs3[1][1].set_ylabel(r'$[rad]$')


  # Vehicle parking state trajectory plot
  scaling = 3
  plt.rcParams["figure.figsize"] = [bound_xlim/scaling, bound_ylim/scaling]
  plt.rcParams["figure.autolayout"] = True
  fig2, axs2 = plt.subplots(1, 1)
  fig2.set_facecolor("slategray")
  axs2.set_facecolor("lightgray")

  prob.fig2 = fig2
  prob.axs2 = axs2

  # ------------ 1st trajectory ----------------

  # initial control sequence
  us = np.concatenate((np.ones((2, prob.N // 2)) *  -0.03,
                      np.ones((2, prob.N // 2)) * 0.03), axis=1)

  # initial state
  xs = prob.traj(us)


  plt.suptitle(r'$Autonomous\ Parking\ Trajectory$', fontsize=16, c='midnightblue')
  axs2.axis('equal')
  axs2.set_xlabel(r'$x\ [m]$')
  axs2.set_ylabel(r'$y\ [m]$')
  axs2.set_xlim([0, bound_xlim])
  axs2.set_xbound(0.0, bound_xlim)
  axs2.set_ylim([0, bound_ylim])
  axs2.set_ybound(0.0, bound_ylim)

  
  axs2.add_patch(Rectangle((0, 0), bound_xlim, bound_ylim, fc='none',
                  color ='midnightblue', lw = 7) )
  
  # Plotting Parked vehicles (obsacles)
  for j in range(prob.Nobst):
    if (j == prob.parking_slot):
      rect = plt.Rectangle((prob.obsts[:,:,j][0,0], prob.obsts[:,:,j][0,1]), 
                            prob.os_w, prob.os_l, color='black', linewidth=2, fill=False)
    else:
      rect = plt.Rectangle((prob.obsts[:,:,j][0,0], prob.obsts[:,:,j][0,1]), 
                            prob.os_w, prob.os_l, color='black')
    axs2.add_patch(rect)

  # plot initial trajectory
  prob.plot_traj(xs, us)
  
  # plot trajectory
  xs, us, cost = trajopt_sqp(xs, us, prob)
  prob.plot_traj(xs, us)
  print("cost for the 1st trajectory = ", cost)
  print("desired final state for 1st trajectory:", xf_1)
  print("actual final state for 1st trajectory:", xs[:,-1])



  # ------------ 2nd trajectory ----------------
  t_2 = x_offset + y_offset - 1    
  # t_2 = x_offset + y_offset + 9.5    # for object avoidance
  print("final time for the 2nd trajectory = ",t_2)


  prob_parallel = Direct_Collocation(int(USER_INP), xf_1, xf_2, t_2, os_c, bound_xlim, bound_ylim, 2)

  prob_parallel.axs1 = axs1

  prob_parallel.fig2 = fig2
  prob_parallel.axs2 = axs2
  prob_parallel.axs3 = axs3

  us_1 = np.concatenate((np.ones((2, prob_parallel.N // 2)) * 0.04,
                        np.ones((2, prob_parallel.N // 2)) * -0.04), axis=1)

  xs_1 = prob_parallel.traj(us_1)
  xs_1, us_1, cost_1 = trajopt_sqp(xs_1, us_1, prob_parallel)

  ## Trajectory animations
  #----------Plot the 1st part of the trajectory----------
  draw = Draw_vehicle()

  draw.part = 1
  for i in range(xs.shape[1]):
    axs2.add_patch(Rectangle((0, 0), bound_xlim, bound_ylim, fc='none',
                   color ='midnightblue', lw = 7) )
                   
    axs2.plot(xs[0, 0:i+1], xs[1, 0:i+1], color="lime", linewidth=3)
    plt.text(left_distance-1, bottom_distance-1, 'cost 1 = '+str(round(cost, 3)), fontsize = 8, weight='bold')

    draw.render(xs[0, i], xs[1, i], xs[2, i], xs[4, i], axs2)
    # enable to save the plots
    # fig2.savefig("./results/results_"+str(prob.parking_slot)+"/traj_1_"+str(i)+".jpg")
    plt.pause(prob.dt)

  
  #----------Plot the 2nd part of the trajectory-----------
  print("cost for the 2nd trajectory = ", cost_1)
  prob_parallel.plot_traj(xs_1, us_1)
  draw.part = 2

  for i in range(xs_1.shape[1]):
    axs2.add_patch(Rectangle((0, 0), bound_xlim, bound_ylim, fc='none',
                   color ='midnightblue', lw = 7) )
                   
    axs2.plot(xs_1[0, 0:i+1], xs_1[1, 0:i+1], color="lime", linewidth=3)
    plt.text(left_distance-1, bottom_distance-2, 'cost 2 = '+str(round(cost_1, 3)), fontsize = 8, weight='bold')
    plt.text(bound_xlim-right_distance-11, bound_ylim-2, 'desired state = '+str(np.round(xf_2, 3)), fontsize = 8, weight='bold')
    plt.text(bound_xlim-right_distance-11, bound_ylim-3, 'final state = '+str(np.round(xs_1[:,-1], 3)), fontsize = 8, weight='bold')
    draw.render(xs_1[0, i], xs_1[1, i], xs_1[2, i], xs_1[4, i], axs2)


    fig2.savefig("../results/results_"+str(prob.parking_slot)+"/traj_2_"+str(i)+".jpg")
    plt.pause(prob.dt)



  plt.ioff()
  print("desired final state for the 2nd trajectory", xf_2)
  print("final state for the 2nd trajectory", xs_1[:,-1])

  print("min dist. for 1st trajectory = ", prob.min_dist)
  print("min dist. for 2nd trajectory = ", prob_parallel.min_dist)

  # enable to save the plots
  # fig1.savefig("./results/results_"+str(prob.parking_slot)+"/control_"+str(prob.parking_slot)+".jpg")
  # fig3.savefig("./results/results_"+str(prob.parking_slot)+"/V_Steering_"+str(prob.parking_slot)+".jpg")

  plt.show()
  plt.savefig('')
  