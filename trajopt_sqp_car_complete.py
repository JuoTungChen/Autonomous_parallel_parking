## EN.530.603 Applied Optimal Control
# Final Project

## -- Packages
# general
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from functools import partial
from trajopt_sqp import trajopt_sqp
# import userinput packages
import tkinter as tk
from tkinter import simpledialog
from tkinter import messagebox
import cv2

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

  def __init__(self, parking_slot, x0, xf, tf, os_c, bound_xlim, bound_ylim, part):

    # time horizon and segments
    self.parking_slot = parking_slot
    self.tf = int(tf)
    # self.dt = 1
    # self.N = (self.tf)
    self.N = 16

    self.dt = self.tf / self.N

    self.xlim = bound_xlim
    self.ylim = bound_ylim

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
    self.u_bound = np.array([[-3, 3], [-2, 2]])
    self.beta = 1.0

    # initial state
    self.x0 = x0

    # derised final parking state
    self.xf = xf

    self.part = part

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
    Lx = self.dt * self.Q @ (x- self.xf)
    Lxx = self.dt * self.Q
    Lu = self.dt * self.R @ u
    Luu = self.dt * self.R

    # if hasattr(self, 'u_bound'):
      # print('a')
  #   for i in range(len(self.u_bound)):
      # if k < self.N:
      #     for i in range(len(self.u_bound)):
      #         for j in range(len(self.u_bound[0])):
      #             if j == 0:   # upper bound
      #                 c, u_grad = self.calculate_ineq(u[i], self.u_bound[i][j],  1)
      #             if j == 1:   # lower bound
      #                 c, u_grad = self.calculate_ineq(u[i], self.u_bound[i][j], -1)
      #             ## adding penalties to the cost
      #             L = L + self.beta/2.0 * c **2
      #             Lu[i] = Lu[i] - self.beta * c * u_grad
      #     Luu = Luu + self.beta * np.diag([2, 2])

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
      cs[i] = -minDist

    # Walls constraints
    cs[self.Nobst]   = -min(self.carcoord[0,:]-0)
    cs[self.Nobst+1] = -min(self.xlim-self.carcoord[0,:])
    cs[self.Nobst+2] = -min(self.carcoord[1,:]-0)
    cs[self.Nobst+3] = -min(self.ylim-self.carcoord[1,:])

    # cs[self.Nobst+4] = -u[0]-self.u_bound[0]
    # cs[self.Nobst+5] = -self.u_bound[1]-u[0]
    # cs[self.Nobst+6] = -u[1]-self.u_bound[2]
    # cs[self.Nobst+7] = -self.u_bound[3]-u[1]
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
      cs[i] = -minDist

    # Walls constraints
    cs[self.Nobst]   = -min(self.carcoord[0,:]-0)
    cs[self.Nobst+1] = -min(self.xlim-self.carcoord[0,:])
    cs[self.Nobst+2] = -min(self.carcoord[1,:]-0)
    cs[self.Nobst+3] = -min(self.ylim-self.carcoord[1,:])
    cs[self.Nobst+4] = -min(u[0]-self.u_bound[0])
    cs[self.Nobst+5] = -min(self.u_bound[1]-u[0])
    cs[self.Nobst+6] = -min(u[1]-self.u_bound[2])
    cs[self.Nobst+7] = -min(self.u_bound[3]-u[1])
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
 

    # plot 1st control trajectory
    if (self.part == 1):
      self.axs2.plot(xs[0, :], xs[1, :], '-', c='dodgerblue', linewidth=1)
      # self.axs2.axis('equal')
      self.axs2.set_xlabel('x')
      self.axs2.set_ylabel('y')
      self.axs2.set_xlim([0, bound_xlim])
      # self.axs2.set_xbound(0.0, bound_xlim)
      self.axs2.set_ylim([0, bound_ylim])

      # u_1 = acceleration
      self.axs1[0][0].lines.clear()
      self.axs1[0][0].grid()
      self.axs1[0][0].plot(np.arange(0, self.tf, self.dt), us[0, :], '-b')
      self.axs1[0][0].relim()
      self.axs1[0][0].autoscale_view()

      # u_2 = steering angle rate
      self.axs1[0][1].lines.clear()
      self.axs1[0][1].grid()
      self.axs1[0][1].plot(np.arange(0, self.tf, self.dt), us[1, :], '-r')
      self.axs1[0][1].relim()
      self.axs1[0][1].autoscale_view()


    # plot 1st control trajectory
    if (self.part == 2):
      self.axs2.plot(xs[0, :], xs[1, :], '-', c='royalblue', linewidth=1)
      # self.axs2.axis('equal')
      self.axs2.set_xlabel('x')
      self.axs2.set_ylabel('y')
      self.axs2.set_xlim([0, bound_xlim])
      # self.axs2.set_xbound(0.0, bound_xlim)
      self.axs2.set_ylim([0, bound_ylim])

      self.axs1[1][0].lines.clear()
      self.axs1[1][0].grid()
      self.axs1[1][0].plot(np.arange(0, self.tf, self.dt), us[0, :], '-b')
      self.axs1[1][0].relim()
      self.axs1[1][0].autoscale_view()

      self.axs1[1][1].lines.clear()
      self.axs1[1][1].grid()
      self.axs1[1][1].plot(np.arange(0, self.tf, self.dt), us[1, :], '-r')
      self.axs1[1][1].relim()
      self.axs1[1][1].autoscale_view()

    # drawing updated values
    fig2.canvas.draw()

    # This will run the GUI event
    # loop until all UI events
    # currently waiting have been processed
    fig2.canvas.flush_events()




class Draw_vehicle:
  def __init__(self) -> None:
    self.margin = 0.05
    self.car_length = 4
    self.car_width = 2
    self.wheel_length = 0.75
    self.wheel_width = 7/20
    self.wheel_positions = self.margin * np.array([[25,15],[25,-15],[-25,15],[-25,-15]])
    # self.wheel_positions = np.array([[25,15],[25,-15],[-25,15],[-25,-15]])

    
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
    # rotated_struct += np.array([x,y]) + np.array([10*self.margin,10*self.margin])
    rotated_struct += np.array([x,y])
    # print("wheels = ", self.wheel_struct)
    # rendered = cv2.fillPoly(self.background.copy(), [rotated_struct], self.color)
    plot.fill(rotated_struct[:,0], rotated_struct[:,1], color=self.car_color)
    # 7BC8F6
    rotated_wheel_center = self.rotate_car(self.wheel_positions, angle=psi)
    # print(rotated_wheel_center)

    for i, wheel in enumerate(rotated_wheel_center):
        # rotate front wheels (steering angle + heading)
        if i < 2:
            rotated_wheel = self.rotate_car(self.wheel_struct, angle=delta+psi)
        # rotate back wheels (heading)
        else:
            rotated_wheel = self.rotate_car(self.wheel_struct, angle=psi)
        rotated_wheel += np.array([x,y]) + wheel 
        # print(i, rotated_wheel)

        plot.fill(rotated_wheel[:,0], rotated_wheel[:,1], color=self.wheel_color)

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


  # 2 stage waitpoint parameters
  x_offset = 3
  y_offset = 4
  waitpoints = np.array([[left_distance + x_offset, os_c[0][1]+y_offset],[bound_xlim-right_distance-x_offset, os_c[0][1]+y_offset],
                       [left_distance + x_offset, os_c[2][1]+y_offset],[bound_xlim-right_distance-x_offset, os_c[2][1]+y_offset],
                       [left_distance + x_offset, os_c[4][1]+y_offset],[bound_xlim-right_distance-x_offset, os_c[4][1]+y_offset],
                       [left_distance + x_offset, os_c[6][1]+y_offset],[bound_xlim-right_distance-x_offset, os_c[6][1]+y_offset]]) 

  # User Input desired parking slot ([0,7])
  ROOT = tk.Tk()
  ROOT.withdraw()
  USER_INP = simpledialog.askstring(title="Autonomous Parallel Parking",
                                  prompt="Parking Slot Selection (0~7):")
  
  # Determine x0 and xf based on user input
  # if(int(USER_INP) == 0 or int(USER_INP) == 1 or int(USER_INP) == 2 or int(USER_INP) == 3):
  if(int(USER_INP) >= 0 or int(USER_INP) < 8):

    heading = np.pi/2
    x0_1 = np.array([bound_xlim/2, bottom_distance+1, heading, 0, 0])
    xf_1 = np.array([waitpoints[int(USER_INP)][0], waitpoints[int(USER_INP)][1], heading, 0, 0])
    xf_2 = np.array([os_c[int(USER_INP)][0], os_c[int(USER_INP)][1], heading, 0, 0])


  # elif(int(USER_INP) == 4 or int(USER_INP) == 5 or int(USER_INP) == 6 or int(USER_INP) == 7):
  #   heading = -np.pi/2
  #   x0_1 = np.array([bound_xlim/2, bottom_distance+4/2, heading, 0, 0])
  #   xf_1 = np.array([waitpoints[int(USER_INP)][0], waitpoints[int(USER_INP)][1], heading, 0, 0])
  #   xf_2 = np.array([os_c[int(USER_INP)][0], os_c[int(USER_INP)][1], heading, 0, 0])

  else:             # invalud input
    tk.messagebox.showwarning(title="Autonomous Parallel Parking",message="Invalid Parking Slot")

  # prob
  t_1 = xf_1[1] / 2
  prob = Direct_Collocation(int(USER_INP), x0_1, xf_1, t_1, os_c, bound_xlim, bound_ylim, 1)
  # prob = Direct_Collocation(int(USER_INP), x0_1, xf_1, waitpoints)

  # xs = np.append(xs, xs_1, axis=1)
  # us = np.append(us, us_1, axis=1)

  ## -- Plots
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
  axs1[0][0].set_ylabel(r'$\frac{m}{s^2}$')
  axs1[0][1].set_title(r'$Steering\ Angle\ Rate\ (1st\ trajectory):\ u_2\ versus\ t$')
  axs1[0][1].set_xlabel(r'$t\ [s]$')
  axs1[0][1].set_ylabel(r'$\frac{rad}{s}$')
  axs1[1][0].set_title(r'$Acceleration\ (2st\ trajectory):\ u_1\ versus\ t$')
  axs1[1][0].set_xlabel(r'$t\ [s]$')
  axs1[1][0].set_ylabel(r'$\frac{m}{s^2}$')
  axs1[1][1].set_title(r'$Steering\ Angle\ Rate\ (2st\ trajectory):\ u_2\ versus\ t$')
  axs1[1][1].set_xlabel(r'$t\ [s]$')
  axs1[1][1].set_ylabel(r'$\frac{rad}{s}$')
  
  # Vehicle parking state trajectory plot
  scaling = 4
  plt.rcParams["figure.figsize"] = [bound_xlim/scaling+0.5, bound_ylim/scaling]
  plt.rcParams["figure.autolayout"] = True
  fig2, axs2 = plt.subplots(1, 1)
  fig2.set_facecolor("slategray")
  axs2.set_facecolor("lightgray")

  prob.fig2 = fig2
  prob.axs2 = axs2
   # initial control sequence
  us = np.concatenate((np.ones((2, prob.N // 2)) *  0.02,
                       np.ones((2, prob.N // 2)) * -0.02), axis=1)

  # initial state
  xs = prob.traj(us)


  plt.suptitle(r'$Autonomous\ Parking\ Trajectory$', fontsize=16, c='midnightblue')
  axs2.axis('equal')
  axs2.set_xlabel(r'$x\ [m]$')
  axs2.set_ylabel(r'$y\ [m]$')
  # axs2.set_aspect('auto')
  axs2.set_xlim([0, bound_xlim])
  axs2.set_xbound(0.0, bound_xlim)
  axs2.set_ylim([0, bound_ylim])
  axs2.set_ybound(0.0, bound_ylim)

  
  axs2.add_patch(Rectangle((0, 0), bound_xlim, bound_ylim, fc='none',
                   color ='midnightblue', lw = 7) )
  
  # Parked vehicles (obsacles)
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
  print("cost for the 1st trajectory = ", cost)
  prob.plot_traj(xs, us)
  t_2 = x_offset + y_offset
  prob_parallel = Direct_Collocation(int(USER_INP), xs[:,-1], xf_2, t_2, os_c, bound_xlim, bound_ylim, 2)
  prob_parallel.x0[3]= 0            # reset the initial velocity to 0
  prob_parallel.axs1 = axs1

  prob_parallel.fig2 = fig2
  prob_parallel.axs2 = axs2
  us_1 = np.concatenate((np.ones((2, prob_parallel.N // 2)) * 0.02,
                       np.ones((2, prob_parallel.N // 2)) * -0.02), axis=1)

  xs_1 = prob_parallel.traj(us_1)
  xs_1, us_1, cost_1 = trajopt_sqp(xs_1, us_1, prob_parallel)

  # Trajectory animation
  draw = Draw_vehicle()

  for i in range(xs.shape[1]):
    axs2.add_patch(Rectangle((0, 0), bound_xlim, bound_ylim, fc='none',
                   color ='midnightblue', lw = 7) )
                   
    axs2.plot(xs[0, 0:i+1], xs[1, 0:i+1], color="lime", linewidth=3)
    draw.render(xs[0, i], xs[1, i], xs[2, i], xs[4, i], axs2)

    plt.pause(prob.dt)


  print("cost for the 2nd trajectory = ", cost_1)
  prob_parallel.plot_traj(xs_1, us_1)
  for i in range(xs_1.shape[1]):
    axs2.add_patch(Rectangle((0, 0), bound_xlim, bound_ylim, fc='none',
                   color ='midnightblue', lw = 7) )
                   
    axs2.plot(xs_1[0, 0:i+1], xs_1[1, 0:i+1], color="lime", linewidth=3)
    draw.render(xs_1[0, i], xs_1[1, i], xs_1[2, i], xs_1[4, i], axs2)

    plt.pause(prob.dt)

  plt.ioff()
  print("final state for the 1st trajectory", xs[:,-1])
  print("final state for the 2nd trajectory", xs_1[:,-1])

  plt.show()
  plt.savefig('')