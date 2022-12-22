 # Autonomous Parallel Parking

## Objectives:
The objective of this project – Autonomous Parallel Parking – is to develop an optimal
approach for vehicle’s trajectory generation, as well as obstacle avoidance for performing
parallel parking motion.

The dynamics of the vehicle is given by:

```math
    \dot{x} = f(x,u) = 
    \begin{bmatrix}
        \dot{x} \\
        \dot{y} \\
        \dot{\theta} \\
        \dot{v} \\
        \dot{\delta}
    \end{bmatrix} = 
    \begin{bmatrix}
        v\cos(\theta) \\
        v\sin(\theta) \\
        \frac{v}{L}\tan(\delta) \\
        u_1 \\
        u_2
    \end{bmatrix}
```

The trajectory generation and obstacle avoidance can be formulated as a constrained nonlinear optimal control problem, where we need to minimize the cost of the system. Our goal is to minimize the control effort, so the cost functional $J(x, u, t)$ can be represented by the following equation:
```math
J = \frac{1}{2} [{(x(t_f) - x_{f})}^T Q_f (x(t_f) - x_{f}) \int^{t_f}_{0} \frac{1}{2} [{(x - x_{f})}^T Q (x - x_{f}) + u^T R u]
```

The cost functional is minimized while subjecting to the inequality constraints $f(x, u, t)$ containing the minimum distance between the vehicle and the obstacles, as well as the boundaries for the control inputs. To meet the criteria of a comfortable driving experience, the limit value for acceleration were set to under $1.5\ (m/s^2)$, while the limit for steering angle rate were set to under $0.5\ (rad/s)$.

&nbsp;
## Approach:
We have a nonlinear optimization problem, which requires numerical methods to find a solution. We approach the problem using the \textit{Direct Collocation} method, as it is easier to handle the path constraints, while simultaneously remaining the efficiency, stability, and simplicity under the large-scale complex problem.
As for the implementation, the running cost $L(x, u, t)$ is approximated by $\frac{1}{2} [{(x - x_{f})}^T Q (x - x_{f}) + u^T R u] \Delta t$.
Since every parking slot has a different final state, in the cost function, we subtract the state with the desired final state in order to minimize the difference between them. The penalty weights after experimenting $(Q, R, Q_f)$ were chosen to be:

```math
Q = 0_{5 \times 5},\ 
    R = 
    \begin{bmatrix}
        1& 0\\
        0& 3
    \end{bmatrix},\ 
    Q_f = 
    \begin{bmatrix}
        5 & 0 & 0 & 0 & 0  \\
        0 & 5 & 0 & 0 & 0 \\
        0 & 0 & 30 & 0 & 0 \\
        0 & 0 & 0 & 10 & 0 \\
        0 & 0 & 0 & 0 & 3
    \end{bmatrix}
```
Since we value the final heading angle the most, so in the $Q_f$, we chose it to be the largest among all the other states. On the other hand, we want our final velocity to be as close to 0 as possible, so the weight is also chosen to be slightly larger than the remaining states.

&nbsp;
### Obstacle Avoidance
The vehicle and the obstacles are all rectangular objects, it would be inaccurate if we only calculate the distance between them using their center coordinates (i.e. as we did for disk obstacles). As a result, we have developed a function to calculate the minimum distance between two rectangular objects. The calculation involves the following steps:

1. Given the center coordinates of the rectangles $\rightarrow$ calculate 4 corner coordinates of each rectangle.
2. For each corner of our vehicle, calculate the shortest distance between the corner and the sides of the obstacle (parked vehicle).
   1. If the point lies within the line segment $\rightarrow$ return the vertical distance
   2. If the point lies outside of the line segment $\rightarrow$ return the distance between the point and the closest endpoint

3. Repeat the process for the vehicle and all objects, then return the minimum value among those.

Finally, after obtaining the minimum distance between the obstacles and the vehicle, we can turn it into inequality constraints. 

For safety reason, we want to add a buffer distance between the vehicle and the obstacles, so we add an additional 0.5 meter to each inequality constraint to ensure collisions will never occur. 

&nbsp;

### Two stage parking
The standard procedure to perform a parallel parking usually involves a two-stage parking technique, where the vehicle drives forward to first reach an intermediate goal right in front of the parking slot. Then, the vehicle needs to start reversing and adjust its steering angle to park into the slot. Therefore, we adopt the same technique to perform the autonomous parallel parking. We setup an intermediate goal for each parking slots, where we offset the x inward by 3 meters, and the y upward by 4 meters. 

&nbsp;

## Results:
For simulation, we created a parking lot containing 8 slots, and the user is able to select which slot they want to park in. Once the parking slot is specified, all the remaining slots will become obstacles, and the boundaries of the parking lot will also become walls, where we want to avoid collision.

![alt text](https://github.com/JuoTungChen/Autonomous_parallel_parking/blob/master/result_plots/ParkingSlot_4.gif)
![alt text](https://github.com/JuoTungChen/Autonomous_parallel_parking/blob/master/result_plots/ParkingSlot_5.gif)
![alt text](https://github.com/JuoTungChen/Autonomous_parallel_parking/blob/master/result_plots/ParkingSlot_6.gif)
![alt text](https://github.com/JuoTungChen/Autonomous_parallel_parking/blob/master/result_plots/ParkingSlot_7.gif)





