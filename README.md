 # Autonomous Parallel Parking

## Objectives:
The objective of this project – Autonomous Parallel Parking – is to develop an optimal
approach for vehicle’s trajectory generation, as well as obstacle avoidance for performing
parallel parking motion in the shortest time possible.
The project will be done under the assumption that the vehicle is a nonholonomic system
which has kinematic constraints, i.e., the vehicle is constrained such that it cannot move in
an arbitrary direction in its configuration space. In our case, the vehicle is rear-wheel driven,
operated under the assumption that the wheels roll without slipping.
The dynamics of the vehicle is given by:
$$
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
$$

The trajectory generation and obstacle avoidance can be formulated as a constrained noninear optimal control problem, where we need to minimize the cost of the system. Our goal is to minimize the control parameters $u = (\dot{v}, \dot{\delta})$, so the cost function L(x, u, t) will be a quadratic function subjected to the inequality constraints $f(x, u, t)$. In addition, obstacles will be added as a penalty term to the cost function. 

&nbsp;
## Approach:
Since it would be a nonlinear optimization problem, it will require numerical methods to find a solution. The team selected the *Direct Collocation*
method for the task, as it includes final time as an optimization variable which suits our objectives, while also being efficient.


&nbsp;
## Environmental Setup:



## Results:


![alt text](https://github.com/JuoTungChen/Autonomous_parallel_parking/blob/master/result_plots/Trajectory_0.png)
![alt text](https://github.com/JuoTungChen/Autonomous_parallel_parking/blob/master/result_plots/Trajectory_2.png)
![alt text](https://github.com/JuoTungChen/Autonomous_parallel_parking/blob/master/result_plots/Trajectory_4.png)
![alt text](https://github.com/JuoTungChen/Autonomous_parallel_parking/blob/master/result_plots/Trajectory_5.png)
![alt text](https://github.com/JuoTungChen/Autonomous_parallel_parking/blob/master/result_plots/Trajectory_6.png)




