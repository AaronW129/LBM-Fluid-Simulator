# Explanation

This project is a fluid simulator that uses the Lattice Boltzman Method (LBM), to simulate fluid flow. Typically, the making of a fluid simulator requires the user to solve for the Navier-Stokes equation which is a set of partial differential equations that describe how a fluid evolves over time. This can not only complicate software, but also burden the machine in constantly computing these partial differential equations if implemented incorrectly.

In contrast, LBM allows for the user to use discretized data to describe fluid flow, which is especially convenient when the user wants to choose how detailed they want their simulation to be; they only need to increase the amount of lattice to consider even though all the cells have the same exact algorithm (except for a few boundary conditions)

The main idea of LBM is to model a fluid onto a lattice and each cell in the lattice directly interacts with neighboring cells in terms of weights, i.e. the probability a fluid particle travels from the main cell to the neighboring cell. In which case, there are only 9 cases to consider: top left, top, top right, left, source, right, bottom left, bottom, and bottom right. 

We model the interaction and distribution function using the "number density" (n_i) of a fluid, which represents how many particles are packed per unit of space. From these microscopic properties, i.e. the "number density" and "weights", we can gain info on the macroscopic property of the cell: its macroscopi density, rho, and its fluid velocity, u. This, coupled with 2 equations that we used the program, allows us to predict what the microscopic density of the cell is NOW and what the neighboring cells gained.

And so, the programming / implementation of the method then boils down to a careful handling of arrays within arrays. We set up a lattice of cells and each of the cell is also an array, which houses the data of its microscopic properties (n_i for all i). This means that each cell is an array of 9 numbers, representing the microscopic densities of the cell. From these cells, we can derive their NEW densities from neighboring cells. We also roll each cell using its fluid velocity. We repeat this process for all cells until we get a new lattice that reflects the evolution of the fluid, i.e. the next frame of the animation.

Computationally, this comes with some challenges, especially with the boundary conditions. The Zou-He boundary condition is a computational method specific for the LBM that explains how to recreate border that "absorb" waves rather than either reflecting or wrapping around waves from the right to the left (as is the problem with the np.rolling) function. The boundary condition simply states that for these "inlet-outlet" walls, we make cells consider ONLY the immediate neighboring cells. This dampens alot of reflections and / or wrap arounds in the simulation.

As for the construction of obstacles with continuous representation, like the sphere, we simply start with the algebraic representation of these obstacles and choose all points in the lattice whose x,y coordinate satisfies the algebraic inequality. The variables for the equations on the airfoil can be derived from their NACA code, which can be found from the Wikipedia page. 

For animating the simulation, we render each frame after the time reaches an even number, which is a tick. In which, the process above takes place via code and the next frame is prepared after clearing the previous frame / lattice.

<img width="1000" height="500" alt="windtunnel" src="https://github.com/user-attachments/assets/a82b8398-26c1-44d4-b047-f9c47afbfb75" />
<img width="1000" height="500" alt="airfoil_2" src="https://github.com/user-attachments/assets/126ea71f-9a9c-4144-b176-bddc37edbba1" />


# Credits:
Thanks to the following sources for the theory behind the code!

https://medium.com/swlh/create-your-own-lattice-boltzmann-simulation-with-python-8759e8b53b1c
- Philip Mocz

https://www.youtube.com/watch?v=JFWqCQHg-Hs
- Matias Ortiz
- A video explaining the Mocz's code

https://physics.weber.edu/schroeder/javacourse/LatticeBoltzmann.pdf
- An explanation of the Lattice Boltzmann Method that explains its algorithm and a roadmap on how to implement it.

https://en.wikipedia.org/wiki/NACA_airfoil
- An explanation of the naming of air foils and the algebraic expressions of their shapes.

