# Fluid Box Simulation

This project implements a 2D fluid simulation using two parallel approaches:

- **MPI Version:** Distributed simulation using MPI and OpenGL for visualization. See the source in [openmpi/fluid_box.cpp](openmpi/fluid_box.cpp).
- **OpenMP Version:** Multi-threaded simulation using OpenMP and OpenGL for visualization. See the source in [openmp/fluid_box.cpp](openmp/fluid_box.cpp).

## Prerequisites

- **MPI Version**: MPI library (e.g. Open MPI), OpenGL, and GLUT.
- **OpenMP Version**: A C++ compiler with OpenMP support, OpenGL, and GLUT.

## Building

### MPI Version

Compile with an MPI C++ compiler. For example:

```sh
mpic++ openmpi/fluid_box.cpp -lGL -lGLU -lglut -lm -o fluid_box_mpi
```

Run the simulation with at least 2 processes:

```sh
mpirun -np 4 ./fluid_sim_mpi
```

### OpenMP Version

Compile with OpenMP enabled. For example:

```sh
g++ -fopenmp openmp/fluid_box.cpp -lGL -lGLU -lglut -lm -o fluid_box_omp
```

Run the simulation:

```sh
./fluid_sim_omp [num_threads]
```

(The optional argument `[num_threads]` specifies the number of threads to use.)

## Usage

- **Visualization:** Both simulation versions use OpenGL/GLUT to display the fluid density field.
- **Interaction:** Left-mouse clicks or movements add impulses to the simulation. Use mouse interactions to update the fluid dynamics.

## Credits

- [But How DO Fluid Simulations Work? - Gonkee](https://www.youtube.com/watch?v=qsYE1wMEMPA)
- [Jos Stam - Real Time Fluid Dynamics for Games (2003)](http://graphics.cs.cmu.edu/nsp/course/15-464/Fall09/papers/StamFluidforGames.pdf)
