# Fluid Box Simulation

This project implements a 2D fluid simulation using two parallel approaches:

- **MPI Version:** Distributed simulation using MPI and OpenGL for visualization. See the source in [openmpi/fluid_box.cpp](openmpi/fluid_box.cpp).
- **OpenMP Version:** Multi-threaded simulation using OpenMP and OpenGL for visualization. See the source in [openmp/fluid_box.cpp](openmp/fluid_box.cpp).

## Prerequisites

Before running `make all`, ensure you have the following packages installed:

- **MPI:** Open MPI (e.g. `openmpi-bin` and `libopenmpi-dev` on Ubuntu)
- **OpenGL:** OpenGL development libraries (e.g. `libgl1-mesa-dev`)
- **GLUT:** GLUT development libraries (e.g. `freeglut3-dev`)
- **C++ Compiler:** A C++ compiler with OpenMP support (e.g. `g++`)

## Building

Compile project:

```sh
make all
```

### MPI Version

Run the simulation:

```sh
mpirun -np 4 ./out/fluid_sim_mpi
```
(The argument `4` specifies the number of threads to use.)

### OpenMP Version

Run the simulation:

```sh
./out/fluid_sim_omp 4
```
(The argument `4` specifies the number of threads to use.)

## Usage

- **Visualization:** Both simulation versions use OpenGL/GLUT to display the fluid density field.
- **Interaction:** Left-mouse clicks or movements add impulses to the simulation. Use mouse interactions to update the fluid dynamics.

## Credits

- [But How DO Fluid Simulations Work? - Gonkee](https://www.youtube.com/watch?v=qsYE1wMEMPA)
- [Jos Stam - Real Time Fluid Dynamics for Games (2003)](http://graphics.cs.cmu.edu/nsp/course/15-464/Fall09/papers/StamFluidforGames.pdf)
