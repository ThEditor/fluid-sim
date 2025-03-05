#include <mpi.h>
#include <GL/glut.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>

//---------------------------------------------------------------------
// Global simulation parameters
//---------------------------------------------------------------------
#define N 512                          // resolution
#define IX(i, j) ((i) + (N + 2) * (j)) // 2D indexing
#define SWAP(a, b)  \
  {                 \
    float *tmp = a; \
    a = b;          \
    b = tmp;        \
  }

static int win_x = 512;
static int win_y = 512;

static float dt = 0.1f;         // time step
static float diff = 0.0000001f; // diffusion rate for density
static float visc = 0.0001f;    // viscosity

// For the master process: full display grid size is (N+2) x (N+2)
static int full_size = (N + 2) * (N + 2);
static float *display_dens = NULL;

// MPI globals
int mpi_rank, mpi_size;

//---------------------------------------------------------------------
// Mouse interaction globals (only used by master)
static bool leftButtonDown = false;
static int mouseX, mouseY; // last known mouse coordinates

//---------------------------------------------------------------------
// Function prototypes for local simulation routines (for workers)
// The local grid for a worker has dimensions: (N+2) columns x (local_N+2) rows,
// where local_N = N / (mpi_size-1)
static void add_source(int size, float *x, float *s, float dt);
static void set_bnd_local(int b, float *x, int local_N);
static void diffuse_local(int b, float *x, float *x0, float diff, float dt, int local_N);
static void advect_local(int b, float *d, float *d0, float *u, float *v, float dt, int local_N);
static void project_local(float *u, float *v, float *p, float *div, int local_N);
static void dens_step_local(float *x, float *x0, float *u, float *v, float diff, float dt, int local_N);
static void vel_step_local(float *u, float *v, float *u0, float *v0, float visc, float dt, int local_N);
static void clear_prev_local(float *dens_prev, float *u_prev, float *v_prev, int local_size);

// Function prototypes for master callbacks
static void display();
static void reshape(int w, int h);
static void mouseFunc(int button, int state, int x, int y);
static void motionFunc(int x, int y);
static void master_idle();

//---------------------------------------------------------------------
// add_source: Add external source (density or velocity impulses)
// (Works for any array of given size.)
static void add_source(int size, float *x, float *s, float dt)
{
  for (int i = 0; i < size; i++)
  {
    x[i] += dt * s[i];
  }
}

//---------------------------------------------------------------------
// set_bnd_local: Set boundary conditions for a local grid.
// b==0: scalar field (density); b==1: x-velocity; b==2: y-velocity.
// The local grid has dimensions (N+2) x (local_N+2) where local_N is the number
// of interior rows in this subdomain.
static void set_bnd_local(int b, float *x, int local_N)
{
  // Left and right boundaries (i=0 and i=N+1) for interior rows j=1..local_N:
  for (int j = 1; j <= local_N; j++)
  {
    x[IX(0, j)] = (b == 1) ? -x[IX(1, j)] : x[IX(1, j)];
    x[IX(N + 1, j)] = (b == 1) ? -x[IX(N, j)] : x[IX(N, j)];
  }
  // Top and bottom boundaries (j=0 and j=local_N+1) for interior columns i=1..N:
  for (int i = 1; i <= N; i++)
  {
    x[IX(i, 0)] = (b == 2) ? -x[IX(i, 1)] : x[IX(i, 1)];
    x[IX(i, local_N + 1)] = (b == 2) ? -x[IX(i, local_N)] : x[IX(i, local_N)];
  }
  // Corners:
  x[IX(0, 0)] = 0.5f * (x[IX(1, 0)] + x[IX(0, 1)]);
  x[IX(0, local_N + 1)] = 0.5f * (x[IX(1, local_N + 1)] + x[IX(0, local_N)]);
  x[IX(N + 1, 0)] = 0.5f * (x[IX(N, 0)] + x[IX(N + 1, 1)]);
  x[IX(N + 1, local_N + 1)] = 0.5f * (x[IX(N, local_N + 1)] + x[IX(N + 1, local_N)]);
}

//---------------------------------------------------------------------
// diffuse_local: Diffuse a quantity on the local grid using Gauss-Seidel
static void diffuse_local(int b, float *x, float *x0, float diff, float dt, int local_N)
{
  float a = dt * diff * N * N;
  for (int k = 0; k < 20; k++)
  {
    for (int i = 1; i <= N; i++)
    {
      for (int j = 1; j <= local_N; j++)
      {
        x[IX(i, j)] = (x0[IX(i, j)] +
                       a * (x[IX(i - 1, j)] + x[IX(i + 1, j)] + x[IX(i, j - 1)] + x[IX(i, j + 1)])) /
                      (1 + 4 * a);
      }
    }
    set_bnd_local(b, x, local_N);
  }
}

//---------------------------------------------------------------------
// advect_local: Advect a field d on the local grid along velocity (u,v)
static void advect_local(int b, float *d, float *d0, float *u, float *v, float dt, int local_N)
{
  float dt0 = dt * N;
  for (int i = 1; i <= N; i++)
  {
    for (int j = 1; j <= local_N; j++)
    {
      float x = i - dt0 * u[IX(i, j)];
      float y = j - dt0 * v[IX(i, j)];
      if (x < 0.5f)
        x = 0.5f;
      if (x > N + 0.5f)
        x = N + 0.5f;
      int i0 = (int)x;
      int i1 = i0 + 1;
      if (y < 0.5f)
        y = 0.5f;
      if (y > local_N + 0.5f)
        y = local_N + 0.5f;
      int j0 = (int)y;
      int j1 = j0 + 1;
      float s1 = x - i0;
      float s0 = 1 - s1;
      float t1 = y - j0;
      float t0 = 1 - t1;
      d[IX(i, j)] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) + s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
    }
  }
  set_bnd_local(b, d, local_N);
}

//---------------------------------------------------------------------
// project_local: Make the velocity field divergence-free on the local grid.
static void project_local(float *u, float *v, float *p, float *div, int local_N)
{
  for (int i = 1; i <= N; i++)
  {
    for (int j = 1; j <= local_N; j++)
    {
      div[IX(i, j)] = -0.5f * (u[IX(i + 1, j)] - u[IX(i - 1, j)] + v[IX(i, j + 1)] - v[IX(i, j - 1)]) / N;
      p[IX(i, j)] = 0;
    }
  }
  set_bnd_local(0, div, local_N);
  set_bnd_local(0, p, local_N);

  for (int k = 0; k < 20; k++)
  {
    for (int i = 1; i <= N; i++)
    {
      for (int j = 1; j <= local_N; j++)
      {
        p[IX(i, j)] = (div[IX(i, j)] + p[IX(i - 1, j)] + p[IX(i + 1, j)] + p[IX(i, j - 1)] + p[IX(i, j + 1)]) / 4;
      }
    }
    set_bnd_local(0, p, local_N);
  }

  for (int i = 1; i <= N; i++)
  {
    for (int j = 1; j <= local_N; j++)
    {
      u[IX(i, j)] -= 0.5f * N * (p[IX(i + 1, j)] - p[IX(i - 1, j)]);
      v[IX(i, j)] -= 0.5f * N * (p[IX(i, j + 1)] - p[IX(i, j - 1)]);
    }
  }
  set_bnd_local(1, u, local_N);
  set_bnd_local(2, v, local_N);
}

//---------------------------------------------------------------------
// dens_step_local: Update density on the local grid.
static void dens_step_local(float *x, float *x0, float *u, float *v, float diff, float dt, int local_N)
{
  add_source((N + 2) * (local_N + 2), x, x0, dt);
  SWAP(x0, x);
  diffuse_local(0, x, x0, diff, dt, local_N);
  SWAP(x0, x);
  advect_local(0, x, x0, u, v, dt, local_N);
}

//---------------------------------------------------------------------
// vel_step_local: Update velocity on the local grid.
static void vel_step_local(float *u, float *v, float *u0, float *v0, float visc, float dt, int local_N)
{
  int local_size = (N + 2) * (local_N + 2);
  add_source(local_size, u, u0, dt);
  add_source(local_size, v, v0, dt);
  SWAP(u0, u);
  diffuse_local(1, u, u0, visc, dt, local_N);
  SWAP(v0, v);
  diffuse_local(2, v, v0, visc, dt, local_N);
  project_local(u, v, u0, v0, local_N);
  SWAP(u0, u);
  SWAP(v0, v);
  advect_local(1, u, u0, u0, v0, dt, local_N);
  advect_local(2, v, v0, u0, v0, dt, local_N);
  project_local(u, v, u0, v0, local_N);
}

//---------------------------------------------------------------------
// clear_prev_local: Clear previous arrays for the next time step.
static void clear_prev_local(float *dens_prev, float *u_prev, float *v_prev, int local_size)
{
  for (int i = 0; i < local_size; i++)
  {
    dens_prev[i] = u_prev[i] = v_prev[i] = 0.0f;
  }
}

//---------------------------------------------------------------------
// Master Process Callbacks (Rank 0)
//---------------------------------------------------------------------

// Display callback: Render the full density field assembled from worker segments.
static void display()
{
  glClear(GL_COLOR_BUFFER_BIT);
  float h = 1.0f / N;
  glBegin(GL_QUADS);
  for (int i = 1; i <= N; i++)
  {
    for (int j = 1; j <= N; j++)
    {
      float d = display_dens[IX(i, j)];
      if (d > 1.0f)
        d = 1.0f;
      glColor3f(0.0f, 0.0f, d);
      float x = (i - 1) * h;
      float y = (j - 1) * h;
      glVertex2f(x, y);
      glVertex2f(x + h, y);
      glVertex2f(x + h, y + h);
      glVertex2f(x, y + h);
    }
  }
  glEnd();
  glutSwapBuffers();
}

// Reshape callback: Update OpenGL viewport and projection.
static void reshape(int w, int h)
{
  win_x = w;
  win_y = h;
  glViewport(0, 0, w, h);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, 1, 0, 1, -1, 1);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
}

// Mouse callback: Track left button state.
static void mouseFunc(int button, int state, int x, int y)
{
  if (button == GLUT_LEFT_BUTTON)
  {
    if (state == GLUT_DOWN)
    {
      leftButtonDown = true;
      mouseX = x;
      mouseY = y;
    }
    else if (state == GLUT_UP)
    {
      leftButtonDown = false;
    }
  }
}

// Motion callback: Update mouse coordinates.
static void motionFunc(int x, int y)
{
  mouseX = x;
  mouseY = y;
}

// Idle callback for master: Receive simulation segments from all workers and forward impulses.
static void master_idle()
{
  MPI_Status status;
  int numWorkers = mpi_size - 1;
  // Each worker handles local_N interior rows. Assume even division.
  int local_N = N / numWorkers;
  // For each worker, receive its density segment (of size: (N+2) x local_N).
  for (int worker = 1; worker <= numWorkers; worker++)
  {
    MPI_Recv(&display_dens[IX(1, (worker - 1) * local_N + 1)], (N + 2) * local_N, MPI_FLOAT,
             worker, 0, MPI_COMM_WORLD, &status);
  }
  glutPostRedisplay();

  // If mouse impulse is active, broadcast it to all workers.
  if (leftButtonDown)
  {
    int i = (int)((mouseX / (float)win_x) * N + 1);
    int j = (int)(((win_y - mouseY) / (float)win_y) * N + 1);
    if (i < 1)
      i = 1;
    if (i > N)
      i = N;
    if (j < 1)
      j = 1;
    if (j > N)
      j = N;
    float impulse[4] = {(float)i, (float)j, 100.0f, -20.0f};
    for (int worker = 1; worker <= numWorkers; worker++)
    {
      MPI_Send(impulse, 4, MPI_FLOAT, worker, 1, MPI_COMM_WORLD);
    }
  }
}

//---------------------------------------------------------------------
// Simulation Worker Loop (Ranks >= 1)
//---------------------------------------------------------------------
static void simulation_loop()
{
  // Determine number of simulation workers and local rows per worker.
  int numWorkers = mpi_size - 1;
  int local_N = N / numWorkers; // number of interior rows for each worker
  int local_size = (N + 2) * (local_N + 2);

  // Allocate local arrays for velocity and density.
  float *u_local = new float[local_size];
  float *v_local = new float[local_size];
  float *u_prev_local = new float[local_size];
  float *v_prev_local = new float[local_size];
  float *dens_local = new float[local_size];
  float *dens_prev_local = new float[local_size];

  // Initialize arrays.
  for (int i = 0; i < local_size; i++)
  {
    u_local[i] = v_local[i] = u_prev_local[i] = v_prev_local[i] = 0.0f;
    dens_local[i] = dens_prev_local[i] = 0.0f;
  }

  // Determine neighboring ranks for ghost row exchange.
  // Workers are arranged in order: rank 1 is top, rank numWorkers is bottom.
  int up_rank = (mpi_rank == 1) ? MPI_PROC_NULL : mpi_rank - 1;
  int down_rank = (mpi_rank == numWorkers) ? MPI_PROC_NULL : mpi_rank + 1;

  MPI_Status status;

  // Main simulation loop.
  while (1)
  {
    // Check for any impulse messages from master.
    int flag = 0;
    while (1)
    {
      MPI_Iprobe(0, 1, MPI_COMM_WORLD, &flag, &status);
      if (!flag)
        break;
      float impulse[4];
      MPI_Recv(impulse, 4, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
      // Impulse coordinates are given in global grid terms.
      int global_row = (int)impulse[1];
      int start = (mpi_rank - 1) * local_N + 1; // global row index of first interior row in this worker
      int end = start + local_N - 1;
      if (global_row >= start && global_row <= end)
      {
        int local_j = global_row - start + 1; // convert to local index (1..local_N)
        int i = (int)impulse[0];
        dens_prev_local[IX(i, local_j)] += impulse[2];
        v_prev_local[IX(i, local_j)] += impulse[3];
      }
    }

    // Exchange ghost rows for dens_local.
    // Send top interior row (row 1) upward and receive into ghost row (row 0).
    MPI_Sendrecv(&dens_local[IX(1, 1)], (N + 2), MPI_FLOAT, up_rank, 0,
                 &dens_local[IX(1, 0)], (N + 2), MPI_FLOAT, up_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // Send bottom interior row (row local_N) downward and receive into ghost row (row local_N+1).
    MPI_Sendrecv(&dens_local[IX(1, local_N)], (N + 2), MPI_FLOAT, down_rank, 0,
                 &dens_local[IX(1, local_N + 1)], (N + 2), MPI_FLOAT, down_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // Do similar ghost exchanges for u_local and v_local.
    MPI_Sendrecv(&u_local[IX(1, 1)], (N + 2), MPI_FLOAT, up_rank, 0,
                 &u_local[IX(1, 0)], (N + 2), MPI_FLOAT, up_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&u_local[IX(1, local_N)], (N + 2), MPI_FLOAT, down_rank, 0,
                 &u_local[IX(1, local_N + 1)], (N + 2), MPI_FLOAT, down_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Sendrecv(&v_local[IX(1, 1)], (N + 2), MPI_FLOAT, up_rank, 0,
                 &v_local[IX(1, 0)], (N + 2), MPI_FLOAT, up_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&v_local[IX(1, local_N)], (N + 2), MPI_FLOAT, down_rank, 0,
                 &v_local[IX(1, local_N + 1)], (N + 2), MPI_FLOAT, down_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Run simulation step on local grid.
    vel_step_local(u_local, v_local, u_prev_local, v_prev_local, visc, dt, local_N);
    dens_step_local(dens_local, dens_prev_local, u_local, v_local, diff, dt, local_N);
    clear_prev_local(dens_prev_local, u_prev_local, v_prev_local, local_size);

    // Send the interior density (rows 1 .. local_N) back to master.
    MPI_Send(&dens_local[IX(1, 1)], (N + 2) * local_N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);

    // Sleep briefly to simulate time step delay.
    usleep(10000); // 10ms
  }

  // Clean up (never reached in this loop).
  delete[] u_local;
  delete[] v_local;
  delete[] u_prev_local;
  delete[] v_prev_local;
  delete[] dens_local;
  delete[] dens_prev_local;
}

//---------------------------------------------------------------------
// main: Initialize MPI, set up master or simulation workers.
//---------------------------------------------------------------------
int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  if (mpi_size < 2)
  {
    if (mpi_rank == 0)
    {
      printf("Please run with at least 2 MPI processes.\n");
    }
    MPI_Finalize();
    return 0;
  }

  if (mpi_rank == 0)
  {
    // MASTER PROCESS: Allocate full display grid and set up OpenGL.
    display_dens = new float[full_size];
    for (int i = 0; i < full_size; i++)
    {
      display_dens[i] = 0.0f;
    }
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(win_x, win_y);
    glutCreateWindow("2D Fluid in a Box - Master Process");
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutIdleFunc(master_idle);
    glutMouseFunc(mouseFunc);
    glutMotionFunc(motionFunc);
    glutPassiveMotionFunc(motionFunc);
    glutMainLoop();

    delete[] display_dens;
  }
  else
  {
    // SIMULATION WORKERS: All ranks >= 1 participate in simulation.
    simulation_loop();
  }

  MPI_Finalize();
  return 0;
}
