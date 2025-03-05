#include <GL/glut.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <iostream>
#include <omp.h>

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

static float dt = 0.1f;       // time step
static float diff = 0.00001f; // diffusion rate
static float visc = 0.00001f; // viscosity

static int size = (N + 2) * (N + 2);

// Global simulation arrays
static float *u, *v, *u_prev, *v_prev;
static float *dens, *dens_prev;

// Mouse interaction globals
static bool leftButtonDown = false;
static int mouseX, mouseY;

//-----------------------
// Utility Functions
//-----------------------
void add_source(int n, float *x, float *s, float dt)
{
#pragma omp parallel for
  for (int i = 0; i < n; i++)
  {
    x[i] += dt * s[i];
  }
}

void set_bnd(int b, float *x)
{
// Left and right boundaries
#pragma omp parallel for
  for (int j = 1; j <= N; j++)
  {
    x[IX(0, j)] = (b == 1) ? -x[IX(1, j)] : x[IX(1, j)];
    x[IX(N + 1, j)] = (b == 1) ? -x[IX(N, j)] : x[IX(N, j)];
  }
// Top and bottom boundaries
#pragma omp parallel for
  for (int i = 1; i <= N; i++)
  {
    x[IX(i, 0)] = (b == 2) ? -x[IX(i, 1)] : x[IX(i, 1)];
    x[IX(i, N + 1)] = (b == 2) ? -x[IX(i, N)] : x[IX(i, N)];
  }
  // Corners
  x[IX(0, 0)] = 0.5f * (x[IX(1, 0)] + x[IX(0, 1)]);
  x[IX(0, N + 1)] = 0.5f * (x[IX(1, N + 1)] + x[IX(0, N)]);
  x[IX(N + 1, 0)] = 0.5f * (x[IX(N, 0)] + x[IX(N + 1, 1)]);
  x[IX(N + 1, N + 1)] = 0.5f * (x[IX(N, N + 1)] + x[IX(N + 1, N)]);
}

// Diffuse using a Jacobi iteration (20 iterations)
void diffuse(int b, float *x, float *x0, float diff, float dt)
{
  float a = dt * diff * N * N;
  float *x_new = new float[size];
  // initialize x_new (could also use memset if 0)
  for (int i = 0; i < size; i++)
  {
    x_new[i] = x[i];
  }

  for (int k = 0; k < 20; k++)
  {
#pragma omp parallel for collapse(2)
    for (int i = 1; i <= N; i++)
    {
      for (int j = 1; j <= N; j++)
      {
        x_new[IX(i, j)] = (x0[IX(i, j)] +
                           a * (x[IX(i - 1, j)] + x[IX(i + 1, j)] + x[IX(i, j - 1)] + x[IX(i, j + 1)])) /
                          (1 + 4 * a);
      }
    }
    set_bnd(b, x_new);
// Copy new values back to x for the next iteration.
#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
      x[i] = x_new[i];
    }
  }
  delete[] x_new;
}

void advect(int b, float *d, float *d0, float *u, float *v, float dt)
{
  float dt0 = dt * N;
#pragma omp parallel for collapse(2)
  for (int i = 1; i <= N; i++)
  {
    for (int j = 1; j <= N; j++)
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
      if (y > N + 0.5f)
        y = N + 0.5f;
      int j0 = (int)y;
      int j1 = j0 + 1;
      float s1 = x - i0;
      float s0 = 1 - s1;
      float t1 = y - j0;
      float t0 = 1 - t1;
      d[IX(i, j)] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) +
                    s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
    }
  }
  set_bnd(b, d);
}

void project(float *u, float *v, float *p, float *div)
{
#pragma omp parallel for collapse(2)
  for (int i = 1; i <= N; i++)
  {
    for (int j = 1; j <= N; j++)
    {
      div[IX(i, j)] = -0.5f * (u[IX(i + 1, j)] - u[IX(i - 1, j)] + v[IX(i, j + 1)] - v[IX(i, j - 1)]) / N;
      p[IX(i, j)] = 0;
    }
  }
  set_bnd(0, div);
  set_bnd(0, p);

  for (int k = 0; k < 20; k++)
  {
#pragma omp parallel for collapse(2)
    for (int i = 1; i <= N; i++)
    {
      for (int j = 1; j <= N; j++)
      {
        p[IX(i, j)] = (div[IX(i, j)] + p[IX(i - 1, j)] + p[IX(i + 1, j)] + p[IX(i, j - 1)] + p[IX(i, j + 1)]) / 4;
      }
    }
    set_bnd(0, p);
  }

#pragma omp parallel for collapse(2)
  for (int i = 1; i <= N; i++)
  {
    for (int j = 1; j <= N; j++)
    {
      u[IX(i, j)] -= 0.5f * N * (p[IX(i + 1, j)] - p[IX(i - 1, j)]);
      v[IX(i, j)] -= 0.5f * N * (p[IX(i, j + 1)] - p[IX(i, j - 1)]);
    }
  }
  set_bnd(1, u);
  set_bnd(2, v);
}

void clear_prev(float *x, int n)
{
#pragma omp parallel for
  for (int i = 0; i < n; i++)
  {
    x[i] = 0.0f;
  }
}

//-----------------------
// Simulation Steps
//-----------------------
void dens_step(float *x, float *x0, float *u, float *v, float diff, float dt)
{
  add_source(size, x, x0, dt);
  SWAP(x0, x);
  diffuse(0, x, x0, diff, dt);
  SWAP(x0, x);
  advect(0, x, x0, u, v, dt);
}

void vel_step(float *u, float *v, float *u0, float *v0, float visc, float dt)
{
  add_source(size, u, u0, dt);
  add_source(size, v, v0, dt);
  SWAP(u0, u);
  diffuse(1, u, u0, visc, dt);
  SWAP(v0, v);
  diffuse(2, v, v0, visc, dt);
  project(u, v, u0, v0);
  SWAP(u0, u);
  SWAP(v0, v);
  advect(1, u, u0, u0, v0, dt);
  advect(2, v, v0, u0, v0, dt);
  project(u, v, u0, v0);
}

//-----------------------
// OpenGL Callbacks
//-----------------------
void display()
{
  glClear(GL_COLOR_BUFFER_BIT);
  float h = 1.0f / N;
  glBegin(GL_QUADS);
  for (int i = 1; i <= N; i++)
  {
    for (int j = 1; j <= N; j++)
    {
      float d = dens[IX(i, j)];
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

void reshape(int w, int h)
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

void mouseFunc(int button, int state, int x, int y)
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

void motionFunc(int x, int y)
{
  mouseX = x;
  mouseY = y;
}

// The idle callback runs one simulation time step and then requests a redraw.
void idle()
{
  // If mouse impulse is active, add a source at the corresponding grid cell.
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
    dens_prev[IX(i, j)] += 100.0f;
    v_prev[IX(i, j)] += -20.0f;
  }

  // Update velocity and density fields.
  vel_step(u, v, u_prev, v_prev, visc, dt);
  dens_step(dens, dens_prev, u, v, diff, dt);

  // Clear the source arrays.
  clear_prev(dens_prev, size);
  clear_prev(u_prev, size);
  clear_prev(v_prev, size);

  glutPostRedisplay();
  usleep(10000); // 10ms delay for visualization purposes
}

//-----------------------
// Main Function
//-----------------------
int main(int argc, char **argv)
{
  int num_threads = 4;

  if (argc > 1)
  {
    num_threads = atoi(argv[1]);
  }

  omp_set_num_threads(num_threads);
  std::cout << "Running with " << num_threads << " threads.\n";

  u = new float[size];
  v = new float[size];
  u_prev = new float[size];
  v_prev = new float[size];
  dens = new float[size];
  dens_prev = new float[size];

  for (int i = 0; i < size; i++)
  {
    u[i] = v[i] = u_prev[i] = v_prev[i] = dens[i] = dens_prev[i] = 0.0f;
  }

  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
  glutInitWindowSize(win_x, win_y);
  glutCreateWindow("2D Fluid in a Box - OpenMP Version");
  glutDisplayFunc(display);
  glutReshapeFunc(reshape);
  glutIdleFunc(idle);
  glutMouseFunc(mouseFunc);
  glutMotionFunc(motionFunc);
  glutPassiveMotionFunc(motionFunc);

  glutMainLoop();

  delete[] u;
  delete[] v;
  delete[] u_prev;
  delete[] v_prev;
  delete[] dens;
  delete[] dens_prev;

  return 0;
}
