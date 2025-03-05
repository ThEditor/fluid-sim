CXX = g++
CXXFLAGS = -fopenmp -lGL -lGLU -lglut -lm
MPI_CXX = mpic++
MPI_FLAGS = -lGL -lGLU -lglut -lm

all: create_out_dir fluid_box_mpi fluid_box_omp

create_out_dir:
	mkdir -p out

fluid_box_mpi: openmpi/fluid_box.cpp
	$(MPI_CXX) openmpi/fluid_box.cpp -o out/fluid_box_mpi $(MPI_FLAGS)

fluid_box_omp: openmp/fluid_box.cpp
	$(CXX) openmp/fluid_box.cpp -o out/fluid_box_omp $(CXXFLAGS)

clean:
	rm -f fluid_box_mpi fluid_box_omp
