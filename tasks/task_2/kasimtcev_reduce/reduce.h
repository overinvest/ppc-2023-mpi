// Copyright 2023 Kasimtcev Roman
#ifndef TASKS_TASK_2_KASIMTCEV_REDUCE_REDUCE_H_
#define TASKS_TASK_2_KASIMTCEV_REDUCE_REDUCE_H_

#include <mpi.h>

struct proc_tree_props {
  int parent = -1;
  int children[2] = { -1, -1 };
};

void Calculate(int Rank, int Num, proc_tree_props* props, int* path = NULL, int* path_size = NULL);

int mpi_reduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype type, MPI_Op op, int root, MPI_Comm comm);

#endif  // TASKS_TASK_2_KASIMTCEV_REDUCE_REDUCE_H_
