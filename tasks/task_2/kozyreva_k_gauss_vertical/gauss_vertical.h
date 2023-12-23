// Copyright 2023 Kozyreva Ekaterina
#ifndef TASKS_TASK_2_KOZYREVA_K_GAUSS_VERTICAL_GAUSS_VERTICAL_H_
#define TASKS_TASK_2_KOZYREVA_K_GAUSS_VERTICAL_GAUSS_VERTICAL_H_
#include <vector>

using Matrix = std::vector<double>;

void printMatrix(const Matrix& vec, int rows, int cols);
Matrix RandomMatrix(int rows, int cols);
Matrix SequentialGauss(const Matrix& matrix, int rows, int cols,
                        const Matrix& vec, int vector_size);
Matrix ParallelGauss(const Matrix& matrix, int rows, int cols,
                        const Matrix& vec, int vector_size);

#endif  // TASKS_TASK_2_KOZYREVA_K_GAUSS_VERTICAL_GAUSS_VERTICAL_H_
