// Copyright 2023 Bodrov Daniil
#ifndef MAX_ROW_VALUES_H
#define MAX_ROW_VALUES_H

#include <mpi.h>
#include <vector>

std::vector<int> FindMaxRowValues(const std::vector<int>& matrix, int n);
std::vector<int> FindMaxRowValuesPar(const std::vector<int>& matrix, int n);


#endif  // MAX_ROW_VALUES_H
