// Copyright 2023 Bodrov Daniil
#include "task_3/bodrov_d_batcher_sort/batcher_sort.h"

void batcherMerge(int* arr, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    std::vector<int> L(n1);
    std::vector<int> R(n2);

    for (int i = 0; i < n1; ++i)
        L[i] = arr[l + i];
    for (int j = 0; j < n2; ++j)
        R[j] = arr[m + 1 + j];

    int i = 0, j = 0, k = l;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            ++i;
        } else {
            arr[k] = R[j];
            ++j;
        }
        ++k;
    }

    while (i < n1) {
        arr[k] = L[i];
        ++i;
        ++k;
    }

    while (j < n2) {
        arr[k] = R[j];
        ++j;
        ++k;
    }
}

void batcherSort(int* arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;

        batcherSort(arr, l, m);
        batcherSort(arr, m + 1, r);

        batcherMerge(arr, l, m, r);
    }
}

void parallelBatcherSort(int* arr, int l, int r) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size <= 1) {
        batcherSort(arr, l, r);
        return;
    }

    int localSize = (r - l + 1) / size;
    int remainder = (r - l + 1) % size;
    int* localArr = new int[localSize + (rank < remainder ? 1 : 0)];

    int* sendcounts = new int[size];
    int* displs = new int[size];
    for (int i = 0; i < size; ++i) {
        sendcounts[i] = localSize;
        displs[i] = i * localSize;
    }
    for (int i = 0; i < remainder; ++i) {
        ++sendcounts[i];
    }

    // Scatter the array among processes
    MPI_Scatterv(&arr[l], sendcounts, displs, MPI_INT, localArr,
        localSize + (rank < remainder ? 1 : 0), MPI_INT, 0, MPI_COMM_WORLD);

    // Perform local sort
    batcherSort(localArr, 0, localSize + (rank < remainder ? 1 : 0) - 1);

    // Gather sorted subarrays from all processes
    int mergedSize = (r - l + 1);
    int* mergedArr = new int[mergedSize];
    MPI_Gatherv(localArr, localSize + (rank < remainder ? 1 : 0), MPI_INT,
        mergedArr, sendcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Merge sorted subarrays
        batcherSort(mergedArr, 0, mergedSize - 1);

        // Copy sorted elements back to the original array
        for (int i = 0; i < mergedSize; ++i)
            arr[l + i] = mergedArr[i];
    }

    delete[] localArr;
    delete[] sendcounts;
    delete[] displs;
    delete[] mergedArr;
}

