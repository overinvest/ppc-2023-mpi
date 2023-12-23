// Copyright 2023 Borovkov Sergey
#include <gtest/gtest.h>
#include <mpi.h>
#include <iostream>
#include "./alg_gauss_vert.h"

TEST(Vertical_Gauss_MPI, Test_height_priority_1) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int* image = nullptr;
  const int width = 100, height = 200;

  if (rank == 0) {
    image = getRandomImage(width, height);
  }
  double start = MPI_Wtime();
  int* new_image_parallel = getParallelGauss(image, width, height);
  double end = MPI_Wtime();

  if (rank == 0) {
    double time1 = end - start;
    start = MPI_Wtime();
    int* reference_new_image = getSequentialGauss(image, width, height);
    end = MPI_Wtime();
    double time2 = end - start;
    std::cout << time1 << " " << time2 << " " << time2 / time1 << std::endl;
    for (int i = 0; i < (width - 2) * (height - 2); i += (width - 2)) {
      ASSERT_EQ(new_image_parallel[i], reference_new_image[i]);
    }
  }
}

TEST(Vertical_Gauss_MPI, Test_width_priority_1) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int* image = nullptr;
  const int width = 300, height = 100;

  if (rank == 0) {
    image = getRandomImage(width, height);
  }
  double start = MPI_Wtime();
  int* new_image_parallel = getParallelGauss(image, width, height);
  double end = MPI_Wtime();

  if (rank == 0) {
    double time1 = end - start;
    start = MPI_Wtime();
    int* reference_new_image = getSequentialGauss(image, width, height);
    end = MPI_Wtime();
    double time2 = end - start;
    std::cout << time1 << " " << time2 << " " << time2 / time1 << std::endl;
    for (int i = 0; i < (width - 2) * (height - 2); i += (width - 2)) {
      ASSERT_EQ(new_image_parallel[i], reference_new_image[i]);
    }
  }
}

TEST(Vertical_Gauss_MPI, Test_equal) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int* image = nullptr;
  const int width = 500, height = 500;

  if (rank == 0) {
    image = getRandomImage(width, height);
  }
  double start = MPI_Wtime();
  int* new_image_parallel = getParallelGauss(image, width, height);
  double end = MPI_Wtime();

  if (rank == 0) {
    double time1 = end - start;
    start = MPI_Wtime();
    int* reference_new_image = getSequentialGauss(image, width, height);
    end = MPI_Wtime();
    double time2 = end - start;
    std::cout << time1 << " " << time2 << " " << time2 / time1 << std::endl;
    for (int i = 0; i < (width - 2) * (height - 2); i += (width - 2)) {
      ASSERT_EQ(new_image_parallel[i], reference_new_image[i]);
    }
  }
}

TEST(Vertical_Gauss_MPI, Test_height_priority_2) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int* image = nullptr;
  const int width = 512, height = 1123;

  if (rank == 0) {
    image = getRandomImage(width, height);
  }
  double start = MPI_Wtime();
  int* new_image_parallel = getParallelGauss(image, width, height);
  double end = MPI_Wtime();

  if (rank == 0) {
    double time1 = end - start;
    start = MPI_Wtime();
    int* reference_new_image = getSequentialGauss(image, width, height);
    end = MPI_Wtime();
    double time2 = end - start;
    std::cout << time1 << " " << time2 << " " << time2 / time1 << std::endl;
    for (int i = 0; i < (width - 2) * (height - 2); i += (width - 2)) {
      ASSERT_EQ(new_image_parallel[i], reference_new_image[i]);
    }
  }
}

TEST(Vertical_Gauss_MPI, Test_width_priority_2) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int* image = nullptr;
  const int width = 1230, height = 512;

  if (rank == 0) {
    image = getRandomImage(width, height);
  }
  double start = MPI_Wtime();
  int* new_image_parallel = getParallelGauss(image, width, height);
  double end = MPI_Wtime();

  if (rank == 0) {
    double time1 = end - start;
    start = MPI_Wtime();
    int* reference_new_image = getSequentialGauss(image, width, height);
    end = MPI_Wtime();
    double time2 = end - start;
    std::cout << time1 << " " << time2 << " " << time2 / time1 << std::endl;
    for (int i = 0; i < (width - 2) * (height - 2); i += (width - 2)) {
      ASSERT_EQ(new_image_parallel[i], reference_new_image[i]);
    }
  }
}

int main(int argc, char **argv) {
    int result_code = 0;

    ::testing::InitGoogleTest(&argc, argv);
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();

    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
        MPI_Abort(MPI_COMM_WORLD, -1);
    result_code = RUN_ALL_TESTS();
    MPI_Finalize();

    return result_code;
}
