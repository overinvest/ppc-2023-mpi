// Copyright 2023 Kulaev Eugene
#include <gtest/gtest.h>
#include <iostream>
#include <boost/mpi.hpp>

#include "./increasing_contrast.h"

TEST(Parallel_Contrast_Enha, Test_not_elems) {
    boost::mpi::communicator comm;
    int rankProc = comm.rank();

    int new_min = 0;
    int new_max = 255;

    std::vector<int> image{};
    std::vector copy_image(image);

    ParallelStretching(&image, new_min, new_max);

    if (rankProc == 0) {
        ASSERT_EQ(image, copy_image);
    }
}

TEST(Parallel_Contrast_Enha, Test_only_zero) {
    boost::mpi::communicator comm;
    int rankProc = comm.rank();

    int new_min = 0;
    int new_max = 255;

    std::vector<int> image(100, 0);
    std::vector copy_image(image);

    ParallelStretching(&image, new_min, new_max);

    if (rankProc == 0) {
        int global_min = SeqmetionMin(&copy_image);
        int global_max = SeqmetionMax(&copy_image);

        Stretching(&copy_image, global_min, global_max, new_min, new_max);

        ASSERT_EQ(image, copy_image);
    }
}

TEST(Parallel_Contrast_Enha, Test_rand_elems) {
    boost::mpi::communicator comm;
    int rankProc = comm.rank();

    int new_min = 0;
    int new_max = 255;

    std::vector<int> image(625);
    for (auto &elem : image) elem = generateRandomNumbers(0, 255);

    std::vector copy_image(image);

    ParallelStretching(&image, new_min, new_max);

    if (rankProc == 0) {
        int global_min = SeqmetionMin(&copy_image);
        int global_max = SeqmetionMax(&copy_image);

        Stretching(&copy_image, global_min, global_max, new_min, new_max);

        ASSERT_EQ(image, copy_image);
    }
}

TEST(Parallel_Contrast_Enha, Test_half_zero) {
    boost::mpi::communicator comm;
    int rankProc = comm.rank();

    int new_min = 0;
    int new_max = 255;

    std::vector<int> image{0, 1, 0,
                            1, 1, 1,
                            0, 1, 0};
    std::vector copy_image(image);

    ParallelStretching(&image, new_min, new_max);

    if (rankProc == 0) {
        int global_min = SeqmetionMin(&copy_image);
        int global_max = SeqmetionMax(&copy_image);

        Stretching(&copy_image, global_min, global_max, new_min, new_max);

        ASSERT_EQ(image, copy_image);
    }
}

TEST(Parallel_Contrast_Enha, Test_base) {
    boost::mpi::communicator comm;
    int rankProc = comm.rank();

    int new_min = 0;
    int new_max = 255;

    std::vector<int> image{1, 5, 5, 6, 7, 8, 8, 8,
                            10, 12, 15, 16, 17, 18, 255, 0};
    std::vector copy_image(image);

    ParallelStretching(&image, new_min, new_max);

    if (rankProc == 0) {
        int global_min = SeqmetionMin(&copy_image);
        int global_max = SeqmetionMax(&copy_image);

        Stretching(&copy_image, global_min, global_max, new_min, new_max);

        ASSERT_EQ(image, copy_image);
    }
}


int main(int argc, char** argv) {
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
