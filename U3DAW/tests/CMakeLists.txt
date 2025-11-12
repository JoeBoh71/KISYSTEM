# Google Test setup
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)

add_executable(u3daw_tests
    unit/test_main.cpp
    unit/test_filters.cpp
)

target_link_libraries(u3daw_tests
    gtest
    gtest_main
    u3daw_tep
)

add_test(NAME U3DAW_Tests COMMAND u3daw_tests)
