cmake_minimum_required(VERSION 3.14)
project(bfloat16_impl VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -Wpedantic)

include_directories(include)

add_library(bfloat16 INTERFACE)
target_include_directories(bfloat16 INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.4.0
)
FetchContent_MakeAvailable(Catch2)

enable_testing()
add_executable(bfloat16_tests tests/bfloat16_tests.cpp)
target_link_libraries(bfloat16_tests PRIVATE bfloat16 Catch2::Catch2WithMain)

include(Catch)
catch_discover_tests(bfloat16_tests)

install(TARGETS bfloat16
    EXPORT bfloat16-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/ DESTINATION include)

install(EXPORT bfloat16-targets
    FILE bfloat16-targets.cmake
    NAMESPACE bfloat16::
    DESTINATION lib/cmake/bfloat16
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/bfloat16-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/bfloat16-config.cmake
    INSTALL_DESTINATION lib/cmake/bfloat16
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/bfloat16-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/bfloat16-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/bfloat16-config-version.cmake
    DESTINATION lib/cmake/bfloat16
)
