cmake_minimum_required(VERSION 3.16)
project(football-fan-ex VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Homebrew on Apple Silicon installs packages under /opt/homebrew
if(APPLE AND EXISTS "/opt/homebrew")
    list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")
endif()

find_package(SFML 3 REQUIRED COMPONENTS Graphics Window System)

add_library(football_fan_ex
    src/football_fan_ex.cpp
)
target_include_directories(football_fan_ex
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(football-fan-ex
    src/main.cpp
)
target_link_libraries(football-fan-ex
    PRIVATE
        football_fan_ex
        SFML::Graphics
)
