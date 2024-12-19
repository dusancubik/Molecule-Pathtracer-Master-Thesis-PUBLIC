/*
 * -----------------------------------------------------------------------------
 *  Author: Dusan Cubik
 *  Project: Physically Based Renderer for WebGPU
 *  Institution: Masaryk University
 *  Date: 16. 12. 2024
 *  File: Timer.hpp
 *
 *  Description:
 *  The Timer class provides functionality for measuring time intervals.
 * -----------------------------------------------------------------------------
 */
#pragma once
#include <chrono>
struct TimerValues {
    int minutes;
    int seconds;
    int miliseconds;
};
class Timer {
public:
    void start();

    void stop();

    void reset();

    TimerValues getCurrentTime();

private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    bool running = false;
};
