#include "../../include/Utils/Timer.hpp"

void Timer::start() {
    start_time = std::chrono::high_resolution_clock::now();
    running = true;
}

void Timer::stop() {
    if (running) {
        end_time = std::chrono::high_resolution_clock::now();
        running = false;
    }
}

void Timer::reset() {
    running = false;
    start();
}

TimerValues Timer::getCurrentTime() {
    auto elapsed_time = running
        ? std::chrono::high_resolution_clock::now() - start_time
        : end_time - start_time;

    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count();
    int minutes = milliseconds / 60000;
    milliseconds %= 60000;
    int seconds = milliseconds / 1000;
    milliseconds %= 1000;

    return TimerValues(minutes,seconds,milliseconds);
}