#include "Monitor.h"
#include <psapi.h>
#include <iostream>
#include <iomanip>
#include <thread>

#pragma comment(lib, "psapi.lib")

// --- Timer 实现 ---
void Timer::start() {
    start_time_ = std::chrono::high_resolution_clock::now();
    is_running_ = true;
}

void Timer::end() {
    if (is_running_) {
        end_time_ = std::chrono::high_resolution_clock::now();
        is_running_ = false;
    }
}

double Timer::get_total_time() {
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time_ - start_time_);
    return duration.count() / 1e9;
}

double Timer::get_total_time_ms() {
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time_ - start_time_);
    return duration.count() / 1e6;
}

// --- MemoryMonitor 实现 ---
void MemoryMonitor::displayMemoryRealTime(std::atomic<bool>& running) {
    while (running) {
        PROCESS_MEMORY_COUNTERS pmc;
        if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
            double memMB = pmc.WorkingSetSize / (1024.0 * 1024.0);
            std::cout << "\r[Real-time Monitoring] Current Memory Usage: "
                << std::fixed << std::setprecision(2) << memMB << " MB    " << std::flush;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}