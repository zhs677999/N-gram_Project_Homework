#pragma once
#ifndef MONITOR_H
#define MONITOR_H

#include <chrono>
#include <atomic>
#include <windows.h>

// 计时器类
class Timer {
public:
    void start();
    void end();
    double get_total_time();
    double get_total_time_ms();

private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
    bool is_running_ = false;
};

// 内存与状态实时显示类
class MemoryMonitor {
public:
    // 静态函数，由后台线程调用
    static void displayMemoryRealTime(std::atomic<bool>& running);
};

#endif