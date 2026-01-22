#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <limits>
#include "Generate.h"
#include "Monitor.h"

using namespace std;
namespace fs = std::filesystem;

// UTF-8文件读取函数
std::string read_utf8_file(const fs::path& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        cerr << "Error: Cannot open input file " << file_path << endl;
        return "";
    }
    
    // 获取文件大小
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    if (file_size == 0) {
        cerr << "Error: Input file is empty" << endl;
        return "";
    }
    
    // 读取文件内容
    std::string content(file_size, '\0');
    file.read(&content[0], file_size);
    
    // 移除可能的UTF-8 BOM (Byte Order Mark)
    if (file_size >= 3 && 
        static_cast<unsigned char>(content[0]) == 0xEF &&
        static_cast<unsigned char>(content[1]) == 0xBB &&
        static_cast<unsigned char>(content[2]) == 0xBF) {
        content = content.substr(3);
    }
    
    return content;
}

// UTF-8文件写入函数
bool write_utf8_file(const fs::path& file_path, const std::string& content) {
    std::ofstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        cerr << "Error: Cannot open output file " << file_path << endl;
        return false;
    }
    
    file.write(content.c_str(), content.size());
    return file.good();
}

// 将报告追加到输出内容中
std::string append_report_to_output(const std::string& original_output, 
                                    const std::string& user_input,
                                    double totalS, 
                                    double avgMsPerWord,
                                    const fs::path& output_path) {
    
    std::stringstream report_stream;
    
    // 添加分隔线
    report_stream << "\n\n" << std::string(35, '=') << std::endl;
    report_stream << "Generation Report:" << std::endl;
    report_stream << "-----------------------------------" << std::endl;
    report_stream << "Total time:     " << std::fixed << std::setprecision(3) << totalS << " s" << std::endl;
    report_stream << "Avg time per word: " << std::fixed << std::setprecision(2) << avgMsPerWord << " ms/word" << std::endl;
    report_stream << "Input string:   " << user_input << std::endl;
    report_stream << "Output location: " << output_path.string() << std::endl;
    report_stream << std::string(35, '=') << std::endl;
    
    return original_output + report_stream.str();
}

// 解析输入行（处理可能的空白行和UTF-8字符）
bool parse_input_lines(const std::string& content, std::string& user_input, int& max_length, int& n_required) {
    std::istringstream iss(content);
    std::string line;
    std::vector<std::string> lines;
    
    // 按行分割，跳过空行
    while (std::getline(iss, line)) {
        // 移除回车符
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!line.empty()) {
            lines.push_back(line);
        }
    }
    
    if (lines.size() < 3) {
        cerr << "Error: Not enough lines in input file. Found " << lines.size() << " lines, need 3" << endl;
        for (size_t i = 0; i < lines.size(); ++i) {
            cerr << "Line " << i+1 << ": [" << lines[i] << "]" << endl;
        }
        return false;
    }
    
    user_input = lines[0];
    
    // 使用stringstream解析数字，避免中文字符干扰
    std::istringstream num_stream1(lines[1]);
    if (!(num_stream1 >> max_length)) {
        cerr << "Error: Cannot parse max_length from: [" << lines[1] << "]" << endl;
        return false;
    }
    
    std::istringstream num_stream2(lines[2]);
    if (!(num_stream2 >> n_required)) {
        cerr << "Error: Cannot parse n_required from: [" << lines[2] << "]" << endl;
        return false;
    }
    
    return true;
}

int main() {
    Timer timer;
    std::atomic<bool> isMonitoring(true);

    // 1. 启动内存监控线程
    std::thread monitorWorker(MemoryMonitor::displayMemoryRealTime, std::ref(isMonitoring));

    // 设置输入输出文件夹路径
    fs::path io_dir = "io_files";
    if (!fs::exists(io_dir)) {
        if (!fs::create_directory(io_dir)) {
            cerr << "Error: Cannot create directory " << io_dir << endl;
            return 1;
        }
        cout << "Created directory: " << io_dir << endl;
    }

    fs::path input_path = io_dir / "input.txt";
    fs::path output_path = io_dir / "output.txt";

    // 2. 从 input.txt 读取输入（UTF-8 编码）
    string user_input;
    int max_length, n_required;
    
    // 检查输入文件是否存在
    if (!fs::exists(input_path)) {
        cerr << "Error: Input file does not exist: " << input_path << endl;
        cerr << "Please create " << input_path << " with the following format:" << endl;
        cerr << "Line 1: Starting string" << endl;
        cerr << "Line 2: Maximum length" << endl;
        cerr << "Line 3: Number of generations" << endl;
        
        // 创建示例输入文件
        std::ofstream example_file(input_path);
        if (example_file) {
            example_file << "今天天气真好\n100\n50\n";
            example_file.close();
            cout << "Created example input file at: " << input_path << endl;
            cout << "Please edit it and run the program again." << endl;
        }
        
        isMonitoring = false;
        if (monitorWorker.joinable()) monitorWorker.join();
        
        cout << "\nPress Enter to close..." << endl;
        cin.get();
        return 1;
    }
    
    // 读取UTF-8编码的输入文件
    cout << "Reading input from: " << input_path << endl;
    std::string input_content = read_utf8_file(input_path);
    if (input_content.empty()) {
        cerr << "Error: Input file is empty or cannot be read" << endl;
        return 1;
    }
    
    // 解析输入内容
    cout << "Parsing input content..." << endl;
    if (!parse_input_lines(input_content, user_input, max_length, n_required)) {
        cerr << "Error: Input file format is incorrect. Format should be:" << endl;
        cerr << "Line 1: Starting string" << endl;
        cerr << "Line 2: Maximum length" << endl;
        cerr << "Line 3: Number of generations" << endl;
        
        isMonitoring = false;
        if (monitorWorker.joinable()) monitorWorker.join();
        
        cout << "\nPress Enter to close..." << endl;
        cin.get();
        return 1;
    }
    
    // 删去以下这行：
    // cout << "Successfully read input:" << endl;
    // cout << "Starting string: " << user_input << endl;
    
    cout << "Maximum length: " << max_length << endl;
    cout << "Number of generations: " << n_required << endl;

    // 3. 开始计时并执行生成函数
    string out_put;
    
    // 删去这行：
    // cout << "\nGenerating text..." << endl;
    
    timer.start();
    Generate(user_input, out_put, n_required);
    timer.end();
    
    cout << "Generation completed. Output size: " << out_put.size() << " bytes" << endl;
    if (!out_put.empty()) {
        cout << "First 100 chars of output: " << out_put.substr(0, std::min(100, (int)out_put.size())) << "..." << endl;
    }

    // 4. 计算与输出结果
    double totalS = timer.get_total_time();
    double totalMS = timer.get_total_time_ms();
    double avgMsPerWord = (max_length > 0) ? (totalMS / max_length) : 0;

    // 5. 将报告信息追加到输出内容中
    string final_output = append_report_to_output(out_put, user_input, totalS, avgMsPerWord, output_path);
    
    // 6. 将完整输出（包括报告）写入 output.txt（UTF-8 编码）
    cout << "\nWriting output to: " << output_path << endl;
    if (!write_utf8_file(output_path, final_output)) {
        cerr << "Error: Failed to write output file" << endl;
    } else {
        cout << "Output successfully written to: " << output_path << endl;
        
        // 验证文件是否写入成功
        if (fs::exists(output_path)) {
            auto file_size = fs::file_size(output_path);
            cout << "Output file size: " << file_size << " bytes" << endl;
        }
    }

    // 7. 在控制台显示报告
    std::cout << "\n\n" << std::string(35, '=') << std::endl;
    std::cout << "Generation Report:" << std::endl;
    std::cout << "-----------------------------------" << std::endl;
    std::cout << "Total time:     " << std::fixed << std::setprecision(3) << totalS << " s" << std::endl;
    std::cout << "Avg time per word: " << std::fixed << std::setprecision(2) << avgMsPerWord << " ms/word" << std::endl;
    std::cout << "Input string:   " << user_input << std::endl;
    std::cout << "Output location: " << output_path << std::endl;
    std::cout << std::string(35, '=') << std::endl;

    // 8. 停止监控线程
    isMonitoring = false;
    if (monitorWorker.joinable()) {
        monitorWorker.join();
        cout << "\nMonitoring thread stopped." << endl;
    }

    std::cout << "\nPress Enter to close..." << std::endl;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cin.get();

    return 0;
}