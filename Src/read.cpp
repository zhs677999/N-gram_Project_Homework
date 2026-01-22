#include "read.hpp"

#include <fstream>
#include <iostream>
#include <unordered_map>

#ifdef _WIN32
#include <windows.h>
#endif

/**
 * @file read.cpp
 * @brief N-gram模型的核心功能实现
 * 
 * 该文件实现了read.hpp中声明的所有函数，包括UTF-8字符串处理、
 * N-gram计数构建、前缀索引生成等功能。
 */

/**
 * @brief 截取UTF-8字符串的子串
 * 
 * 该函数正确处理UTF-8编码的字符串，按字符索引而非字节索引进行截取
 * 
 * @param str 原始UTF-8字符串
 * @param start 起始字符索引（以字符数计，非字节数）
 * @param length 要截取的字符数
 * @return 截取得到的UTF-8子串
 */
std::string utf8_substr(const std::string& str, int start, int length) {
    if (str.empty() || length <= 0) {
        return "";
    }

    int i = 0;  // 字节索引
    int char_count = 0;
    int start_index = -1, end_index = -1;

    // 遍历字符串的每个字节，计算字符数
    while (i < static_cast<int>(str.length())) {
        if (char_count == start) {
            start_index = i;  // 记录起始字节位置
        }
        if (char_count == start + length) {
            end_index = i;    // 记录结束字节位置
            break;
        }

        // 判断当前UTF-8字符的字节长度
        unsigned char c = static_cast<unsigned char>(str[i]);
        int char_len = 0;
        if (c <= 0x7F) char_len = 1;            // ASCII字符（单字节）
        else if ((c & 0xE0) == 0xC0) char_len = 2;  // 双字节UTF-8字符
        else if ((c & 0xF0) == 0xE0) char_len = 3;  // 三字节UTF-8字符
        else if ((c & 0xF8) == 0xF0) char_len = 4;  // 四字节UTF-8字符

        i += char_len;
        char_count++;
    }

    // 如果超过字符串末尾，结束位置设为字符串末尾
    if (end_index == -1) end_index = i;
    if (start_index == -1) return "";

    // 截取字节子串并返回
    return str.substr(start_index, end_index - start_index);
}

/**
 * @brief 获取UTF-8字符串的字符数
 * 
 * 该函数正确计算UTF-8编码字符串的字符数（非字节数）
 * 
 * @param str UTF-8字符串
 * @return 字符串的字符数
 */
int utf8_length(const std::string& str) {
    int i = 0;
    int char_count = 0;
    while (i < static_cast<int>(str.length())) {
        unsigned char c = static_cast<unsigned char>(str[i]);
        // 根据UTF-8编码规则判断字符的字节长度
        if (c <= 0x7F) {
            i += 1;  // ASCII字符（单字节）
        } else if ((c & 0xE0) == 0xC0) {
            i += 2;  // 双字节UTF-8字符
        } else if ((c & 0xF0) == 0xE0) {
            i += 3;  // 三字节UTF-8字符
        } else if ((c & 0xF8) == 0xF0) {
            i += 4;  // 四字节UTF-8字符
        } else {
            i += 1;  // 无效字符，按单字节处理
        }
        char_count++;
    }
    return char_count;
}

/**
 * @brief 获取UTF-8字符串的最后一个字符
 * 
 * @param str UTF-8字符串
 * @return 字符串的最后一个字符
 */
std::string last_utf8_char(const std::string& str) {
    int total_chars = utf8_length(str);
    if (total_chars == 0) {
        return "";
    }
    return utf8_substr(str, total_chars - 1, 1);
}

/**
 * @brief 获取UTF-8字符串的最后count个字符
 * 
 * @param str UTF-8字符串
 * @param count 要获取的字符数
 * @return 字符串的最后count个字符
 */
std::string last_utf8_chars(const std::string& str, int count) {
    int total_chars = utf8_length(str);
    if (total_chars == 0 || count <= 0) {
        return "";
    }

    if (count >= total_chars) {
        return str;
    }

    return utf8_substr(str, total_chars - count, count);
}

/**
 * @brief 确保控制台支持UTF-8输出
 * 
 * 在Windows系统上设置控制台输入输出代码页为UTF-8，
 * 确保中文等非ASCII字符能正确显示
 */
void ensure_utf8_console() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
}

/**
 * @brief 从指定目录的文本文件中构建N-gram计数
 * 
 * 遍历指定目录中的所有.txt文件，读取文本内容并构建N-gram计数
 * 
 * @param dir 包含文本文件的目录路径
 * @param n N-gram的n值（如2表示二元语法，3表示三元语法）
 * @return 包含N-gram及其出现次数的映射
 */
NgramCounts build_ngram_counts(const std::filesystem::path& dir, int n) {
    NgramCounts word_count;
    
    // 遍历目录中的所有文件
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        // 只处理.txt文件
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            std::cout << "Processing file: " << entry.path().filename() << "\n";

            std::ifstream file(entry.path());
            if (!file) {
                std::cerr << "Failed to open file: " << entry.path() << "\n";
                continue;
            }
            
            // 读取文件内容
            std::string line;
            std::string content;
            while (std::getline(file, line)) {
                content += line;
            }
            
            // 构建N-gram计数
            const int char_length = utf8_length(content);
            for (int i = 0; i <= char_length - n; i++) {
                std::string ngram = utf8_substr(content, i, n);
                if (!ngram.empty()) {
                    word_count[ngram]++;
                }
            }
        }
    }

    return word_count;
}

/**
 * @brief 从N-gram计数构建前缀索引
 * 
 * 将N-gram计数转换为前缀索引，用于快速查找给定前缀的后续字符
 * 
 * @param ngrams N-gram计数映射
 * @param n N-gram的n值
 * @return 前缀索引映射
 */
PrefixIndex build_prefix_index(const NgramCounts& ngrams, int n) {
    PrefixIndex prefix_index;
    
    // 遍历所有N-gram
    for (const auto& pair : ngrams) {
        // 提取前缀（前n-1个字符）
        std::string prefix = utf8_substr(pair.first, 0, n - 1);
        // 提取后缀（最后1个字符）
        std::string suffix = utf8_substr(pair.first, n - 1, 1);
        // 将后缀及其出现次数添加到前缀对应的列表中
        prefix_index[prefix].emplace_back(suffix, pair.second);
    }
    return prefix_index;
}

/**
 * @brief 将N-gram训练数据保存到文件
 * 
 * 将N-gram计数保存到指定文件，每行格式为："N-gram 出现次数"
 * 
 * @param ngrams N-gram计数映射
 * @param output_path 输出文件路径
 */
void save_training_data(const NgramCounts& ngrams, const std::filesystem::path& output_path) {
    std::ofstream outfile(output_path);
    for (const auto& pair : ngrams) {
        outfile << pair.first << " " << pair.second << "\n";
    }
}

/**
 * @brief 从前缀索引中选择出现频率最高的后续字符
 * 
 * @param prefix_index 前缀索引映射
 * @param prefix 要查找的前缀
 * @return 出现频率最高的后续字符，如果没有找到则返回空字符串
 */
std::string choose_most_frequent(const PrefixIndex& prefix_index, const std::string& prefix) {
    auto iter = prefix_index.find(prefix);
    if (iter == prefix_index.end()) {
        return "";
    }

    const auto& candidates = iter->second;
    auto best = candidates.front();
    
    // 找到出现频率最高的后续字符
    for (const auto& candidate : candidates) {
        if (candidate.second > best.second) {
            best = candidate;
        }
    }
    return best.first;
}
/**
 * @brief 将前缀索引保存为二进制文件
 * 
 * @param prefix_index 前缀索引映射
 * @param file_path 输出文件路径
 */
void save_prefix_index_binary(const PrefixIndex& prefix_index, const std::string& file_path) {
    std::ofstream bin_file(file_path, std::ios::binary);
    if (!bin_file) {
        std::cerr << "Error: Failed to create binary file " << file_path << "\n";
        return;
    }
    
    // 写入映射大小
    size_t map_size = prefix_index.size();
    bin_file.write(reinterpret_cast<const char*>(&map_size), sizeof(map_size));
    
    // 写入每个前缀及其候选项
    for (const auto& entry : prefix_index) {
        // 写入前缀长度和前缀内容
        size_t prefix_len = entry.first.size();
        bin_file.write(reinterpret_cast<const char*>(&prefix_len), sizeof(prefix_len));
        bin_file.write(entry.first.c_str(), prefix_len);
        
        // 写入候选项数量
        size_t candidates_size = entry.second.size();
        bin_file.write(reinterpret_cast<const char*>(&candidates_size), sizeof(candidates_size));
        
        // 写入每个候选项
        for (const auto& candidate : entry.second) {
            // 写入后缀长度和后缀内容
            size_t suffix_len = candidate.first.size();
            bin_file.write(reinterpret_cast<const char*>(&suffix_len), sizeof(suffix_len));
            bin_file.write(candidate.first.c_str(), suffix_len);
            
            // 写入计数
            bin_file.write(reinterpret_cast<const char*>(&candidate.second), sizeof(candidate.second));
        }
    }
}

/**
 * @brief 从文本文件加载训练数据
 * 
 * @param file_path 输入文件路径
 * @return N-gram计数映射
 */
NgramCounts load_training_data(const std::string& file_path) {
    NgramCounts ngrams;
    std::ifstream infile(file_path);
    
    if (!infile) {
        std::cerr << "Error: Failed to open file " << file_path << "\n";
        return ngrams;
    }
    
    std::string line;
    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        
        // 查找最后一个空格，分隔N-gram和计数
        size_t last_space = line.find_last_of(' ');
        if (last_space == std::string::npos) continue;
        
        std::string ngram = line.substr(0, last_space);
        std::string count_str = line.substr(last_space + 1);
        
        try {
            int count = std::stoi(count_str);
            ngrams[ngram] = count;
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to parse line: " << line << "\n";
        }
    }
    
    return ngrams;
}