#ifndef READ_HPP
#define READ_HPP

#include <filesystem>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

/**
 * @file read.hpp
 * @brief N-gram模型的核心数据结构和函数声明
 * 
 * 该头文件定义了N-gram模型所需的数据结构和函数接口，
 * 包括UTF-8字符串处理、N-gram计数构建、前缀索引生成等功能。
 */

/**
 * @brief 二元语法计数映射类型
 * 
 * 用于存储二元语法（Bigram）及其出现次数
 */
using BigramCounts = std::unordered_map<std::string, int>;

/**
 * @brief N元语法计数映射类型
 * 
 * 用于存储N元语法（N-gram）及其出现次数
 */
using NgramCounts = std::unordered_map<std::string, int>;

/**
 * @brief 前缀索引映射类型
 * 
 * 用于存储前缀到后续字符列表的映射，每个后续字符包含其出现次数
 */
using PrefixIndex = std::unordered_map<std::string, std::vector<std::pair<std::string, int>>>;

/**
 * @brief 截取UTF-8字符串的子串
 * 
 * @param str 原始UTF-8字符串
 * @param start 起始字符索引（以字符数计，非字节数）
 * @param length 要截取的字符数
 * @return 截取得到的UTF-8子串
 */
std::string utf8_substr(const std::string& str, int start, int length);

/**
 * @brief 获取UTF-8字符串的字符数
 * 
 * @param str UTF-8字符串
 * @return 字符串的字符数（非字节数）
 */
int utf8_length(const std::string& str);

/**
 * @brief 获取UTF-8字符串的最后一个字符
 * 
 * @param str UTF-8字符串
 * @return 字符串的最后一个字符
 */
std::string last_utf8_char(const std::string& str);

/**
 * @brief 获取UTF-8字符串的最后count个字符
 * 
 * @param str UTF-8字符串
 * @param count 要获取的字符数
 * @return 字符串的最后count个字符
 */
std::string last_utf8_chars(const std::string& str, int count);

/**
 * @brief 确保控制台支持UTF-8输出
 * 
 * 在Windows系统上设置控制台代码页为UTF-8
 */
void ensure_utf8_console();

/**
 * @brief 从指定目录的文本文件中构建N-gram计数
 * 
 * @param dir 包含文本文件的目录路径
 * @param n N-gram的n值（如2表示二元语法）
 * @return 包含N-gram及其出现次数的映射
 */
NgramCounts build_ngram_counts(const std::filesystem::path& dir, int n);

/**
 * @brief 从N-gram计数构建前缀索引
 * 
 * 将N-gram计数转换为前缀索引，用于快速查找给定前缀的后续字符
 * 
 * @param ngrams N-gram计数映射
 * @param n N-gram的n值
 * @return 前缀索引映射
 */
PrefixIndex build_prefix_index(const NgramCounts& ngrams, int n);

/**
 * @brief 将N-gram训练数据保存到文件
 * 
 * @param ngrams N-gram计数映射
 * @param output_path 输出文件路径
 */
void save_training_data(const NgramCounts& ngrams, const std::filesystem::path& output_path);

/**
 * @brief 从前缀索引中选择出现频率最高的后续字符
 * 
 * @param prefix_index 前缀索引映射
 * @param prefix 要查找的前缀
 * @return 出现频率最高的后续字符
 */
std::string choose_most_frequent(const PrefixIndex& prefix_index, const std::string& prefix);
/**
 * @brief 将前缀索引保存为二进制文件
 * 
 * @param prefix_index 前缀索引映射
 * @param file_path 输出文件路径
 */
void save_prefix_index_binary(const PrefixIndex& prefix_index, const std::string& file_path);

/**
 * @brief 从文本文件加载训练数据
 * 
 * @param file_path 输入文件路径
 * @return N-gram计数映射
 */
NgramCounts load_training_data(const std::string& file_path);
#endif
