#include "Generate.h"
#include "read.hpp"
#include <string>
#include <iostream>
#include <fstream>
#include <random>
#include <unordered_map>
#include <vector>
#include <filesystem>

// 定义全局变量，用于存储预加载的N-gram索引
static PrefixIndex prefix_index_10;
static PrefixIndex prefix_index_5;
static PrefixIndex prefix_index_3;
static PrefixIndex prefix_index_2;
static bool is_initialized = false;

// 从文件加载N-gram数据
// 需要添加的辅助函数声明（放在load_ngram_data函数之前）
// 需要添加的辅助函数声明（放在load_ngram_data函数之前）
void save_prefix_index_binary(const PrefixIndex& prefix_index, const std::string& file_path);
NgramCounts load_training_data(const std::string& file_path);

// 从文件加载N-gram数据
static bool load_ngram_data() {
    namespace fs = std::filesystem;
    
    // 定义模型文件路径
    const std::string base_path = "./Model/";
    const std::vector<std::pair<std::string, int>> models = {
        {"Model_n10", 10},
        {"Model_n5", 5},
        {"Model_n3", 3},
        {"Model_n2", 2}
    };
    
    // 检查二进制模型文件是否存在
    bool all_bin_exist = true;
    for (const auto& model : models) {
        std::string bin_path = base_path + model.first + ".bin";
        if (!fs::exists(bin_path)) {
            all_bin_exist = false;
            break;
        }
    }
    
    // 如果二进制模型文件存在，直接加载
    if (all_bin_exist) {
        std::cout << "Loading N-gram models from binary files...\n";
        
        for (const auto& model : models) {
            std::string bin_path = base_path + model.first + ".bin";
            std::ifstream bin_file(bin_path, std::ios::binary);
            
            if (!bin_file) {
                std::cerr << "Error: Failed to open binary file " << bin_path << "\n";
                return false;
            }
            
            // 读取前缀索引大小
            size_t map_size;
            bin_file.read(reinterpret_cast<char*>(&map_size), sizeof(map_size));
            
            PrefixIndex prefix_index;
            for (size_t i = 0; i < map_size; ++i) {
                // 读取前缀
                size_t prefix_len;
                bin_file.read(reinterpret_cast<char*>(&prefix_len), sizeof(prefix_len));
                std::string prefix(prefix_len, '\0');
                bin_file.read(&prefix[0], prefix_len);
                
                // 读取候选项数量
                size_t candidates_size;
                bin_file.read(reinterpret_cast<char*>(&candidates_size), sizeof(candidates_size));
                
                std::vector<std::pair<std::string, int>> candidates;
                for (size_t j = 0; j < candidates_size; ++j) {
                    // 读取后缀
                    size_t suffix_len;
                    bin_file.read(reinterpret_cast<char*>(&suffix_len), sizeof(suffix_len));
                    std::string suffix(suffix_len, '\0');
                    bin_file.read(&suffix[0], suffix_len);
                    
                    // 读取计数
                    int count;
                    bin_file.read(reinterpret_cast<char*>(&count), sizeof(count));
                    
                    candidates.emplace_back(suffix, count);
                }
                
                prefix_index[prefix] = std::move(candidates);
            }
            
            // 根据模型类型分配到对应的全局变量
            if (model.second == 10) {
                prefix_index_10 = std::move(prefix_index);
            } else if (model.second == 5) {
                prefix_index_5 = std::move(prefix_index);
            } else if (model.second == 3) {
                prefix_index_3 = std::move(prefix_index);
            } else if (model.second == 2) {
                prefix_index_2 = std::move(prefix_index);
            }
            
            std::cout << "Loaded " << model.first << " model\n";
        }
        
        std::cout << "N-gram models loaded successfully!\n";
        return true;
    }
    
    // 如果二进制模型文件不存在，检查是否需要重新训练
    std::cout << "Binary model files missing or incomplete, checking training data...\n";
    
    bool need_training = false;
    for (const auto& model : models) {
        std::string txt_path = base_path + model.first + ".txt";
        if (!fs::exists(txt_path)) {
            need_training = true;
            break;
        }
    }
    
    // 如果需要训练，从原始数据重新构建
    if (need_training) {
        std::cout << "Training data not found, regenerating from original text...\n";
        
        // 确保Model目录存在
        if (!fs::exists(base_path)) {
            fs::create_directories(base_path);
        }
        
        // 构建各阶N-gram计数
        for (const auto& model : models) {
            std::cout << "Building " << model.first << " model...\n";
            NgramCounts word_count = build_ngram_counts("Data", model.second);
            
            // 保存文本格式的训练数据
            std::string txt_path = base_path + model.first + ".txt";
            save_training_data(word_count, txt_path);
            std::cout << "Saved text training data to " << txt_path << "\n";
            
            // 构建并保存二进制格式的前缀索引
            PrefixIndex prefix_index = build_prefix_index(word_count, model.second);
            std::string bin_path = base_path + model.first + ".bin";
            save_prefix_index_binary(prefix_index, bin_path);
            std::cout << "Saved binary model to " << bin_path << "\n";
            
            // 存储到对应的全局变量
            if (model.second == 10) {
                prefix_index_10 = std::move(prefix_index);
            } else if (model.second == 5) {
                prefix_index_5 = std::move(prefix_index);
            } else if (model.second == 3) {
                prefix_index_3 = std::move(prefix_index);
            } else if (model.second == 2) {
                prefix_index_2 = std::move(prefix_index);
            }
        }
        
        std::cout << "Model training completed!\n";
        return true;
    }
    
    // 如果已有文本训练数据但没有二进制模型，从文本数据构建并保存二进制模型
    std::cout << "Text training data exists, building binary models...\n";
    
    for (const auto& model : models) {
        std::cout << "Loading " << model.first << " text data...\n";
        
        // 从文本文件构建N-gram计数（需要实现load_training_data函数）
        std::string txt_path = base_path + model.first + ".txt";
        NgramCounts word_count = load_training_data(txt_path);
        
        // 构建并保存二进制格式的前缀索引
        PrefixIndex prefix_index = build_prefix_index(word_count, model.second);
        std::string bin_path = base_path + model.first + ".bin";
        save_prefix_index_binary(prefix_index, bin_path);
        std::cout << "Saved binary model to " << bin_path << "\n";
        
        // 存储到对应的全局变量
        if (model.second == 10) {
            prefix_index_10 = std::move(prefix_index);
        } else if (model.second == 5) {
            prefix_index_5 = std::move(prefix_index);
        } else if (model.second == 3) {
            prefix_index_3 = std::move(prefix_index);
        } else if (model.second == 2) {
            prefix_index_2 = std::move(prefix_index);
        }
    }
    
    std::cout << "Binary models built successfully!\n";
    return true;
}


// 加权随机选择下一个字符
static std::string pick_weighted_next(const PrefixIndex& prefix_index, const std::string& prefix, std::mt19937& gen) {
    auto iter = prefix_index.find(prefix);
    if (iter == prefix_index.end()) {
        return "";
    }
    const auto& candidates = iter->second;
    int totalcount = 0;
    for (const auto& candidate : candidates) {
        totalcount += candidate.second;
    }
    
    int choose_ = std::uniform_int_distribution<>(0, totalcount - 1)(gen);
    int current_count = 0;
    for (const auto& candidate : candidates) {
        current_count += candidate.second;
        if (current_count > choose_) {
            return candidate.first;
        }
    }
    
    return "";
}

// 使用N-gram模型生成文本
static std::string generate_text(const std::string& user_input, int max_length,int n_required) {
    std::string generated_text = user_input;
    const int n = n_required;  // 使用10元模型,找不到就会退化到5元、3元和2元模型
    
    // 检查输入长度
    if (utf8_length(user_input) < 2) {
        generated_text += ", this is a short input, I need more context to generate coherent text.";
        return generated_text;
    }
    
    int dot_count = 0;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::string prefix ;
    
    while (dot_count < 15 && utf8_length(generated_text) < max_length) {
        std::string next_char = "";
        if (n==10&&std::uniform_int_distribution<>(0, 100)(gen)>1){
            prefix = last_utf8_chars(generated_text, 9);
            next_char = pick_weighted_next(prefix_index_10, prefix, gen);
        }
        if ((n>5 && next_char.empty())|| 
        (n==5&&std::uniform_int_distribution<>(0, 100)(gen)>1) ){
            prefix = last_utf8_chars(generated_text, 4);
            next_char = pick_weighted_next(prefix_index_5, prefix, gen);
        }
        if ((n>3 && next_char.empty())||
        (n==3&&std::uniform_int_distribution<>(0, 100)(gen)>1) ) {
            prefix = last_utf8_chars(generated_text, 2);
            next_char = pick_weighted_next(prefix_index_3, prefix, gen);
        }
        if ((n>2 && next_char.empty())|| n==2 ) {
            prefix = last_utf8_chars(generated_text, 1);
            next_char = pick_weighted_next(prefix_index_2, prefix, gen);
        }
        if (next_char.empty()) {
            break;
        }
        
        generated_text += next_char;
        prefix = last_utf8_chars(generated_text, n - 1);
        
        // 统计句子结束符数量
        if (next_char == "。" || next_char == "！" || next_char == "？") {
            dot_count++;
        }
    }
    generated_text+="\n\n(n="+std::to_string(n)+")";
    return generated_text;
}

// 生成函数实现
void Generate(const std::string& prompt, std::string& output, int n) {
    // 初始化N-gram模型（只在第一次调用时执行）
    if (!is_initialized) {
        is_initialized = load_ngram_data();
        if (!is_initialized) {
            output = "Failed to initialize N-gram models, please check training data!";
            return;
        }
    }
    
    if (prompt.empty()) {
        output = "Please enter some content, I will generate text based on N-gram models.";
        return;
    }
    
    // 使用N-gram模型生成文本
    // 注意：参数n在这里可以用于调整生成文本的长度或其他参数
    int max_length = 800;  // 根据n值调整生成文本的最大长度
    output = generate_text(prompt, max_length,n);
}
