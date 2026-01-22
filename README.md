# Ngram 语言模型项目

这是一个基于 Qt 的 Ngram 语言模型项目，用于中文文本的处理和生成。

## 1. 编译指南

### 1.1 环境准备

1. **下载并安装 Qt Creator**
   - 访问 Qt 官方网站：https://www.qt.io/download
   - 下载适合你操作系统的 Qt 安装包（建议使用 Qt 6.x 版本）
   - 安装时确保勾选以下组件：
     - Qt Creator IDE
     - Qt 核心库
     - MinGW-w64 编译器（如果你的系统是 Windows）

2. **配置环境变量**
   - **MinGW-w64**：确保 MinGW-w64 的 `bin` 目录已添加到系统环境变量 `PATH` 中
   - **qmake**：确保 Qt 的 `bin` 目录已添加到系统环境变量 `PATH` 中（通常位于 Qt 安装目录下的 `Tools/mingwxx_x64/bin` 或类似路径）

### 1.2 编译项目

使用以下命令在项目根目录下编译：

```bash
qmake NgramChat.pro
mingw32-make
```

### 1.3 运行项目

编译成功后，使用以下命令运行程序：

```bash
./release/NgramChat.exe
```

或者在 Qt Creator 中直接点击运行按钮。

## 2. 代码实现架构和逻辑

详见 `Doc` 文件夹中的文档（文档正在编写中）。
