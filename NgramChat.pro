# Ngram Chat Application
# Qt Project File

QT += widgets core

# 项目名称
TARGET = NgramChat
TEMPLATE = app

# 源文件路径
SOURCES += Src/main.cpp \
           Src/ChatWindow.cpp \
           Src/Generate.cpp \
           Src/read.cpp

# 头文件路径
HEADERS += Inc/ChatWindow.h \
           Inc/Generate.h \
           Inc/read.hpp

# 包含路径
INCLUDEPATH += $$PWD/Inc

# 资源文件（如果有）
# RESOURCES += res.qrc

# 目标文件输出目录
DESTDIR = $$PWD/Build

# 中间文件目录
OBJECTS_DIR = $$PWD/Build/obj
MOC_DIR = $$PWD/Build/moc
UI_DIR = $$PWD/Build/ui
RCC_DIR = $$PWD/Build/rcc