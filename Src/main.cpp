#include <QApplication>
#include "ChatWindow.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    
    // 设置应用程序样式
    a.setStyle("Fusion");
    
    // 创建聊天窗口
    ChatWindow w;
    w.show();
    
    return a.exec();
}
