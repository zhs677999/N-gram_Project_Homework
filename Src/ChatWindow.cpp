#include "ChatWindow.h"
#include "Generate.h"
#include <QDateTime>
#include <QScrollBar>
#include <QThread>
#include <QTimer>

// 聊天气泡实现
ChatBubble::ChatBubble(const QString& text, MessageType type, QWidget* parent)
    : QWidget(parent), m_text(text), m_type(type)
{
    initUI();
    setText(text);
}

ChatBubble::~ChatBubble()
{
}

void ChatBubble::initUI()
{
    m_layout = new QHBoxLayout(this);
    m_layout->setContentsMargins(10, 10, 10, 10);
    m_layout->setSpacing(10);

    // 创建头像
    m_avatar = new QLabel(this);
    m_avatar->setFixedSize(40, 40);
    
    if (m_type == MessageType::User) {
        m_avatar->setStyleSheet("border-radius: 20px; background-color: #4CAF50; border: none;");
        m_layout->addStretch();
        m_layout->addWidget(m_avatar);
    } else {
        m_avatar->setStyleSheet("border-radius: 20px; background-color: #c0c0c0; border: none;");
        m_layout->addWidget(m_avatar);
        m_layout->addStretch();
    }

    // 创建内容标签
    m_content = new QLabel(this);
    m_content->setWordWrap(true);
    m_content->setStyleSheet("font-size: 14px; font-family: 'Microsoft YaHei', 'SimHei', sans-serif; color: #333; padding: 0; margin: 0;");
    m_content->setAlignment(Qt::AlignLeft | Qt::AlignTop);
    // 设置文本可选中
    m_content->setTextInteractionFlags(Qt::TextSelectableByMouse);
    m_content->setCursor(Qt::IBeamCursor);

    // 创建聊天气泡容器
    QWidget* bubbleContainer = new QWidget(this);
    m_bubbleLayout = new QVBoxLayout(bubbleContainer);
    // 设置气泡容器的布局属性
    m_bubbleLayout->setContentsMargins(15, 15, 15, 15); // 增加外边距，放大圆角矩形
    m_bubbleLayout->setAlignment(Qt::AlignLeft | Qt::AlignTop); // 文字左对齐顶部对齐
    m_bubbleLayout->addWidget(m_content);
    // 确保内容标签能够自适应大小
    m_content->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
    m_content->setMaximumWidth(this->parentWidget() ? (this->parentWidget()->width() * 3 / 5 - 40) : 360);
    
    // 设置气泡容器样式和文字颜色
    if (m_type == MessageType::User) {
        bubbleContainer->setStyleSheet("background-color: #4CAF50; border-radius: 18px; padding: 20px;");
        m_content->setStyleSheet("font-size: 14px; font-family: 'Microsoft YaHei', 'SimHei', sans-serif; color: white; padding: 0; margin: 0;");
        m_layout->insertWidget(1, bubbleContainer);
    } else {
        bubbleContainer->setStyleSheet("background-color: #f0f0f0; border-radius: 18px; padding: 20px;");
        m_content->setStyleSheet("font-size: 14px; font-family: 'Microsoft YaHei', 'SimHei', sans-serif; color: #333; padding: 0; margin: 0;");
        m_layout->insertWidget(1, bubbleContainer);
    }

    this->setLayout(m_layout);
}

void ChatBubble::setText(const QString& text)
{
    // 使用QFontMetrics计算文本宽度，实现更精确的换行
    QFontMetrics fm(m_content->font());
    int maxWidth = this->parentWidget() ? (this->parentWidget()->width() * 3 / 5 - 40) : 360; // 减去气泡内边距
    
    QString wrappedText;
    QString line;
    
    for (const QChar& c : text) {
        if (c == '\n') {
            // 保留原有的换行符
            wrappedText += line + "\n";
            line.clear();
            continue;
        }
        
        QString testLine = line + c;
        int lineWidth = fm.horizontalAdvance(testLine);
        
        if (lineWidth > maxWidth) {
            // 当行宽超过最大宽度时换行
            wrappedText += line + "\n";
            line = c;
        } else {
            line = testLine;
        }
    }
    
    // 添加最后一行
    if (!line.isEmpty()) {
        wrappedText += line;
    }
    
    m_text = wrappedText;
    m_content->setText(wrappedText);
    this->updateGeometry();
}

QString ChatBubble::text() const
{
    return m_text;
}

void ChatBubble::paintEvent(QPaintEvent* event)
{
    QWidget::paintEvent(event);
}

QSize ChatBubble::sizeHint() const
{
    int maxWidth = this->parentWidget() ? (this->parentWidget()->width() * 3 / 5) : 400;
    QFontMetrics fm(m_content->font());
    QRect rect = fm.boundingRect(QRect(0, 0, maxWidth - 32, 1000), Qt::TextWordWrap, m_text);
    
    int width = rect.width() + 32;
    int height = rect.height() + 24;
    
    return QSize(width, height);
}

QSize ChatBubble::minimumSizeHint() const
{
    return sizeHint();
}

// 不再使用基于空格的换行逻辑，已在setText中实现每30字符换行

// 聊天窗口实现
ChatWindow::ChatWindow(QWidget* parent)
    : QWidget(parent)
{
    initUI();
    loadStyleSheet();
}

ChatWindow::~ChatWindow()
{
}

void ChatWindow::initUI()
{
    // 设置窗口属性
    this->setWindowTitle("AI Chat");
    this->setMinimumSize(600, 800);
    this->setStyleSheet("background-color: #f8f9fa;");

    // 创建主布局
    m_mainLayout = new QVBoxLayout(this);
    m_mainLayout->setContentsMargins(0, 0, 0, 0);
    m_mainLayout->setSpacing(0);

    // 创建标题栏
    QWidget* titleBar = new QWidget(this);
    titleBar->setFixedHeight(60);
    titleBar->setStyleSheet("background-color: white; border-bottom: 1px solid #e0e0e0;");
    
    QHBoxLayout* titleLayout = new QHBoxLayout(titleBar);
    titleLayout->setContentsMargins(20, 0, 20, 0);
    
    m_titleLabel = new QLabel("小学生作文生成器", this);
    m_titleLabel->setStyleSheet("font-size: 18px; font-weight: bold; color: #333;");
    titleLayout->addWidget(m_titleLabel);
    titleLayout->addStretch();
    
    m_mainLayout->addWidget(titleBar);

    // 创建消息区域
    m_scrollArea = new QScrollArea(this);
    m_scrollArea->setWidgetResizable(true);
    m_scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    m_scrollArea->setStyleSheet("QScrollArea { border: none; background-color: #f8f9fa; }" 
                              "QScrollBar:vertical { width: 6px; background: transparent; }" 
                              "QScrollBar::handle:vertical { background: #c0c0c0; border-radius: 3px; }" 
                              "QScrollBar::handle:vertical:hover { background: #a0a0a0; }" 
                              "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }" 
                              "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }");
    
    m_messagesWidget = new QWidget(this);
    m_messagesWidget->setStyleSheet("background-color: #f8f9fa;");
    m_messagesLayout = new QVBoxLayout(m_messagesWidget);
    m_messagesLayout->setContentsMargins(20, 20, 20, 20);
    m_messagesLayout->setSpacing(15);
    m_messagesLayout->addStretch();
    
    m_scrollArea->setWidget(m_messagesWidget);
    m_mainLayout->addWidget(m_scrollArea, 1);

    // 创建输入区域
    QWidget* inputArea = new QWidget(this);
    inputArea->setFixedHeight(80);
    inputArea->setStyleSheet("background-color: white; border-top: 1px solid #e0e0e0;");
    
    m_inputLayout = new QHBoxLayout(inputArea);
    m_inputLayout->setContentsMargins(20, 15, 20, 15);
    m_inputLayout->setSpacing(15);
    
    m_inputEdit = new QLineEdit(this);
    m_inputEdit->setPlaceholderText("请输入开头...");
    m_inputEdit->setStyleSheet("border: 1px solid #e0e0e0; border-radius: 20px; padding: 0 16px; font-size: 14px; height: 50px; background-color: #f8f9fa;");
    m_inputEdit->setFixedHeight(50);
    connect(m_inputEdit, &QLineEdit::returnPressed, this, &ChatWindow::onReturnPressed);
    
    // 创建下拉框按钮
    m_nComboBox = new QComboBox(this);
    m_nComboBox->addItem("n=2");
    m_nComboBox->addItem("n=3");
    m_nComboBox->addItem("n=5");
    m_nComboBox->addItem("n=10");
    m_nComboBox->setStyleSheet("QComboBox { border: 1px solid #e0e0e0; border-radius: 20px; padding: 0 12px; font-size: 14px; height: 50px; background-color: white; text-align: center; box-shadow: none; } QComboBox::drop-down { border: none; width: 20px; } QComboBox::down-arrow { image: none; } QComboBox QAbstractItemView { border: 1px solid #e0e0e0; outline: none; box-shadow: none; background-color: white; } QComboBox QAbstractItemView::item { padding: 10px 15px; border: 1px solid #e0e0e0; text-align: center; margin: 0; } QComboBox QAbstractItemView::item:hover { background-color: #4CAF50; color: white; border: 1px solid #4CAF50; }");
    m_nComboBox->setEditable(false);
    m_nComboBox->setView(new QListView());
    m_nComboBox->view()->window()->setWindowFlags(Qt::Popup | Qt::FramelessWindowHint | Qt::NoDropShadowWindowHint);
    m_nComboBox->view()->window()->setAttribute(Qt::WA_TranslucentBackground);
    m_nComboBox->setFixedHeight(50);
    
    m_sendButton = new QPushButton("发送", this);
    m_sendButton->setStyleSheet("background-color: #4CAF50; color: white; border: none; border-radius: 20px; padding: 0 24px; font-size: 14px; font-weight: bold; height: 50px;");
    m_sendButton->setFixedHeight(50);
    connect(m_sendButton, &QPushButton::clicked, this, &ChatWindow::onSendButtonClicked);
    
    m_inputLayout->addWidget(m_inputEdit, 1);
    m_inputLayout->addWidget(m_nComboBox);
    m_inputLayout->addWidget(m_sendButton);
    
    m_mainLayout->addWidget(inputArea);

    this->setLayout(m_mainLayout);
}

void ChatWindow::loadStyleSheet()
{
    // 可以在这里加载外部样式表
}

void ChatWindow::resizeEvent(QResizeEvent* event)
{
    QWidget::resizeEvent(event);
    // 更新所有气泡的大小
    for (int i = 0; i < m_messagesLayout->count() - 1; ++i) {
        QLayoutItem* item = m_messagesLayout->itemAt(i);
        if (ChatBubble* bubble = qobject_cast<ChatBubble*>(item->widget())) {
            bubble->updateGeometry();
        }
    }
}

void ChatWindow::onSendButtonClicked()
{
    QString input = m_inputEdit->text().trimmed();
    if (!input.isEmpty()) {
        processUserInput(input);
    }
}

void ChatWindow::onReturnPressed()
{
    QString input = m_inputEdit->text().trimmed();
    if (!input.isEmpty()) {
        processUserInput(input);
    }
}

void ChatWindow::processUserInput(const QString& input)
{
    // 添加用户消息
    addMessage(input, MessageType::User);
    
    // 清空输入框
    m_inputEdit->clear();
    
    // 获取下拉框选择的n值
    int nValue = 1; // 默认值
    int selectedIndex = m_nComboBox->currentIndex();
    switch(selectedIndex) {
        case 0: nValue = 2; break;
        case 1: nValue = 3; break;
        case 2: nValue = 5; break;
        case 3: nValue = 10; break;
        default: nValue = 10; break;
    }
    
    // 调用Generate函数生成回复
    std::string outputStr;
    Generate(input.toStdString(), outputStr, nValue);
    QString output = QString::fromStdString(outputStr);
    
    // 添加AI回复
    addMessage(output, MessageType::AI);
}

void ChatWindow::addMessage(const QString& text, MessageType type)
{
    ChatBubble* bubble = new ChatBubble(text, type, m_messagesWidget);
    m_messagesLayout->insertWidget(m_messagesLayout->count() - 1, bubble);
    scrollToBottom();
}

void ChatWindow::scrollToBottom()
{
    QTimer::singleShot(100, [this]() {
        m_scrollArea->verticalScrollBar()->setValue(m_scrollArea->verticalScrollBar()->maximum());
    });
}
