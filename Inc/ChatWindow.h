#ifndef CHATWINDOW_H
#define CHATWINDOW_H

#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QScrollArea>
#include <QTextBrowser>
#include <QListWidget>
#include <QListWidgetItem>
#include <QPainter>
#include <QPen>
#include <QBrush>
#include <QEvent>
#include <QFontMetrics>
#include <QFile>
#include <QTextStream>
#include <QComboBox>
#include <QListView>

// 聊天消息类型枚举
enum class MessageType {
    User,
    AI
};

// 聊天气泡组件
class ChatBubble : public QWidget {
    Q_OBJECT
public:
    explicit ChatBubble(const QString& text, MessageType type, QWidget* parent = nullptr);
    ~ChatBubble() override;

    void setText(const QString& text);
    QString text() const;

protected:
    void paintEvent(QPaintEvent* event) override;
    QSize sizeHint() const override;
    QSize minimumSizeHint() const override;

private:
    QString m_text;
    MessageType m_type;
    QLabel* m_avatar;
    QLabel* m_content;
    QHBoxLayout* m_layout;
    QVBoxLayout* m_bubbleLayout;

    void initUI();
    QString wrapText(const QString& text, int maxWidth);
};

// 聊天窗口主类
class ChatWindow : public QWidget {
    Q_OBJECT
public:
    explicit ChatWindow(QWidget* parent = nullptr);
    ~ChatWindow() override;

protected:
    void resizeEvent(QResizeEvent* event) override;

private slots:
    void onSendButtonClicked();
    void onReturnPressed();
    void addMessage(const QString& text, MessageType type);
    void processUserInput(const QString& input);

private:
    QVBoxLayout* m_mainLayout;
    QScrollArea* m_scrollArea;
    QWidget* m_messagesWidget;
    QVBoxLayout* m_messagesLayout;
    QHBoxLayout* m_inputLayout;
    QLineEdit* m_inputEdit;
    QPushButton* m_sendButton;
    QLabel* m_titleLabel;
    QComboBox* m_nComboBox;

    void initUI();
    void loadStyleSheet();
    void scrollToBottom();
};

#endif // CHATWINDOW_H