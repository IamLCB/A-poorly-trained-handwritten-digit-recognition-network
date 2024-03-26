import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QAction
from PyQt5.QtGui import QImage, QPainter, QPen, QPixmap
from PyQt5.QtCore import Qt, QPoint
from tensorflow.keras.models import load_model
from PIL import Image
from predict import predict_digit


class PaintWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.model = load_model('models/mnist_model.h5')

    def initUI(self):
        self.canvas = QImage(self.size(), QImage.Format_RGB32)
        self.canvas.fill(Qt.white)

        self.drawing = False
        self.brushSize = 20
        self.brushColor = Qt.black
        self.lastPoint = QPoint()

        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu("File")

        clearAction = QAction("Clear", self)
        clearAction.triggered.connect(self.clearCanvas)
        fileMenu.addAction(clearAction)

        predictAction = QAction("Predict", self)
        predictAction.triggered.connect(self.predictDigit)
        fileMenu.addAction(predictAction)

        self.setWindowTitle("Handwriting Board")
        self.setGeometry(100, 100, 600, 600)

    # def mousePressEvent(self, event):
    #     if event.button() == Qt.LeftButton:
    #         self.drawing = True
    #         self.lastPoint = event.pos()
    #
    # def mouseMoveEvent(self, event):
    #     if (event.buttons() & Qt.LeftButton) & self.drawing:
    #         painter = QPainter(self.canvas)
    #         painter.setPen(QPen(self.brushColor, self.brushSize, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    #         painter.drawLine(self.lastPoint, event.pos())
    #         self.lastPoint = event.pos()
    #         self.update()
    #
    # def mouseReleaseEvent(self, event):
    #     if event.button() == Qt.LeftButton:
    #         self.drawing = False
    #
    # def paintEvent(self, event):
    #     canvasPainter = QPainter(self)
    #     canvasPainter.drawImage(self.rect(), self.canvas, self.canvas.rect())

    def clearCanvas(self):
        self.canvas.fill(Qt.white)
        self.update()

    def predictDigit(self):
        # 保存当前画布内容
        temp_img_path = "temp_digit.png"
        # 转换 QImage 到 QPixmap，因为 QPixmap 有方便的缩放方法
        pixmap = QPixmap.fromImage(self.canvas)

        # 缩放 QPixmap 到新尺寸 200x200
        scaled_pixmap = pixmap.scaled(200, 200, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

        # 保存缩放后的 QPixmap 到文件
        scaled_pixmap.save(temp_img_path)
        # 使用 predict.py 中的方法进行预测
        predicted_digit = predict_digit(temp_img_path)
        # 显示预测结果，例如，更新窗口标题
        self.setWindowTitle(f"Handwriting Board - Predicted: {predicted_digit}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = PaintWindow()
    main.show()
    sys.exit(app.exec_())
