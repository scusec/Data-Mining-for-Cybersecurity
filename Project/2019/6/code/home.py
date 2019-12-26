# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'home.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
import sys
import os
from normal import Ui_Normal
from malicious import Ui_Malicious
from testNaveBayes import testNB


class Normal(QtWidgets.QDialog, Ui_Normal):
    def __init__(self):
        super(Normal, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)


class Malicious(QtWidgets.QDialog, Ui_Malicious):
    def __init__(self):
        super(Malicious, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)

class Ui_Form(QtWidgets.QWidget):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(636, 424)
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(440, 190, 75, 31))
        self.pushButton.setStyleSheet("font: 12pt \"Consolas\";")
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(Form)
        self.pushButton_2.setGeometry(QtCore.QRect(260, 290, 111, 41))
        self.pushButton_2.setStyleSheet("color: rgb(255, 170, 0);\n"
        "font: 16pt \"Consolas\";\n"
        "color: rgb(0, 0, 0);\n"
        "background-color: rgb(255, 170, 0);")
        self.pushButton_2.setObjectName("pushButton_2")
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(90, 191, 51, 31))
        self.label.setStyleSheet("font: 12pt \"Consolas\";")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(73, 90, 541, 51))
        self.label_2.setStyleSheet("color: rgb(170, 85, 0);\n"
        "font: 28pt \"方正姚体\";\n"
        "font: 36pt \"方正姚体\";\n"
        "font: 28pt \"方正舒体\";")
        self.label_2.setObjectName("label_2")
        self.lineEdit = QtWidgets.QLineEdit(Form)
        self.lineEdit.setGeometry(QtCore.QRect(150, 190, 241, 31))
        self.lineEdit.setStyleSheet("font: 12pt \"Consolas\";")
        self.lineEdit.setObjectName("lineEdit")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "主页"))
        self.pushButton.setText(_translate("Form", "选择文件"))
        self.pushButton.clicked.connect(self.select_file)
        self.pushButton_2.setText(_translate("Form", "开始检测"))
        self.pushButton_2.clicked.connect(self.detection)
        self.label.setText(_translate("Form", "文件名："))
        self.label_2.setText(_translate("Form", "请选择你要检测的APK文件吧！"))

    def select_file(self):
        fileName, fileType = QFileDialog.getOpenFileName(self, "选择文件", "./sample", "All Files(*);;APK Files (*.apk)")
        file = open("filepath.txt", "w")
        file.write(fileName)
        file.close()
        self.lineEdit.setText(fileName)

    def detection(self):
        path = self.lineEdit.text()
        if path != "":
            result, pro = testNB(path)
            front, back = str(pro[0][result[0]]).split(".")
            probalility = float(front + "." + back[0:2])
            print(result)
            print(pro)
            if result[0] == 1:
                dialog = Normal()
                dialog.label_2.setText("可能性:%.2f" % probalility)
                dialog.pushButton.clicked.connect(dialog.close)
                dialog.show()
                dialog.exec_()
            else:
                dialog = Malicious()
                dialog.label_3.setText("可能性:%.2f" % probalility)
                dialog.pushButton.clicked.connect(self.deleteFile)
                dialog.pushButton_2.clicked.connect(dialog.close)
                dialog.show()
                dialog.exec_()
            self.lineEdit.clear()
        else:
            pass

    def deleteFile(self):
        fp = open("filepath.txt", "r")
        path = fp.read()
        os.remove(path)
        fp.close()
        QMessageBox.about(self, "提示", "文件成功删除！")



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    widget = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(widget)
    widget.show()
    sys.exit(app.exec_())

