# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'malicious.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import sys

class Ui_Malicious(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Ui_Malicious")
        Dialog.resize(572, 347)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(180, 30, 231, 51))
        self.label.setStyleSheet("font: 28pt \"方正姚体\";\n"
        "font: 36pt \"方正姚体\";\n"
        "color: rgb(170, 0, 0);")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(70, 100, 451, 51))
        self.label_2.setStyleSheet("color: rgb(170, 0, 0);\n"
        "font: 24pt \"方正姚体\";")
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(190, 170, 191, 41))
        self.label_3.setStyleSheet("color: rgb(170, 0, 0);\n"
        "font: 22pt \"方正姚体\";")
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(180, 240, 201, 41))
        self.label_4.setStyleSheet("font: 12pt \"Consolas\";")
        self.label_4.setObjectName("label_4")
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(164, 290, 81, 31))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(300, 290, 81, 31))
        self.pushButton_2.setObjectName("pushButton_2")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "结果"))
        self.label.setText(_translate("Dialog", "警告！！！"))
        self.label_2.setText(_translate("Dialog", "恶意APK文件，建议勿安装！！！"))
        self.label_3.setText(_translate("Dialog", "可能性："))
        self.label_4.setText(_translate("Dialog", "需要直接删除掉这个文件？"))
        self.pushButton.setText(_translate("Dialog", "是"))
        self.pushButton_2.setText(_translate("Dialog", "否"))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    widget = QtWidgets.QWidget()
    ui = Ui_Malicious()
    ui.setupUi(widget)
    widget.show()
    sys.exit(app.exec_())

