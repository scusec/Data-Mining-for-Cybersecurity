# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'normal.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import sys


class Ui_Normal(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Ui_Normal")
        Dialog.resize(572, 345)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(80, 60, 471, 61))
        self.label.setStyleSheet("font: 24pt \"方正姚体\";\n"
        "color: rgb(0, 170, 0);")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(190, 150, 251, 31))
        self.label_2.setStyleSheet("font: 22pt \"方正姚体\";\n"
        "color: rgb(0, 170, 0);")
        self.label_2.setObjectName("label_2")
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(230, 240, 111, 41))
        self.pushButton.setStyleSheet("font: 20pt \"Consolas\";\n"
        "background-color: rgb(255, 170, 0);")
        self.pushButton.setObjectName("pushButton")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "结果"))
        self.label.setText(_translate("Dialog", "正常APK文件，可以放心安装！"))
        self.label_2.setText(_translate("Dialog", "可能性："))
        self.pushButton.setText(_translate("Dialog", "确 定"))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    widget = QtWidgets.QWidget()
    ui = Ui_Normal()
    ui.setupUi(widget)
    widget.show()
    sys.exit(app.exec_())


