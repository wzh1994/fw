import sys
from PyQt5.QtWidgets import QApplication, QWidget, QToolTip, QPushButton, QDesktopWidget, QComboBox, QTextBrowser, QFileDialog, QMessageBox, QLabel
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import Qt
from interface import interface, FireworkType
from predict import predict


#if __name__ == '__main__':
#	m_name = '2.avi'
#	args = predict(m_name, FireworkType.DualMixture)
#	print(args)
#	args = interface.call_mfc(FireworkType.DualMixture, args, m_name)
#	print(args)


def test(widget):
    print(widget.type_combo.currentData())


class FireworkUI(QWidget):
    def __init__(self):
        super().__init__()
        self.movie_name = None
        self.args = None
        self.setUI()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def set_type_combo(self):
        txt = QTextBrowser(self)
        txt.append('请选择烟花类型：')
        txt.resize(230, 30)
        txt.move(10, 0)
        self.type_combo = QComboBox(self, minimumWidth=100)
        self.type_combo.addItem('球状烟花', FireworkType.Normal)
        self.type_combo.addItem('两次爆炸', FireworkType.MultiExplosion)
        self.type_combo.addItem('两个球状混合', FireworkType.DualMixture)
        self.type_combo.setCurrentIndex(-1)
        self.type_combo.resize(self.type_combo.sizeHint())
        self.type_combo.move(110,5)

    def _get_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self,
            "选取文件",
            "./movies",
            "AVI Files (*.avi);;MP4 Files (*.mp4);;MGEG Files (*.mpeg);;3GP Files (*.3gp)")
        self.movie_name = file_name.split('/')[-1]
        self.open_file_txt.clear()
        self.open_file_txt.setAlignment(Qt.AlignRight)
        self.open_file_txt.append(self.movie_name)

    def set_open_file(self):
        self.open_file_txt = QTextBrowser(self)
        self.open_file_txt.resize(230, 30)
        self.open_file_txt.move(10, 40)
        self.open_file_btn = QPushButton("打开文件", self)
        self.open_file_btn.setToolTip("打开文件")
        self.open_file_btn.clicked.connect(self._get_file)
        self.open_file_btn.move(30, 45)
        self.open_file_btn.resize(self.open_file_btn.sizeHint())

    def _call_mfc(self):
        if self.movie_name is None or self.type_combo.currentIndex() == -1:
            message = '请选择烟花类型！ ' if self.type_combo.currentIndex() == -1 else ''
            message += '请打开烟花文件！ ' if self.movie_name is None else ''
            QMessageBox.warning(self, "警告", message, QMessageBox.Yes)
        else:
            self.analysis_btn.setDisabled(True)
            fw_type = self.type_combo.currentData()
            self.args = predict(self.movie_name, fw_type)
            self.args = interface.call_mfc(fw_type, self.args, self.movie_name)
            self.analysis_btn.setEnabled(True)

    def set_analysis_button(self):
        self.analysis_btn = QPushButton("分析视频", self)
        self.analysis_btn.clicked.connect(self._call_mfc)
        self.analysis_btn.resize(100, 20)
        self.analysis_btn.move(30, 80)

    def _call_play(self):
        if self.movie_name is None or self.type_combo.currentIndex() == -1 or self.args is None:
            message = '请选择烟花类型！ ' if self.type_combo.currentIndex() == -1 else ''
            message += '请打开烟花文件！ ' if self.movie_name is None else ''
            message += '请先分析烟花视频！ ' if self.args is None else ''
            QMessageBox.warning(self, "警告", message, QMessageBox.Yes)
        else:
            self.play_btn.setDisabled(True)
            fw_type = self.type_combo.currentData()
            interface.play(fw_type, self.args)
            self.play_btn.setEnabled(True)

    def set_play_button(self):
        self.play_btn = QPushButton("播放视频", self)
        self.play_btn.clicked.connect(self._call_play)
        self.play_btn.resize(100, 20)
        self.play_btn.move(130, 80)

    def setUI(self):
        self.setFixedSize(250, 150)
        self.setWindowTitle("Firework")
        self.set_analysis_button()
        self.set_open_file()
        self.set_type_combo()
        self.set_play_button()
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = FireworkUI()
    sys.exit(app.exec_())
