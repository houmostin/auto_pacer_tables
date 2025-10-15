from PyQt5 import QtWidgets, QtCore

class Panel(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trainer")
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        self.lbl_state   = QtWidgets.QLabel("state: —")
        self.lbl_answer  = QtWidgets.QLabel("decision: —")
        self.btn_mark    = QtWidgets.QPushButton("Зафиксировать результат (F8)")
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.lbl_state); lay.addWidget(self.lbl_answer); lay.addWidget(self.btn_mark)

    def show_state(self, s): self.lbl_state.setText(s)
    def show_answer(self, a): self.lbl_answer.setText(a)
