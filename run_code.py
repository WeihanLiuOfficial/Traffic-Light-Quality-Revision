import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

from main import detect_lights, compare_boards, run_testing_board
# -------------------------
# PyQt GUI
# -------------------------
class BoardTester(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LED Board Tester")
        self.setGeometry(100, 100, 900, 700)

        self.golden_img_path = None
        self.test_img_path = None

        # Widgets
        self.label = QLabel("Load golden & test board images to begin.")
        self.label.setAlignment(Qt.AlignCenter)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        self.image_label.setScaledContents(False)  # we’ll handle scaling manually
        self.image_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )

        btn_golden = QPushButton("Load Golden Board")
        btn_golden.clicked.connect(self.load_golden)

        btn_test = QPushButton("Load Test Board")
        btn_test.clicked.connect(self.load_test)

        btn_run = QPushButton("Run Test")
        btn_run.clicked.connect(self.run_test)

        # Layouts
        hbox = QHBoxLayout()
        hbox.addWidget(btn_golden)
        hbox.addWidget(btn_test)
        hbox.addWidget(btn_run)

        vbox = QVBoxLayout()
        vbox.addWidget(self.label)
        vbox.addWidget(self.image_label)
        vbox.addLayout(hbox)

        container = QWidget()
        container.setLayout(vbox)
        self.setCentralWidget(container)

    def load_golden(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Golden Board", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.golden_img_path = path
            self.label.setText(f"Golden board loaded: {path}")

    def load_test(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Test Board", "",
                                              "Images (*.png *.jpg *.jpeg)")
        if path:
            self.test_img_path = path
            self.label.setText(f"Test board loaded: {path}")

            # Load and show test image immediately
            img = cv2.imread(path)
            if img is not None:
                self.current_img = img
                self.display_image(img)

    def display_image(self, img):
        """Display a cv2 image (BGR) scaled to the QLabel size."""
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        if self.image_label.width() > 0 and self.image_label.height() > 0:
            pixmap = pixmap.scaled(
                self.image_label.width(),
                self.image_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        self.image_label.setPixmap(pixmap)

    def resizeEvent(self, event):
        if hasattr(self, "current_img") and self.current_img is not None:
            self.display_image(self.current_img)
        super().resizeEvent(event)

    def run_test(self):
        if not self.golden_img_path or not self.test_img_path:
            QMessageBox.warning(self, "Error", "Please load both golden and test board images.")
            return

        golden_lights = detect_lights(self.golden_img_path)
        test_lights = detect_lights(self.test_img_path)

        img = cv2.imread(self.test_img_path)
        passed, marked_img = compare_boards(golden_lights, test_lights, img)

        # Store and display
        self.current_img = marked_img
        self.display_image(marked_img)

        # Update label text
        if passed:
            self.label.setText("Board passes ✅")
        else:
            self.label.setText("Board fails ❌")


# -------------------------
# Run App
# -------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BoardTester()
    window.show()
    sys.exit(app.exec_())
