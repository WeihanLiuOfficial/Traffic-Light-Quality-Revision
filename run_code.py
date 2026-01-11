import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox, QSizePolicy,
    QGroupBox, QListWidget, QProgressBar, QSplitter, QFrame, QSpinBox,
    QDoubleSpinBox, QFormLayout, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QToolBar, QAction, QStatusBar
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QFont, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QRectF

from main import detect_lights, compare_boards

# -----------------------------
# Stylesheet (Industrial Dark)
# -----------------------------
DARK_STYLESHEET = """
QMainWindow {
    background-color: #2d2d2d;
    color: #ffffff;
}
QWidget {
    background-color: #2d2d2d;
    color: #e0e0e0;
    font-family: "Segoe UI", sans-serif;
    font-size: 10pt;
}
QGroupBox {
    border: 1px solid #4d4d4d;
    border-radius: 5px;
    margin-top: 30px;
    padding-top: 25px;
    font-weight: bold;
    color: #007acc;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 5px;
}
QPushButton {
    background-color: #007acc;
    color: white;
    border: none;
    padding: 15px 20px;
    border-radius: 4px;
    font-weight: bold;
    font-size: 11pt;
}
QPushButton:hover {
    background-color: #0098ff;
}
QPushButton:pressed {
    background-color: #005c99;
}
QPushButton:disabled {
    background-color: #4d4d4d;
    color: #808080;
}
QListWidget {
    background-color: #1e1e1e;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
}
QLabel#SuccessLabel {
    color: #4caf50;
    font-weight: bold;
    font-size: 14pt;
}
QLabel#FailLabel {
    color: #f44336;
    font-weight: bold;
    font-size: 14pt;
}
QProgressBar {
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    text-align: center;
    background-color: #1e1e1e;
}
QProgressBar::chunk {
    background-color: #007acc;
}
QSpinBox, QDoubleSpinBox {
    background-color: #1e1e1e;
    border: 1px solid #3d3d3d;
    padding: 4px;
    border-radius: 3px;
}
"""

# -----------------------------
# Custom Image Viewer
# -----------------------------
class ImageViewer(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        
        # Configuration
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.NoFrame)
        self.setBackgroundBrush(QColor("#1e1e1e"))

    def set_image(self, cv_img):
        if cv_img is None:
            return
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor

        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor

        self.scale(zoom_factor, zoom_factor)

# -----------------------------
# Golden Loader Thread
# -----------------------------
class GoldenLoaderThread(QThread):
    finished = pyqtSignal(list)

    def __init__(self, path, params):
        super().__init__()
        self.path = path
        self.params = params

    def run(self):
        # Detect lights using default or passed parameters
        # Note: We use the parameters for brightness/resize if relevant
        lights = detect_lights(
            self.path,
            brightness_ratio=self.params.get('brightness', 0.6),
            resize_factor=0.3
        )
        self.finished.emit(lights)

# -----------------------------
# Worker Thread
# -----------------------------
class AnalysisWorker(QThread):
    finished = pyqtSignal(object)  # Emits (passed, marked_img, details, golden_lights)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    log = pyqtSignal(str)

    def __init__(self, golden_path, test_path, cached_golden_lights, params):
        super().__init__()
        self.golden_path = golden_path
        self.test_path = test_path
        self.cached_golden_lights = cached_golden_lights
        self.params = params

    def run(self):
        try:
            self.progress.emit(10)
            self.log.emit("Starting analysis...")
            
            # 1. Get Golden Lights
            if self.cached_golden_lights:
                self.log.emit("Using cached golden board data.")
                golden_lights = self.cached_golden_lights
            else:
                self.log.emit(f"Detecting lights in golden board: {self.golden_path}")
                golden_lights = detect_lights(
                    self.golden_path, 
                    brightness_ratio=self.params['brightness'],
                    resize_factor=0.3
                )
            
            self.progress.emit(40)
            
            # 2. Get Test Lights
            self.log.emit(f"Detecting lights in test board: {self.test_path}")
            test_lights = detect_lights(
                self.test_path,
                brightness_ratio=self.params['brightness'],
                resize_factor=0.3
            )
            
            self.progress.emit(70)
            
            # 3. Compare
            self.log.emit("Comparing boards...")
            img = cv2.imread(self.test_path)
            passed, marked_img, details = compare_boards(
                golden_lights, 
                test_lights, 
                img,
                pos_tolerance=self.params['pos_tol'],
                color_tolerance=self.params['col_tol']
            )
            
            self.progress.emit(100)
            self.log.emit("Analysis complete.")
            
            # Emits tuple: (passed, marked_img, details, golden_lights_for_cache)
            self.finished.emit((passed, marked_img, details, golden_lights))
            
        except Exception as e:
            self.error.emit(str(e))

# -----------------------------
# Main Application
# -----------------------------
class BoardTester(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LED Inspection Suite Pro")
        self.resize(1200, 800)
        self.setStyleSheet(DARK_STYLESHEET)
        
        self.golden_img_path = None
        self.test_img_path = None
        self.current_img = None
        
        # Caching
        self.cached_golden_lights = None
        self.last_golden_path = None
        self.worker = None
        self.golden_loader = None

        self.setup_ui()

    def setup_ui(self):
        # Central Widget & Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # --- LEFT SIDEBAR (Controls) ---
        left_panel = QFrame()
        left_panel.setFixedWidth(380)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Image Loading Group
        grp_load = QGroupBox("Image Loading")
        vbox_load = QVBoxLayout()
        
        self.btn_golden = QPushButton("Load Golden Board")
        self.btn_golden.clicked.connect(self.load_golden)
        self.lbl_golden_status = QLabel("No file selected")
        self.lbl_golden_status.setStyleSheet("color: #808080;")
        self.lbl_golden_status.setWordWrap(True)
        
        self.btn_test = QPushButton("Load Test Board")
        self.btn_test.clicked.connect(self.load_test)
        self.lbl_test_status = QLabel("No file selected")
        self.lbl_test_status.setStyleSheet("color: #808080;")
        self.lbl_test_status.setWordWrap(True)
        
        vbox_load.addWidget(self.btn_golden)
        vbox_load.addWidget(self.lbl_golden_status)
        vbox_load.addSpacing(10)
        vbox_load.addWidget(self.btn_test)
        vbox_load.addWidget(self.lbl_test_status)
        grp_load.setLayout(vbox_load)
        
        # Parameters Group
        grp_params = QGroupBox("Testing Parameters")
        form_params = QFormLayout()
        
        self.spin_pos_tol = QSpinBox()
        self.spin_pos_tol.setRange(10, 200)
        self.spin_pos_tol.setValue(60)
        
        self.spin_col_tol = QSpinBox()
        self.spin_col_tol.setRange(10, 200)
        self.spin_col_tol.setValue(50)
        
        self.spin_bright = QDoubleSpinBox()
        self.spin_bright.setRange(0.1, 1.0)
        self.spin_bright.setSingleStep(0.1)
        self.spin_bright.setValue(0.6)
        
        form_params.addRow("Pos Tolerance:", self.spin_pos_tol)
        form_params.addRow("Color Tolerance:", self.spin_col_tol)
        form_params.addRow("Brightness Ratio:", self.spin_bright)
        grp_params.setLayout(form_params)
        
        # Action Group
        grp_action = QGroupBox("Execution")
        vbox_action = QVBoxLayout()
        
        self.btn_run = QPushButton("RUN INSPECTION")
        self.btn_run.setFixedHeight(70)
        self.btn_run.setStyleSheet("background-color: #007acc; font-size: 14pt;")
        self.btn_run.clicked.connect(self.run_test)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        
        vbox_action.addWidget(self.btn_run)
        vbox_action.addWidget(self.progress_bar)
        grp_action.setLayout(vbox_action)
        
        # Add all to left layout
        left_layout.addWidget(grp_load)
        left_layout.addWidget(grp_params)
        left_layout.addWidget(grp_action)
        left_layout.addStretch()

        # --- CENTER (Image Viewer) ---
        self.viewer = ImageViewer()
        
        # --- RIGHT SIDEBAR (Analysis) ---
        right_panel = QFrame()
        right_panel.setFixedWidth(300)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        grp_results = QGroupBox("Analysis Results")
        vbox_results = QVBoxLayout()
        
        self.lbl_result = QLabel("READY")
        self.lbl_result.setAlignment(Qt.AlignCenter)
        self.lbl_result.setStyleSheet("font-size: 16pt; font-weight: bold; color: #e0e0e0;")
        
        self.lbl_stats = QLabel("Matches: -/-\nAccuracy: -%")
        self.lbl_stats.setStyleSheet("font-size: 11pt;")
        
        self.list_fails = QListWidget()
        
        vbox_results.addWidget(self.lbl_result)
        vbox_results.addWidget(self.lbl_stats)
        vbox_results.addWidget(QLabel("Fail Reasons:"))
        vbox_results.addWidget(self.list_fails)
        grp_results.setLayout(vbox_results)
        
        right_layout.addWidget(grp_results)

        # Assemble Main Layout
        # Using a Splitter for resizing flexibility if needed, but fixed sidebars are often better for tools
        # For this requirement "Dashboard Layout", manual layout is cleaner.
        
        main_layout.addWidget(left_panel)
        main_layout.addWidget(self.viewer, stretch=1)
        main_layout.addWidget(right_panel)

        # --- TOOLBAR ---
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        
        action_export = QAction("Export Report", self)
        action_export.triggered.connect(self.export_report)
        toolbar.addAction(action_export)

        # --- STATUS BAR ---
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.log("System Ready.")

    def log(self, message):
        self.status_bar.showMessage(message)

    def load_golden(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Golden Board", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.golden_img_path = path
            self.lbl_golden_status.setText(f"Loading: {path.split('/')[-1]}...")
            self.log(f"Loaded Golden: {path}. Processing...")
            
            # Reset cache if file changed
            if path != self.last_golden_path:
                self.cached_golden_lights = None
                self.last_golden_path = path

            # Start background detection
            self.start_golden_detection()

    def start_golden_detection(self):
        params = {'brightness': self.spin_bright.value()}
        self.golden_loader = GoldenLoaderThread(self.golden_img_path, params)
        self.golden_loader.finished.connect(self.on_golden_loaded)
        self.golden_loader.start()
        # Disable run button while processing? Optional, but good practice.
        # But requirement says "Initialize Cache... Update load_golden".
        # We will keep run button enabled but it will fallback to re-processing if clicked too early (handled by AnalysisWorker).

    def on_golden_loaded(self, lights):
        self.cached_golden_lights = lights
        count = len(lights)
        self.lbl_golden_status.setText(f"Golden loaded & processed ({count} lights)")
        self.log(f"Golden processing complete. Detected {count} lights.")

    def load_test(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Test Board", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.test_img_path = path
            self.lbl_test_status.setText(path.split('/')[-1])
            self.log(f"Loaded Test: {path}")
            
            # Load and display immediately
            img = cv2.imread(path)
            if img is not None:
                self.current_img = img
                self.viewer.set_image(img)

    def run_test(self):
        if not self.golden_img_path or not self.test_img_path:
            QMessageBox.warning(self, "Missing Files", "Please load both Golden and Test board images.")
            return

        # Disable UI
        self.btn_run.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.list_fails.clear()
        self.lbl_result.setText("ANALYZING...")
        self.lbl_result.setStyleSheet("color: #e0e0e0; font-size: 16pt; font-weight: bold;")
        
        # Prepare params
        params = {
            'pos_tol': self.spin_pos_tol.value(),
            'col_tol': self.spin_col_tol.value(),
            'brightness': self.spin_bright.value()
        }

        # Start Worker
        # Note: self.cached_golden_lights might be None if loader isn't finished yet,
        # or if it was reset. The Worker handles this by detecting if needed.
        self.worker = AnalysisWorker(
            self.golden_img_path, 
            self.test_img_path, 
            self.cached_golden_lights,
            params
        )
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.log.connect(self.log)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.error.connect(self.on_analysis_error)
        self.worker.start()

    def on_analysis_finished(self, result):
        passed, marked_img, details, golden_lights = result
        
        # Cache updates
        self.cached_golden_lights = golden_lights
        
        # UI Updates
        self.btn_run.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.current_img = marked_img
        self.viewer.set_image(marked_img)
        
        # Stats
        total_g = details['total_golden']
        matches = details['matches']
        acc = details['accuracy']
        self.lbl_stats.setText(f"Matches: {matches}/{total_g}\nAccuracy: {acc:.1f}%")
        
        # Fail Reasons
        for reason in details['fail_reasons']:
            self.list_fails.addItem(reason)
            
        # Overall Result
        if passed:
            self.lbl_result.setText("PASS")
            self.lbl_result.setObjectName("SuccessLabel")
        else:
            self.lbl_result.setText("FAIL")
            self.lbl_result.setObjectName("FailLabel")
            
        # Re-apply stylesheet to update label color by ID
        self.lbl_result.style().unpolish(self.lbl_result)
        self.lbl_result.style().polish(self.lbl_result)

    def on_analysis_error(self, err_msg):
        self.btn_run.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Analysis Error", err_msg)
        self.log("Error occurred.")

    def export_report(self):
        QMessageBox.information(self, "Export", "Export Report feature coming soon!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BoardTester()
    window.show()
    sys.exit(app.exec_())
