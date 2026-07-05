from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QProcess, Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QWidget,
)

PACKAGE_NAME = (__package__ or "piezo2mic_app").split(".")[0]

STYLE_CHOICES = ("autodetect", "finger", "picked")
MODEL_CHOICES = ("offline", "realtime")
PICKUP_CHOICES = ("piezo", "magnetic")


def build_worker_command(
    input_path: str,
    output_path: str,
    style: str,
    model: str,
    pickup: str = "piezo",
) -> list[str]:
    if getattr(sys, "frozen", False):
        return [
            sys.executable,
            "--worker",
            "--input",
            input_path,
            "--output",
            output_path,
            "--style",
            style,
            "--model",
            model,
            "--pickup",
            pickup,
        ]

    return [
        sys.executable,
        "-m",
        PACKAGE_NAME,
        "--worker",
        "--input",
        input_path,
        "--output",
        output_path,
        "--style",
        style,
        "--model",
        model,
        "--pickup",
        pickup,
    ]


class Piezo2MicWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("Piezo2Mic")
        self.resize(760, 460)

        self.proc: QProcess | None = None
        self._closing = False

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QGridLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(10)
        layout.setColumnStretch(1, 1)
        layout.setRowStretch(6, 1)

        layout.addWidget(QLabel("Input"), 0, 0)
        self.input_edit = QLineEdit()
        layout.addWidget(self.input_edit, 0, 1)

        self.input_button = QPushButton("Browse…")
        self.input_button.clicked.connect(self._browse_input)
        layout.addWidget(self.input_button, 0, 2)

        layout.addWidget(QLabel("Output"), 1, 0)
        self.output_edit = QLineEdit()
        layout.addWidget(self.output_edit, 1, 1)

        self.output_button = QPushButton("Browse…")
        self.output_button.clicked.connect(self._browse_output)
        layout.addWidget(self.output_button, 1, 2)

        layout.addWidget(QLabel("Style"), 2, 0)
        self.style_combo = QComboBox()
        self.style_combo.addItems(STYLE_CHOICES)
        self.style_combo.setCurrentText("autodetect")
        layout.addWidget(self.style_combo, 2, 1, alignment=Qt.AlignmentFlag.AlignLeft)

        layout.addWidget(QLabel("Model"), 3, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(MODEL_CHOICES)
        self.model_combo.setCurrentText("offline")
        layout.addWidget(self.model_combo, 3, 1, alignment=Qt.AlignmentFlag.AlignLeft)

        layout.addWidget(QLabel("Pickup"), 4, 0)
        self.pickup_combo = QComboBox()
        self.pickup_combo.addItems(PICKUP_CHOICES)
        self.pickup_combo.setCurrentText("piezo")
        layout.addWidget(self.pickup_combo, 4, 1, alignment=Qt.AlignmentFlag.AlignLeft)

        buttons = QHBoxLayout()
        buttons.addStretch(1)

        self.process_button = QPushButton("Process")
        self.process_button.clicked.connect(self._start_processing)
        buttons.addWidget(self.process_button)

        self.quit_button = QPushButton("Quit")
        self.quit_button.clicked.connect(self.close)
        buttons.addWidget(self.quit_button)

        layout.addLayout(buttons, 5, 0, 1, 3)

        layout.addWidget(QLabel("Log"), 6, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 6, 1, 1, 2)

        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label, 7, 0, 1, 3)

    def _browse_input(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose input audio",
            "",
            "Audio files (*.wav *.aif *.aiff *.flac *.mp3);;All files (*.*)",
        )
        if not path:
            return

        self.input_edit.setText(path)

        if not self.output_edit.text().strip():
            self.output_edit.setText(self._default_output_for(path))

    def _browse_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Choose output file",
            self.output_edit.text().strip() or "",
            "WAV file (*.wav);;All files (*.*)",
        )
        if not path:
            return

        if not Path(path).suffix:
            path += ".wav"

        self.output_edit.setText(path)

    def _default_output_for(self, input_path: str) -> str:
        src = Path(input_path)
        return str(src.with_name(f"{src.stem}_mic.wav"))

    def _is_running(self) -> bool:
        return self.proc is not None and self.proc.state() != QProcess.ProcessState.NotRunning

    def _start_processing(self) -> None:
        if self._is_running():
            return

        input_path = self.input_edit.text().strip()
        output_path = self.output_edit.text().strip()
        style = self.style_combo.currentText().strip()
        model = self.model_combo.currentText().strip()
        pickup = self.pickup_combo.currentText().strip()

        if not input_path:
            QMessageBox.critical(self, "Piezo2Mic", "Please choose an input file.")
            return

        in_file = Path(input_path).expanduser()
        if not in_file.is_file():
            QMessageBox.critical(self, "Piezo2Mic", "Input file does not exist.")
            return

        if not output_path:
            QMessageBox.critical(self, "Piezo2Mic", "Please choose an output file.")
            return

        out_file = Path(output_path).expanduser()
        out_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            if in_file.resolve() == out_file.resolve():
                QMessageBox.critical(
                    self,
                    "Piezo2Mic",
                    "Input and output must be different files.",
                )
                return
        except Exception:
            pass

        cmd = build_worker_command(
            str(in_file),
            str(out_file),
            style,
            model,
            pickup,
        )

        self._append_log(
            f"Starting\n"
            f"  input : {in_file}\n"
            f"  output: {out_file}\n"
            f"  style : {style}\n"
            f"  model : {model}\n"
            f"  pickup: {pickup}\n\n"
        )

        self.status_label.setText("Processing…")
        self._set_controls_enabled(False)

        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._on_ready_read)
        self.proc.finished.connect(self._on_finished)
        self.proc.errorOccurred.connect(self._on_error)

        program, *args = cmd
        self.proc.start(program, args)

        if not self.proc.waitForStarted(5000):
            msg = self.proc.errorString() or "Unknown error"
            self._append_log(f"\nFailed to start worker: {msg}\n")
            self.status_label.setText("Failed to start")
            self._set_controls_enabled(True)
            QMessageBox.critical(self, "Piezo2Mic", f"Failed to start worker:\n{msg}")
            self.proc.deleteLater()
            self.proc = None

    def _on_ready_read(self) -> None:
        if self.proc is None:
            return

        data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        if data:
            self._append_log(data)

    def _on_error(self, error: QProcess.ProcessError) -> None:
        if self.proc is None:
            return

        if error == QProcess.ProcessError.FailedToStart:
            return

        self._append_log(f"\nProcess error: {self.proc.errorString()}\n")

    def _on_finished(self, exit_code: int, _exit_status) -> None:
        self._on_ready_read()

        proc = self.proc
        self.proc = None
        if proc is not None:
            proc.deleteLater()

        if self._closing:
            return

        self._set_controls_enabled(True)

        if exit_code == 0:
            self.status_label.setText("Done")
            self._append_log("\nFinished successfully.\n")
            QMessageBox.information(self, "Piezo2Mic", "Processing complete.")
        else:
            self.status_label.setText(f"Failed (exit code {exit_code})")
            self._append_log(f"\nFailed with exit code {exit_code}.\n")
            QMessageBox.critical(
                self,
                "Piezo2Mic",
                f"Processing failed with exit code {exit_code}.",
            )

    def _append_log(self, text: str) -> None:
        self.log.moveCursor(QTextCursor.MoveOperation.End)
        self.log.insertPlainText(text)
        self.log.moveCursor(QTextCursor.MoveOperation.End)
        self.log.ensureCursorVisible()

    def _set_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.input_edit,
            self.output_edit,
            self.input_button,
            self.output_button,
            self.style_combo,
            self.model_combo,
            self.pickup_combo,
            self.process_button,
        ):
            widget.setEnabled(enabled)

        self.quit_button.setEnabled(True)

    def closeEvent(self, event) -> None:
        if self._is_running():
            answer = QMessageBox.question(
                self,
                "Piezo2Mic",
                "A job is still running. Stop it and quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return

            self._closing = True

            if self.proc is not None:
                self.proc.terminate()
                if not self.proc.waitForFinished(3000):
                    self.proc.kill()
                    self.proc.waitForFinished(1000)

        event.accept()


def launch() -> int:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    app.setApplicationName("Piezo2Mic")

    window = Piezo2MicWindow()
    window.show()

    return app.exec()