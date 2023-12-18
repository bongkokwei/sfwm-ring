from PyQt5.QtWidgets import (
    QDialog,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QPushButton,
    QFormLayout,
)


class PlotSettings(QDialog):
    def __init__(self, main_window, settings_dict, parent=None):
        super(PlotSettings, self).__init__(parent)
        self.main_window = main_window  # Store a reference to the main window
        self.settings_dict = settings_dict  # Settings dictionary

        self.setWindowTitle("Plot Settings")
        self.setGeometry(200, 200, 400, 200)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        form_layout = QFormLayout()

        for setting_name, setting_info in self.settings_dict.items():
            label = QLabel(setting_info["label"])
            edit = QLineEdit(self)
            edit.setText(str(setting_info["value"]))  # Set the default value
            form_layout.addRow(label, edit)

            # Store references to the QLineEdit widgets for later retrieval
            setting_info["edit"] = edit

        layout.addLayout(form_layout)

        # Add a button to apply the settings
        apply_button = QPushButton("Apply", self)
        apply_button.clicked.connect(self.apply_settings)
        layout.addWidget(apply_button)

        self.setLayout(layout)

    def apply_settings(self):
        # Update the values in the settings dictionary
        for setting_name, setting_info in self.settings_dict.items():
            edit = setting_info["edit"]
            setting_info["value"] = float(
                edit.text()
            )  # Assuming all values are numeric

        # Update the main window based on the new settings
        self.main_window.update_settings(self.settings_dict)

        # Close the popup window
        self.accept()
