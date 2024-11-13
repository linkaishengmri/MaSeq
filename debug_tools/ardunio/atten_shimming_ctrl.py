 
import numpy as np
import sys
import serial
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, 
                             QSpinBox, QPushButton, QDoubleSpinBox, QHBoxLayout)

# To find your port, use this command:
# sudo dmesg | grep ttyUSB*
# then type here:
COMPORT = '/dev/ttyUSB0'
# before you run this python file, run this command first.
# sudo chmod 666 /dev/ttyUSB0 

class SerialControl(QWidget):
    def __init__(self):
        super().__init__()
        self.serial_port = None
        self.initUI()

    def initUI(self):
        # Layouts
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Connection Button
        self.connect_button = QPushButton('Connect', self)
        self.connect_button.clicked.connect(self.connect_arduino)
        main_layout.addWidget(self.connect_button)

        # Layout for two attenuation and voltage groups
        attenuation_voltage_layout = QHBoxLayout()

        # Left side (Attenuation 1 + 4 Voltage Channels)
        left_layout = QVBoxLayout()
        self.attu1_label = QLabel('Attenuation 1 (dB):', self)
        self.attu1_spinbox = QDoubleSpinBox(self)
        self.attu1_spinbox.setRange(0, 31.75)
        self.attu1_spinbox.setSingleStep(0.25)
        left_layout.addWidget(self.attu1_label)
        left_layout.addWidget(self.attu1_spinbox)

        self.voltage_labels = []
        self.voltage_spinboxes = []
        for i in range(4):
            label = QLabel(f'Channel {i+1} Voltage (V):', self)
            spinbox = QDoubleSpinBox(self)
            spinbox.setRange(-2, 2)
            spinbox.setSingleStep(0.001)
            spinbox.setDecimals(4)
            self.voltage_labels.append(label)
            self.voltage_spinboxes.append(spinbox)
            left_layout.addWidget(label)
            left_layout.addWidget(spinbox)
        
        attenuation_voltage_layout.addLayout(left_layout)

        # Right side (Attenuation 2 + 4 Voltage Channels)
        right_layout = QVBoxLayout()
        self.attu2_label = QLabel('Attenuation 2 (dB):', self)
        self.attu2_spinbox = QDoubleSpinBox(self)
        self.attu2_spinbox.setRange(0, 31.75)
        self.attu2_spinbox.setSingleStep(0.25)
        right_layout.addWidget(self.attu2_label)
        right_layout.addWidget(self.attu2_spinbox)

        for i in range(4, 8):
            label = QLabel(f'Channel {i+1} Voltage (V):', self)
            spinbox = QDoubleSpinBox(self)
            spinbox.setRange(-2, 2)
            spinbox.setSingleStep(0.001)
            spinbox.setDecimals(4)
            self.voltage_labels.append(label)
            self.voltage_spinboxes.append(spinbox)
            right_layout.addWidget(label)
            right_layout.addWidget(spinbox)
        
        attenuation_voltage_layout.addLayout(right_layout)

        main_layout.addLayout(attenuation_voltage_layout)

        # Send Button
        self.send_button = QPushButton('Send', self)
        self.send_button.setEnabled(False)
        self.send_button.clicked.connect(self.send_data)
        main_layout.addWidget(self.send_button)

        self.setWindowTitle('Serial Control Panel')
        self.show()

    def connect_arduino(self):
        try:
            self.serial_port = serial.Serial(COMPORT, 4800, timeout=10)  # Adjust port and baudrate as needed
            self.serial_port.write(b'\xFE')  # Send byte FE
            response = self.serial_port.read(1)  # Read response

            if response == b'\xFD':  # If response is FD, enable other controls
                self.send_button.setEnabled(True)
                self.connect_button.setEnabled(False)
                print("Connection successful!")
            else:
                print("Failed to connect to Arduino.")
        except Exception as e:
            print(f"Error connecting to Arduino: {e}")

    def calc_attu_bin_val(self, value):
        # Placeholder for calculating attenuation binary value (8-bit byte)
        return round(value * 4)

    def calc_volt_bin_val(self, value):
        # Placeholder for calculating voltage binary value (24-bit byte)
        
        # Perform the calculation
        long_int = np.round(131071.49 * value / 10).astype(np.uint32) & 0x3FFFF
        
        # Convert the integer into bytes
        byte_1 = (long_int >> 16) & 0xFF  # Extract the most significant byte
        byte_2 = (long_int >> 8) & 0xFF   # Extract the middle byte
        byte_3 = long_int & 0xFF          # Extract the least significant byte
    
        return bytes([byte_1, byte_2, byte_3])

    def send_data(self):
        try:
            # Send Attenuation 1 Value (E0 xx 0E)
            attu1_value = self.attu1_spinbox.value()
            attu1_bin = self.calc_attu_bin_val(attu1_value)
            attu1_command = bytes([0xE0, attu1_bin, 0x0E])
            self.serial_port.write(attu1_command)
            print(f'Sent attenuation command (#1 ATTEN.): {attu1_command}')

            # Send Attenuation 2 Value (E1 xx 1E)
            attu2_value = self.attu2_spinbox.value()
            attu2_bin = self.calc_attu_bin_val(attu2_value)
            attu2_command = bytes([0xE1, attu2_bin, 0x1E])
            self.serial_port.write(attu2_command)
            print(f'Sent attenuation command (#2 ATTEN.): {attu2_command}')

            # Send Voltage for Channel 0-7 (F0-F7 xx xx xx)
            for i in range(8):
                voltage_value = self.voltage_spinboxes[i].value()
                volt_bin = self.calc_volt_bin_val(voltage_value)
                volt_command = bytes([0xF0 + i]) + volt_bin + bytes([0x0F | (i << 4)])
                self.serial_port.write(volt_command)
                print(f'Sent voltage command for Channel {i+1}: {volt_command}')
                
        except Exception as e:
            print(f"Error sending data: {e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SerialControl()
    sys.exit(app.exec_())
