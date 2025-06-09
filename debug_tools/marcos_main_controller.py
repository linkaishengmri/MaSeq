#!/usr/bin/env python

import os
import sys
#*****************************************************************************
# Get the directory of the current script
main_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(main_directory)
parent_directory = os.path.dirname(parent_directory)
# Define the subdirectories you want to add to sys.path
subdirs = ['MaSeq', 'marcos_client']

# Add the subdirectories to sys.path
for subdir in subdirs:
    full_path = os.path.join(parent_directory, subdir)
    sys.path.append(full_path)
#******************************************************************************
import time

import subprocess
import platform
import experiment as ex
import numpy as np
import shutil
import configs.hw_config_pseq as hw
import autotuning.autotuning as autotuning # Just to use an arduino
import threading


class MarcosMainController():
    
    def __init__(self, *args, **kwargs):
        """
        Initializes the MarcosController.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
       

        src_dir = os.path.join(parent_directory, "marcos_extras")
        dst_dir = os.path.join(parent_directory, "MaSeq")
        shutil.copy(os.path.join(src_dir, "copy_bitstream.sh"), dst_dir)
        shutil.copy(os.path.join(src_dir, "marcos_fpga_rp-122.bit"), dst_dir)
        shutil.copy(os.path.join(src_dir, "marcos_fpga_rp-122.bit.bin"), dst_dir)
        shutil.copy(os.path.join(src_dir, "marcos_fpga_rp-122.dtbo"), dst_dir)
        shutil.copy(os.path.join(src_dir, "readme.org"), dst_dir)


    def search_sdrlab(self):
        # Get the IP of the SDRLab
        if not self.main.demo:
            try:
                hw.rp_ip_address = self.get_sdrlab_ip()[0]
            except:
                print("ERROR: No SDRLab found.")
                try:
                    hw.rp_ip_address = self.get_sdrlab_ip()[0]
                except:
                    print("ERROR: No communication with SDRLab.")
                    print("ERROR: Try manually.")

    @staticmethod
    def get_sdrlab_ip():
        print("Searching for SDRLabs...")

        ip_addresses = []
        subnet = '192.168.31.'
        timeout = 0.1  # Adjust timeout value as needed

        for i in range(101,106):  # Scan IP range 192.168.1.101 to 192.168.1.132
            ip = subnet + str(i)
            try:
                if platform.system() == 'Linux':
                    result = subprocess.run(['ping', '-c', '1', ip], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   timeout=timeout)
                elif platform.system() == 'Windows':
                    result = subprocess.run(['ping', '-n', '1', ip], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   timeout=timeout)
                if result.returncode == 0:
                    print(f"Checking ip {ip}...")
                    # Attempt SSH connection without authentication
                    ssh_command = ['ssh', '-o', 'BatchMode=yes', '-o', f'ConnectTimeout={5}',
                                   f'{"root"}@{ip}', 'exit']
                    ssh_result = subprocess.run(ssh_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                    if ssh_result.returncode == 0:  # SSH was successful
                        ip_addresses.append(ip)
                    else:
                        print(f"WARNING: No SDRLab found at ip {ip}.")
            except:
                pass

        for ip in ip_addresses:
            print("READY: SDRLab found at IP " + ip)

        return ip_addresses

    # def startMaRCoS(self):
    #     """
    #     Starts the MaRCoS system.

    #     Executes startRP.sh: copy_bitstream.sh & marcos_server.
    #     """
    #     try:
    #         local_maseq_path = os.path.join(parent_directory, "MaSeq")
    #         communicate_script = os.path.join(local_maseq_path, "communicateRedPitaya.sh")
    #         start_script = os.path.join(local_maseq_path, "startRP.sh")

    #         subprocess.run([hw.bash_path, "--", communicate_script, hw.rp_ip_address, "killall marcos_server"])
    #         subprocess.run([hw.bash_path, "--", start_script, hw.rp_ip_address, hw.rp_version])

    #         self.initgpa()
    #         print("READY: MaRCoS updated, server connected, gpa initialized.")
    #     except:
    #         print("ERROR: Server connection not found! Please verify if the blue LED is illuminated on the Red Pitaya.")

    def runServer(self):
        try:
            local_maseq_path = os.path.join(parent_directory, "MaSeq")
            communicate_script = os.path.join(local_maseq_path, "communicateRedPitaya.sh")
            subprocess.run([hw.bash_path, "--", communicate_script, hw.rp_ip_address, "killall marcos_server"])
            subprocess.run([hw.bash_path, "--", communicate_script, hw.rp_ip_address, "~/marcos_server &"])
            print("READY: MaRCoS Server connected")
            self.initgpa()
            print("READY: gpa initialized.")
        except:
            print("ERROR: Server connection not found! Please verify if the blue LED is illuminated on the Red Pitaya.")
    
    def stopServer(self):
        try:
            local_maseq_path = os.path.join(parent_directory, "MaSeq")
            communicate_script = os.path.join(local_maseq_path, "communicateRedPitaya.sh")
            subprocess.run([hw.bash_path, "--", communicate_script, hw.rp_ip_address, "killall marcos_server"])
            print("MaRCoS Server disconnected now!")
             
        except:
            print("ERROR: Server disconnection not found!")
        
    # def controlMarcosServer(self):
    #     """
    #     Controls the MaRCoS server connection.

    #     Connects or disconnects from the MaRCoS server.
    #     """
    #     if not self.main.demo:
    #         if not self.action_server.isChecked():
    #             subprocess.run([hw.bash_path, "--", "./communicateRedPitaya.sh", hw.rp_ip_address, "killall marcos_server"])
    #             self.action_server.setStatusTip('Connect to marcos server')
    #             self.action_server.setToolTip('Connect to marcos server')
    #             print("Server disconnected")
    #         else:
    #             try:
    #                 subprocess.run([hw.bash_path, "--", "./communicateRedPitaya.sh", hw.rp_ip_address, "killall marcos_server"])
    #                 time.sleep(1.5)
    #                 subprocess.run([hw.bash_path, "--", "./communicateRedPitaya.sh", hw.rp_ip_address, "~/marcos_server"])
    #                 time.sleep(1.5)
    #                 self.action_server.setStatusTip('Kill marcos server')
    #                 self.action_server.setToolTip('Kill marcos server')

    #                 expt = ex.Experiment(init_gpa=False)
    #                 expt.add_flodict({
    #                     'grad_vx': (np.array([100]), np.array([0])),
    #                 })
    #                 expt.run()
    #                 expt.__del__()
    #                 print("READY: Server connected!")

    #             except Exception as e:
    #                 print("ERROR: Server not connected!")
    #                 print("ERROR: Try to connect to the server again.")
    #                 print(e)
    #     else:
    #         print("This is a demo\n")

    def copyBitStream(self):
        """
        Copies the MaRCoS bitstream to the remote platform.

        Executes copy_bitstream.sh.
        """
        maseq_path = os.path.join(parent_directory, "MaSeq")
        communicate_script = os.path.join(maseq_path, "communicateRedPitaya.sh")
        copy_bitstream_script = os.path.join(maseq_path, "copy_bitstream.sh")
        print(copy_bitstream_script)
        try:
            os.chdir(maseq_path)
            subprocess.run([hw.bash_path, "--", communicate_script, hw.rp_ip_address, "killall marcos_server"])
            subprocess.run([hw.bash_path, "--", copy_bitstream_script, hw.rp_ip_address, "rp-122"], timeout=100)
            print("READY: MaRCoS updated")
        except subprocess.TimeoutExpired as e:
            print("ERROR: MaRCoS init timeout")
            print(e)

    def initgpa(self):
        """
        Initializes the GPA board.
        """
        if True:
            if True:
                link = False
                while not link:
                    try:
                        # Check if GPA available
                        # received_string = self.arduino.send("GPA_VERB 1;").decode()
                        # if received_string[0:4] != ">OK;":
                        #     print("WARNING: GPA not available.")
                        # else:
                        #     print("READY: GPA available.")

                        # # Remote communication with GPA
                        # received_string = self.arduino.send("GPA_SPC:CTL 1;").decode()  # Activate remote control
                        # if received_string[0:4] != ">OK;":  # If wrong response
                        #     print("WARNING: Error enabling GPA remote control.")
                        # else:  # If good response
                        #     print("READY: GPA remote communication succeed.")

                        # # Check if RFPA available
                        # received_string = self.arduino.send("RFPA_VERB 1;").decode()
                        # if received_string[0:4] != ">OK;":
                        #     print("WARNING: RFPA not available.")
                        # else:
                        #     print("READY: RFPA available.")

                        # # Remote communication with RFPA
                        # received_string = self.arduino.send("RFPA_SPC:CTL 1;").decode()
                        # if received_string[0:4] != ">OK;":
                        #     print("WARNING: Error enabling RFPA remote control.")
                        # else:
                        #     print("READY: RFPA remote communication succeed.")

                        
                        # Run init_gpa sequence
                        expt = ex.Experiment(init_gpa=True, halt_and_reset=True, flush_old_rx =True)
                        expt.add_flodict({
                            'grad_vx': (np.array([100]), np.array([0])),
                        })
                        expt.run()
                        expt.__del__()
                        link = True
                        print("READY: GPA init done!")

                        # Enable power modules
                        # Enable GPA module
                        # received_string = self.arduino.send("GPA_ON 1;").decode()  # Enable power module
                        # if received_string[0:4] != ">OK;":  # If wrong response
                        #     print("WARNING: Error activating GPA power module.")
                        # else:  # If good reponse
                        #     print("READY: GPA power enabled.")

                        # # Enable RFPA module
                        # received_string = self.arduino.send("RFPA_RF 1;").decode()
                        # if received_string[0:4] != ">OK;":
                        #     print("WARNING: Error activating RFPA power module.")
                        # else:
                        #     print("READY: RFPA power enabled.")

                    except:
                        link = False
                        time.sleep(1)
        else:
            print("ERROR: No connection to the server")
            print("Please, connect to MaRCoS server first")
