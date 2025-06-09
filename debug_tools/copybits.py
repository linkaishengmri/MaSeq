#!/usr/bin/env python

from marcos_main_controller import MarcosMainController

if __name__ == "__main__":
    print("This is a module for MaSeq, not a standalone script.")
    ctl=MarcosMainController()
    ctl.get_sdrlab_ip()
    ctl.copyBitStream()
