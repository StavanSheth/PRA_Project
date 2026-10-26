"""
Main application entry point for the Live Hand Gesture Recognition Desktop Application.
"""

import os
import sys
import tkinter as tk

# Ensure workspace root is in python search path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.controller import GestureAppController
from app.gui import GestureAppGUI


def main():
    root = tk.Tk()

    # Initialize controller and GUI
    controller = GestureAppController(root)
    gui = GestureAppGUI(root, controller)
    controller.set_gui(gui)

    # Begin asynchronous model loading on startup
    root.after(100, controller.initialize_model_async)

    # Start main event loop
    root.mainloop()


if __name__ == "__main__":
    main()
