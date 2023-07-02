import tkinter as tk
from tkinter import messagebox

def close_window():
    root.destroy()

def create_window():
    global root

    root = tk.Tk()
    root.title("Package Installation")

    label = tk.Label(root, text="Installing package...")
    label.pack()

    root.update()

    # Simulate package installation (replace this with your actual installation logic)
    import time
    time.sleep(5)  # Simulating a delay of 5 seconds

    messagebox.showinfo("Package Installation", "Package successfully installed!")

    close_window()


# Call other code or operations after the window is closed



def py4pdLoadObjects():
    pd.addobject(create_window, "tk")
