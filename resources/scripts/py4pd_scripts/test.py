import tkinter as tk
import threading
import time
import pd

# Function to run in the background
def background_task():
    while True:
        print("Background task is running")
        time.sleep(2)
        break

def tkThing():
    window = tk.Tk()
    greeting = tk.Label(text="Hello, Tkinter")
    greeting.pack()

    # Set the window title
    window.title("My Window")

    # Set the window size
    window.geometry("400x300")

    # Create a label widget
    label = tk.Label(window, text="Hello, World!")
    label.pack()

    window.update()

    # Create a thread for the background task
    background_thread = threading.Thread(target=background_task) 
    background_thread.daemon = True
    background_thread.start()
    background_thread.join()

    window.quit()



def py4pdLoadObjects():
    pd.addobject(tkThing, "tk")
