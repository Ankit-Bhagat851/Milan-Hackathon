from tkinter import Tk, Label

def show_overlay(message):
    root = Tk()
    root.overrideredirect(True)
    root.geometry("300x80+100+100")
    root.configure(bg="black")
    label = Label(root, text=message, fg="white", bg="black", font=("Arial", 16))
    label.pack(expand=True)
    root.after(2000, lambda: root.destroy())
    root.mainloop()