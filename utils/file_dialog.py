from shared import dependencies as dep


def select_image_file():
    root = dep.tk.Tk()
    root.withdraw()
    return dep.filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
