import tkinter as tk
from PIL import Image
import numpy as np
import joblib
import matplotlib.pyplot as plt
from preprocessing import adjust_colors, resize_image, adjust_center, binary_image
from sklearn.preprocessing import LabelEncoder

# Class to represent the GUI 
class App(tk.Tk):
    def __init__(self, path):
        tk.Tk.__init__(self)

        self.x = self.y = 0
        self.model = joblib.load(path+"trained_mlp_model.joblib")

        # Creating elements
        self.canvas = tk.Canvas(self, width=300, height=300, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Waiting...", font=("Helvetica", 48))
        self.classify_btn = tk.Button(
            self, text="Recognize", command=self.classify_handwriting
        )
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_all)

        # Grid structure
        self.canvas.grid(
            row=0,
            column=0,
            pady=2,
            sticky=tk.W,
        )
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)

        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def predict_digit(self, img):
        # Convert rgb to grayscale
        img = adjust_colors(img)
        img = img.resize((28, 28))

        # Resize image to 28x28 pixels
        img = binary_image(img)

        # Reshaping to support our model input and normalizing
        # img = adjust_center(img)
        img = img.reshape(1, 28, 28, 1)
        # img = img / 255.0

        # Predicting the image class
        np_img = np.array(img).flatten().astype(int).reshape(1, -1)

        res = self.model.predict(np_img)[0]

        return res

    # Cleaning the screen
    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        self.canvas.postscript(file="img.eps")
        # use PIL to convert to PNG
        img = Image.open("img.eps")

        character = self.predict_digit(img)
        self.label.configure(text=str(character))

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        radius = 8

        self.canvas.create_oval(
            self.x - radius,
            self.y - radius,
            self.x + radius,
            self.y + radius,
            fill="black",
        )


def main():
    App("../../trained_models/")
    tk.mainloop()


if __name__ == "__main__":
    main()