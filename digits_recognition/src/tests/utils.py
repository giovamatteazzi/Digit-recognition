import torch
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import seaborn as sns
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import random

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
PLOTS_DIR = ROOT_DIR / "plots"
DATA_DIR = ROOT_DIR / "data"
PLOTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return torch.sum(preds == labels).item() / len(labels)


def visualize_all_stats(all_labels, all_preds, test_loss, test_acc, test_loader):
    conf_matrix = confusion_matrix(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    # print("\nConfusion Matrix: \n", conf_matrix, '\n')
    print("Classification Report:\n")
    print(classification_report(all_labels, all_preds, digits=4))
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}\n\n")
    print(f'ACCURACY: {test_acc:.4f}')

def save_confmat(all_labels, all_preds):
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Pred")
    plt.ylabel("Label")
    plt.title("Confusion Matrix")
    plt.savefig(PLOTS_DIR / "confmat.png")
    plt.show()


def visualize_sample(device, model, test_loader):
    samples, labels = next(iter(test_loader))   # extract an input batch from test_data
    samples = samples.to(device)
    outputs = model(samples)
    _, preds = torch.max(outputs, 1)    # predicts using the model

    n_lines = 4
    n_columns = 8
    num_examples = min(samples.size(0), n_columns*n_lines)
    random_indices = random.sample(range(samples.size(0)), num_examples)    # randomly picks num_examples images

    samples = samples.numpy()
    fig, axes = plt.subplots(n_lines, n_columns, figsize=(10, 10))    # plots
    for i, ax in enumerate(axes.ravel()[:num_examples]):
        index = random_indices[i]
        if preds[index] != labels[index]:
            ax.imshow(samples[index].squeeze(), cmap='Reds')
        else:
            ax.imshow(samples[index].squeeze(), cmap='gray')
        ax.set_title(f'L: {labels[index]}, P: {preds[index]}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "samples.png")    # saves
    plt.show()
import torch
import random
import matplotlib.pyplot as plt

def visualize_mistakes(device, model, test_loader):

    model.eval()
    wrong_samples = []
    wrong_labels = []
    wrong_preds = []

    with torch.no_grad():
        for samples, labels in test_loader:
            samples = samples.to(device)
            labels = labels.to(device)
            outputs = model(samples)
            _, preds = torch.max(outputs, 1)

            wrong_mask = preds != labels

            if wrong_mask.any():
                wrong_samples.append(samples[wrong_mask].cpu())
                wrong_labels.append(labels[wrong_mask].cpu())
                wrong_preds.append(preds[wrong_mask].cpu())

    if len(wrong_samples) == 0:
        print("No error in test set")
        return

    # concatenating
    wrong_samples = torch.cat(wrong_samples)
    wrong_labels = torch.cat(wrong_labels)
    wrong_preds = torch.cat(wrong_preds)

    n_lines = 8
    n_columns = 8
    num_examples = min(len(wrong_samples), n_lines * n_columns)

    indices = random.sample(range(len(wrong_samples)), num_examples)
    samples_np = wrong_samples.numpy()

    # plot
    fig, axes = plt.subplots(n_lines, n_columns, figsize=(10, 10))

    for i, ax in enumerate(axes.ravel()):
        if i >= num_examples:
            ax.axis('off')
            continue
        idx = indices[i]
        ax.imshow(samples_np[idx].squeeze(), cmap='gray')
        ax.set_title(f'L: {wrong_labels[idx].item()}, P: {wrong_preds[idx].item()}')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "mistakes.png")
    plt.show()



class DigitRecognizerApp:
    def __init__(self, master, model):
        self.model = model
        
        self.master = master
        self.master.title("Draw a digit (0-9)")
        
        self.canvas_width = 200
        self.canvas_height = 200

        self.canvas = tk.Canvas(self.master, width=self.canvas_width, height=self.canvas_height, bg='white')    # create canva
        self.canvas.pack()

        self.button_frame = tk.Frame(self.master)    # creates container for buttons
        self.button_frame.pack()
        self.clear_button = tk.Button(self.button_frame, text='Clear', command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT)
        self.predict_button = tk.Button(self.button_frame, text='Predict', command=self.predict_digit)
        self.predict_button.pack(side=tk.LEFT)

        self.label = tk.Label(self.master, text="Draw a digit...")
        self.label.pack()

        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=255)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)     # binds mouse drawing in image

    def paint(self, event):
        x1, y1 = (event.x - 6), (event.y - 6)
        x2, y2 = (event.x + 6), (event.y + 6)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black')
        self.draw.ellipse([x1, y1, x2, y2], fill=0)     # draws circles

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.label.config(text="Draw a digit...")

    def preprocess_image(self):
        # resizing to 28x28
        image_resized = self.image.resize((28, 28), resample=Image.Resampling.LANCZOS)
        image_inverted = ImageOps.invert(image_resized)

        # normalizes and converts into in array numpy (black on white -> white on black for MNIST)
        image_array = np.asarray(image_inverted) / 255.0

        # (1, 1, 28, 28) -> batch x channel x height x width
        input_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return input_tensor

    def predict_digit(self):
        input_tensor = self.preprocess_image()

        with torch.no_grad():    # inference
            output = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = torch.max(probabilities).item() * 100

        probs = probabilities.squeeze().numpy()
        probs_str = " | ".join([f"{i}: {p*100:.1f}%" for i, p in enumerate(probs)])
        self.label.config(text=f"Prediction: {prediction} ({confidence:.1f}%)\n{probs_str}")

def draw_interface(model):   # create an instance of DigitRecognizer
    root = tk.Tk()
    app = DigitRecognizerApp(root, model)
    root.mainloop()