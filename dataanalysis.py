import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torchvision.utils import make_grid
from sklearn.metrics import confusion_matrix
import pandas as pd

class DataAnalysisUtils:
    def __init__(self, model, data_module, device):
        self.model = model
        self.data_module = data_module
        self.device = device

    def load_data(self, file_path):
        try:
            return np.load(file_path)
        except FileNotFoundError:
            print(f"파일을 찾을 수 없습니다: {file_path}")
            return None
        
    def plot_loss_curve(self, train_losses_path, val_losses_path):
        train_losses = self.load_data(train_losses_path)
        val_losses = self.load_data(val_losses_path)
        if train_losses is None or val_losses is None:
            return None
        
        fig = plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curves')
        plt.legend()
        return fig

    def plot_accuracy_curve(self, train_accuracies_path, val_accuracies_path):
        train_accuracies = self.load_data(train_accuracies_path)
        val_accuracies = self.load_data(val_accuracies_path)
        
        if train_accuracies is None or val_accuracies is None:
            return
        
        fig = plt.figure(figsize=(10, 6))
        plt.plot(train_accuracies, label='Training Accuracy')
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy Curves')
        plt.legend()
        return fig

    def plot_confusion_matrix(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.data_module.val_dataloader():
                images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        plt.close()

    def visualize_model_predictions(self, num_images=16):
        self.model.eval()
        dataiter = iter(self.data_module.val_dataloader())
        images, labels = next(dataiter)
        images = images[:num_images].to(self.device)
        labels = labels[:num_images]

        with torch.no_grad():
            outputs = self.model(images)
            _, preds = torch.max(outputs, 1)

        fig = plt.figure(figsize=(20, 20))
        for idx in range(num_images):
            ax = fig.add_subplot(4, 4, idx+1, xticks=[], yticks=[])
            img = images[idx].cpu().permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min())
            plt.imshow(img)
            ax.set_title(f"Pred: {preds[idx]}, True: {labels[idx]}")
        plt.savefig('model_predictions.png')
        plt.close()

    def analyze_misclassifications(self):
        self.model.eval()
        misclassified = []
        
        with torch.no_grad():
            for batch in self.data_module.val_dataloader():
                images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                
                misclassified_mask = preds != labels
                misclassified.extend(list(zip(
                    images[misclassified_mask].cpu(),
                    preds[misclassified_mask].cpu(),
                    labels[misclassified_mask].cpu()
                )))

        fig = plt.figure(figsize=(20, 20))
        for idx, (img, pred, true) in enumerate(misclassified[:16]):
            ax = fig.add_subplot(4, 4, idx+1, xticks=[], yticks=[])
            img = img.permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min())
            plt.imshow(img)
            ax.set_title(f"Pred: {pred.item()}, True: {true.item()}")
        plt.savefig('misclassified_examples.png')
        plt.close()

    def plot_learning_rate(self, learning_rates):
        plt.figure(figsize=(10, 6))
        plt.plot(learning_rates)
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.yscale('log')
        plt.savefig('learning_rate_schedule.png')
        plt.close()

# Usage example (to be added to your main training loop):
# utils = DataAnalysisUtils(model, data_module, device)
# 
# # After each epoch:
# utils.plot_loss_curve(train_losses, val_losses)
# utils.plot_accuracy_curve(train_accuracies, val_accuracies)
# 
# # After training:
# utils.plot_confusion_matrix()
# utils.visualize_model_predictions()
# utils.analyze_misclassifications()
# utils.plot_learning_rate(learning_rates)