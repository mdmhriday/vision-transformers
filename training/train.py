import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ..utils.logging import setup_logger

import os
from tqdm import tqdm

class Train:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        num_epoch=10,
        learning_rate=0.03,
        batch_size=32,
        device=None,
        save_path=None,
        weight_decay=1e-4):

        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epoch = num_epoch
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.save_path = save_path
        self.weight_decay = weight_decay

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        self.logger = setup_logger()

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)

            loss = self.criterion(outputs, labels)
            running_loss += loss.item()

            loss.backward()
            self.optimizer.step()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(self.train_loader)
        accuracy = 100 * correct / total

        self.logger.info(f"Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.2f}%")
        return avg_loss, accuracy

    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.val_loader: 
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)

                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
        avg_loss = running_loss / len(self.val_loader)
        accuracy = 100 * correct / total

        self.logger.info(f"Val Loss {avg_loss:.4f}, Val Accuracy {accuracy:.4f}")

        return avg_loss, accuracy

    def save_checkpoint(self, epoch, best_val_acc):
        if self.save_path and not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        checkpoint_path = f"{self.save_path}/checkpoint_epoch_{epoch}_val_acc_{best_val_acc:.4f}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': best_val_acc,
        }, checkpoint_path)
        
        self.logger.info(f"Model checkpoint saved to {checkpoint_path}")

    def train(self):
        best_val_acc = 0.0
        for epoch in range(self.num_epoch):
            train_loss, train_acc = self.train_epoch()

            val_loss, val_acc = self.validate_epoch()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(epoch, best_val_acc)

        self.logger.info("Training complete!")