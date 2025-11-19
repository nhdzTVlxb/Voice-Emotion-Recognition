import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import f1_score

from utils.dataset import AudioEmotionDataset
from utils.config import Config
from models.model import SERModel


class EarlyStopping:
    def __init__(self, patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_f1 = 0.0

    def __call__(self, val_f1, model, path):
        if self.best_score is None or val_f1 > self.best_score:
            self.best_score = val_f1
            self.counter = 0
            self.best_f1 = val_f1
            torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), path)
            if self.verbose:
                print(f"EarlyStopping: Improved val F1 to {val_f1:.4f}. Model saved.")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: No improvement {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


def train():
    config = Config()

    dataset = AudioEmotionDataset(config.data_path, config, augment=True)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    model = SERModel(num_classes=len(config.labels))

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = nn.DataParallel(model)

    model = model.to(config.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    writer = SummaryWriter(log_dir=config.log_dir)
    early_stopping = EarlyStopping(patience=7, verbose=True)

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        all_preds, all_labels = [], []

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}"):
            inputs, labels = inputs.to(config.device), labels.to(config.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_f1 = f1_score(all_labels, all_preds, average='macro')

        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(config.device), labels.to(config.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_f1 = f1_score(val_labels, val_preds, average='macro')

        print(f'Epoch [{epoch + 1}/{config.epochs}] '
              f'Train Loss: {avg_train_loss:.4f} F1: {train_f1:.4f} '
              f'Val Loss: {avg_val_loss:.4f} F1: {val_f1:.4f}')

        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val', avg_val_loss, epoch)
        writer.add_scalar('F1/Train', train_f1, epoch)
        writer.add_scalar('F1/Val', val_f1, epoch)

        scheduler.step(val_f1)

        early_stopping(val_f1, model, os.path.join(config.result_path, 'best_model.pth'))
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    writer.close()


if __name__ == "__main__":
    train()
