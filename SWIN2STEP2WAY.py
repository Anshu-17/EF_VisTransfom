import timm
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import shutil
import torch.nn.functional as F
from pathlib import Path
from torch import nn, optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Paths to training and test datasets
train_dir = r'/home/aprasad/EF_VisTransfom/TrainTestuhh5050/train'
test_dir = r'/home/aprasad/EF_VisTransfom/TrainTestuhh5050/test'
# train_dir = r'/home/aprasad/EF_VisTransfom/TrainTestuhh6040/train'
# test_dir = r'/home/aprasad/EF_VisTransfom/TrainTestuhh6040/test'

# Number of workers
NUM_WORKERS = os.cpu_count()

def create_dataloaders(train_dir: str, test_dir: str, transform: transforms.Compose, batch_size: int, val_split: float = 0.2, num_workers: int = NUM_WORKERS):
    full_train_data = datasets.ImageFolder(train_dir, transform=transform)
    val_size = int(len(full_train_data) * val_split)
    train_size = len(full_train_data) - val_size
    train_data, val_data = random_split(full_train_data, [train_size, val_size])
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    class_names = full_train_data.classes

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader, class_names

# Image transformations
IMG_SIZE = 256
manual_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Batch size
BATCH_SIZE = 32
train_dataloader, val_dataloader, test_dataloader, class_names = create_dataloaders(train_dir, test_dir, transform=manual_transforms, batch_size=BATCH_SIZE)

# Print class information
print(len(class_names))
print(class_names)

# Dataset size information
print(f'Total number of training images: {len(train_dataloader.dataset)}')
print(f'Total number of validation images: {len(val_dataloader.dataset)}')
print(f'Total number of test images: {len(test_dataloader.dataset)}')


# Model initialization
model = timm.create_model('swinv2_base_window12to16_192to256', pretrained=True, num_classes=len(class_names), drop_path_rate=0.3)
model = model.to(device)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
     
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Cross entropy loss
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Convert targets to one-hot encoding
        targets_one_hot = torch.zeros_like(inputs).scatter(1, targets.unsqueeze(1), 1)

        # Compute pt (the predicted probability for the true class)
        pt = torch.exp(-BCE_loss)
        
        # Focal Loss calculation
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        # Apply reduction (mean or sum or no reduction)
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

# Define loss, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
#criterion = FocalLoss(alpha=1, gamma=2)
#optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=6, verbose=True)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_dataloader), epochs=50)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

early_stopping = EarlyStopping(patience=7, verbose=True)

# Training and evaluation function
def train_and_evaluate(model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler, num_epochs, early_stopping):
    train_losses, val_losses, test_losses = [], [], []
    train_accuracies, val_accuracies, test_accuracies = [], [], []
    y_true, y_pred = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_accuracy = correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_accuracy)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_train_loss:.4f}, Training Accuracy: {epoch_train_accuracy:.4f}")

        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_accuracy = correct_val / total_val
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_accuracy)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {epoch_val_loss:.4f}, Validation Accuracy: {epoch_val_accuracy:.4f}")

        test_loss, correct_test, total_test = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        epoch_test_loss = test_loss / len(test_loader.dataset)
        epoch_test_accuracy = correct_test / total_test
        test_losses.append(epoch_test_loss)
        test_accuracies.append(epoch_test_accuracy)
        print(f"Epoch {epoch+1}/{num_epochs}, Testing Loss: {epoch_test_loss:.4f}, Testing Accuracy: {epoch_test_accuracy:.4f}")

        scheduler.step(epoch_val_loss)
        early_stopping(epoch_val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    return train_losses, val_losses, test_losses, train_accuracies, val_accuracies, test_accuracies, y_true, y_pred

# Number of epochs
num_epochs = 50

# Train and evaluate the model
train_losses, val_losses, test_losses, train_accuracies, val_accuracies, test_accuracies, y_true, y_pred = train_and_evaluate(
    model, train_dataloader, val_dataloader, test_dataloader, criterion, optimizer, scheduler, num_epochs, early_stopping)

# Plot accuracy and loss
epochs = range(1, len(train_losses) + 1)
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Training Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.plot(epochs, test_losses, label='Testing Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label='Training Accuracy')
plt.plot(epochs, val_accuracies, label='Validation Accuracy')
plt.plot(epochs, test_accuracies, label='Testing Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('SWIN-accuracy_loss_plot.png')

# Confusion matrix and classification report
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues')
plt.savefig('SWINCFM.png')

print(classification_report(y_true, y_pred, target_names=class_names))


# Directories for storing classified images
unhealthy_tp_dir = Path('/home/aprasad/EF_VisTransfom/TrPos/Unhealthy')
unhealthy_fn_dir = Path('/home/aprasad/EF_VisTransfom/FalNeg/Unhealthy')
unhealthy_fp_dir = Path('/home/aprasad/EF_VisTransfom/FalPos/Unhealthy')
healthy_tn_dir = Path('/home/aprasad/EF_VisTransfom/TrNeg/Healthy')

# Create directories if they don't exist
unhealthy_tp_dir.mkdir(parents=True, exist_ok=True)
unhealthy_fn_dir.mkdir(parents=True, exist_ok=True)
unhealthy_fp_dir.mkdir(parents=True, exist_ok=True)
healthy_tn_dir.mkdir(parents=True, exist_ok=True)



def create_test_dataloader_with_paths(test_dir: str, transform: transforms.Compose, batch_size: int, num_workers: int = NUM_WORKERS):
    # Create dataset and preserve paths
    test_data = datasets.ImageFolder(test_dir, transform=transform)
    
    # Modify the dataset to return file paths alongside images and labels
    test_data.samples = [(path, label) for (path, label) in test_data.samples]
    
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    # Extract the file paths
    file_paths = [s[0] for s in test_data.samples]
    return test_loader, file_paths

def classify_and_store_images_extended(model, test_loader, class_names, file_paths, confidence_threshold=0.9):
    model.eval()  # Set model to evaluation mode
    y_true, y_pred = [], []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            confidences = F.softmax(outputs, dim=1)  # Get softmax probabilities
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

            # For each prediction, store true positives with confidence percentage
            batch_size = inputs.size(0)
            for idx, input in enumerate(inputs):
                confidence_percentage = confidences[idx][predicted[idx]].item() * 100
                file_path = file_paths[i * test_loader.batch_size + idx]

                # True Positive (TP)
                if predicted[idx] == labels[idx] and class_names[predicted[idx]] == 'unhealthy':
                    if confidence_percentage >= confidence_threshold:
                        shutil.copy(file_path, unhealthy_tp_dir / Path(file_path).name)

                # False Negative (FN): Actual Unhealthy, Predicted Healthy
                elif labels[idx] == class_names.index('unhealthy') and predicted[idx] == class_names.index('healthy'):
                    shutil.copy(file_path, unhealthy_fn_dir / Path(file_path).name)

                # False Positive (FP): Actual Healthy, Predicted Unhealthy
                elif labels[idx] == class_names.index('healthy') and predicted[idx] == class_names.index('unhealthy'):
                    shutil.copy(file_path, unhealthy_fp_dir / Path(file_path).name)

                # True Negative (TN): Actual Healthy, Predicted Healthy
                elif labels[idx] == predicted[idx] and class_names[predicted[idx]] == 'healthy':
                    if confidence_percentage >= confidence_threshold:
                        shutil.copy(file_path, healthy_tn_dir / Path(file_path).name)

                print(f"File {file_path} classified as {class_names[predicted[idx]]} with {confidence_percentage:.2f}% confidence.")
# Create the test dataloader and extract file paths
test_dataloader_with_paths, test_file_paths = create_test_dataloader_with_paths(test_dir, manual_transforms, BATCH_SIZE)

# Example usage
# Assume 'test_loader' provides the test data, 'class_names' lists the class labels, and 'test_file_paths' contains the image paths
classify_and_store_images_extended(model, test_dataloader_with_paths, class_names, test_file_paths)