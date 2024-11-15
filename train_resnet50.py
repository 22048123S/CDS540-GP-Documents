import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.models.resnet import ResNet50_Weights
from torch.utils.data import DataLoader
from datasets import load_from_disk
from tqdm import tqdm
import evaluate
import time

# 1. Load the dataset
dataset = load_from_disk("./processed_dataset")

train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# 2. Preprocessing and Data Augmentation
# Define image transformations for training and evaluation
train_transforms = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # Normalize using ImageNet's mean and standard deviation
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

eval_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # Normalize using ImageNet's mean and standard deviation
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# 3. Prepare labels
class_label = train_dataset.features["label"]
num_classes = class_label.num_classes


# 4. Define a custom Dataset class
# Because the pytorch model and the huggingface have different logic, we need to define a new data class.
class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get image and label
        image = self.dataset[idx]["image"]

        # Convert grayscale image to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Apply transformations
        image = self.transforms(image)
        label = self.dataset[idx]["label"]
        return image, label


# 5. Create DataLoaders
batch_size = 64

train_dataset = CustomImageDataset(train_dataset, train_transforms)
eval_dataset = CustomImageDataset(eval_dataset, eval_transforms)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)
eval_loader = DataLoader(
    eval_dataset, batch_size=batch_size, shuffle=False, num_workers=4
)

# 6. Define the model
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Load a pretrained ResNet-50 model
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

# Modify the final fully connected layer to match the number of classes
model.fc = nn.Linear(model.fc.in_features, num_classes)

model = model.to(device)

# 7. Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# 8. Training function
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    model.train()
    total_loss = 0
    scaler = torch.amp.GradScaler()  # Initialize gradient scaler for mixed precision

    for images, labels in tqdm(train_loader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Use autocast for mixed precision
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Scale the loss for backpropagation
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()  # Update the scaler for the next iteration

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss


# 9. Evaluation function
def evaluate_model(
    model: nn.Module,
    eval_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    model.eval()
    total_loss = 0

    # Load multiple metrics from evaluate
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")

    # Start timing for throughput calculation
    start_time = time.time()
    total_images = 0

    with torch.no_grad():
        for images, labels in tqdm(eval_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)  # Get batch size for throughput
            total_images += batch_size

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Get predictions
            _, preds = torch.max(outputs, 1)

            # Add predictions and references to each metric
            accuracy_metric.add_batch(predictions=preds, references=labels)
            precision_metric.add_batch(predictions=preds, references=labels)
            recall_metric.add_batch(predictions=preds, references=labels)
            f1_metric.add_batch(predictions=preds, references=labels)

    # End timing for throughput calculation
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Compute metrics
    avg_loss = total_loss / len(eval_loader)
    accuracy = accuracy_metric.compute()["accuracy"]
    precision = precision_metric.compute(average="weighted")["precision"]
    recall = recall_metric.compute(average="weighted")["recall"]
    f1 = f1_metric.compute(average="weighted")["f1"]

    # Calculate throughput
    throughput = total_images / elapsed_time  # Images per second

    return {
        "test_loss": avg_loss,
        "acc": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "throughput": throughput,
    }


# 10. Training loop
if __name__ == "__main__":
    num_epochs = 3

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        eval_matrics = evaluate_model(model, eval_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Eval Matrix: {eval_matrics}")

    # 11. Save the trained model
    torch.save(model.state_dict(), "./models/resnet50_trained.pth")

    # Optionally, save the entire model
    torch.save(model, "./models/resnet50_model.pth")
