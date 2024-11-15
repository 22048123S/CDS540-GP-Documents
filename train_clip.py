import torch
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import CLIPProcessor, CLIPModel
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from tqdm import tqdm
import evaluate
import time

MODEL_PATH = "models/clip-vit-base-patch32"
# 1. load the pretrained CLIP model and its preprocessor
model = CLIPModel.from_pretrained(MODEL_PATH)
processor = CLIPProcessor.from_pretrained(MODEL_PATH)

# 2. load data and extract the labels
dataset = load_from_disk("./processed_dataset")

train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# 2.1 Convert labels from strings to integers
class_label = train_dataset.features["label"]
id_to_label = {i: class_label.int2str(i) for i in range(class_label.num_classes)}
label_to_id = {v: k for k, v in id_to_label.items()}


# 2.3 Design prompts for the text transformer

car_labels = list(label_to_id.keys())
car_prompts = [f"A photo of {car_type}" for car_type in car_labels]


# 2.4 Define DataLoader & Data collection function
def collate_fn(batch: dict[str, object]):
    images = [item["image"] for item in batch]
    labels = [item["label"] for item in batch]
    text_inputs = [car_prompts[label] for label in labels]
    inputs = processor(
        images=images,
        text=text_inputs,
        return_tensors="pt",
        padding=True,
    )
    inputs["labels"] = torch.tensor(labels)
    return inputs


# def collate_fn_test(batch):
#     images = [np.array(item["image"]) for item in batch]
#     labels = [item["label"] for item in batch]
#     text_inputs = [emotion_prompts[label] for label in labels]
#     inputs = processor(
#         images=images,
#         text=emotion_prompts,
#         return_tensors="pt",
#         padding=True,
#     )
#     inputs["labels"] = torch.tensor(labels)
#     return inputs


train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=4
)
eval_loader = DataLoader(
    eval_dataset,
    batch_size=64,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=4,
)


# 3. Training function
def train_model(
    model: CLIPModel, train_loader: DataLoader, optimizer: Adam, device: torch.device
):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        labels = batch["labels"].to(device)

        outputs = model.forward(**inputs, return_loss=True)
        logits = outputs.logits_per_image  # Image-text similarity score
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


# 4. Evaluation function
def evaluate_model(
    model: CLIPModel,
    eval_loader: DataLoader,
    device: torch.device,
):
    model.eval()

    # Load multiple metrics
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")

    # Precompute text features for all class prompts
    with torch.no_grad():
        text_inputs = processor(text=car_prompts, return_tensors="pt", padding=True).to(
            device
        )
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        )  # Normalize text features

    total_images = 0
    start_time = time.time()

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            batch_size = labels.size(0)
            total_images += batch_size

            # Get normalized image features
            image_features = model.get_image_features(batch["pixel_values"])
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Compute cosine similarities (image features @ text features)
            logits = image_features @ text_features.T

            # Predictions based on maximum similarity
            preds = logits.argmax(dim=1)

            # Add predictions and references to each metric
            accuracy_metric.add_batch(predictions=preds, references=labels)
            precision_metric.add_batch(predictions=preds, references=labels)
            recall_metric.add_batch(predictions=preds, references=labels)
            f1_metric.add_batch(predictions=preds, references=labels)

    # Calculate total time and throughput
    end_time = time.time()
    elapsed_time = end_time - start_time
    throughput = total_images / elapsed_time  # Images per second

    # Compute metrics
    accuracy = accuracy_metric.compute()["accuracy"]
    precision = precision_metric.compute(average="weighted")["precision"]
    recall = recall_metric.compute(average="weighted")["recall"]
    f1 = f1_metric.compute(average="weighted")["f1"]

    return {
        "acc": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "throughput": throughput,
    }


# def evaluate_model(model: CLIPModel, eval_loader: DataLoader, device: torch.device):
#     model.eval()
#     total_loss = 0
#     metric = evaluate.load("accuracy")

#     with torch.no_grad():
#         for batch in tqdm(eval_loader, desc="Evaluating"):
#             inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
#             labels = batch["labels"].to(device)

#             outputs = model.forward(**inputs, return_loss=True)
#             logits = outputs.logits_per_image  # Image-text similarity score
#             loss = outputs.loss
#             total_loss += loss.item()

#             preds = logits.argmax(dim=1)
#             metric.add_batch(predictions=preds, references=labels)

#     accuracy = metric.compute()["accuracy"]
#     return total_loss / len(eval_loader), accuracy


if __name__ == "__main__":
    # 1. Fine-tuning setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # copied from https://github.com/openai/CLIP/issues/83
    optimizer = AdamW(
        model.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2
    )
    epochs = 30

    # 2. Evaluate the performance before fine-tuning. (ZERO-SHOT Performance)
    test_matrix = evaluate_model(model, eval_loader, device)
    print("Zero shot performance: {}".format(test_matrix))
    # 3. Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train_model(model, train_loader, optimizer, device)
        test_matrix = evaluate_model(model, eval_loader, device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Eval Accuracy: {test_matrix}")

    print("Performance after finetuning: {}".format(test_matrix))
    # 4. After training, save the model
    model.save_pretrained("./models/finetuned_clip_model")
    processor.save_pretrained("./models/finetuned_clip_model")
