import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from typing import List, Dict, Any, Optional
from ..storage.database import get_db
from ..image.classifier import get_classifier

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "checkpoints"))

class LocalImageDataset(Dataset):
    def __init__(self, samples: List[Any], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

class ModelTrainingManager:
    """
    Manages local transfer learning and fine-tuning jobs for MobileNetV3-Small.
    Freezes convolutional backbone and trains a custom classification head on local CPU.
    """
    _current_job: Dict[str, Any] = {
        "status": "idle",
        "progress_pct": 0,
        "current_epoch": 0,
        "total_epochs": 0,
        "epoch_loss": 0.0,
        "val_accuracy": 0.0,
        "error": None
    }

    @classmethod
    def get_training_status(cls) -> Dict[str, Any]:
        return cls._current_job

    @classmethod
    def list_model_versions(cls) -> List[Dict[str, Any]]:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM model_versions ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()

        res = [{
            "id": "base-pretrained",
            "model_name": "MobileNetV3-Small (ImageNet Pretrained)",
            "version": "1.0.0-base",
            "is_active": (get_classifier().active_version == "pretrained-mobilenetv3-small"),
            "checkpoint_path": None,
            "metrics": {"source": "torchvision_pretrained", "val_accuracy": 1.0},
            "created_at": 0.0
        }]

        for r in rows:
            res.append({
                "id": r["id"],
                "model_name": r["model_name"],
                "version": r["version"],
                "is_active": bool(r["is_active"]),
                "checkpoint_path": r["checkpoint_path"],
                "metrics": json.loads(r["metrics_json"] or "{}"),
                "created_at": r["created_at"]
            })
        return res

    @classmethod
    def activate_model_version(cls, version_id: str) -> Dict[str, Any]:
        classifier = get_classifier()
        if version_id == "base-pretrained":
            classifier.revert_to_known_good()
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute("UPDATE model_versions SET is_active = 0")
            conn.commit()
            conn.close()
            return {"status": "success", "active_version": "pretrained-mobilenetv3-small"}

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM model_versions WHERE id = ?", (version_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            raise ValueError(f"Version {version_id} not found.")

        checkpoint_path = row["checkpoint_path"]
        metrics = json.loads(row["metrics_json"] or "{}")
        classes = metrics.get("classes", [])

        classifier.load_custom_checkpoint(checkpoint_path, classes, row["version"])

        cursor.execute("UPDATE model_versions SET is_active = 0")
        cursor.execute("UPDATE model_versions SET is_active = 1 WHERE id = ?", (version_id,))
        conn.commit()
        conn.close()

        return {"status": "success", "active_version": row["version"]}

    @classmethod
    def run_transfer_learning(cls,
                              dataset_path: str,
                              classes: List[str],
                              epochs: int = 3,
                              lr: float = 0.001,
                              batch_size: int = 8) -> Dict[str, Any]:
        """
        Runs local transfer learning:
        - Freezes MobileNetV3 features
        - Replaces classifier with linear head matching len(classes)
        - Trains for N epochs on CPU
        - Saves checkpoint to models/checkpoints/<version>.pt
        """
        os.makedirs(MODELS_DIR, exist_ok=True)
        cls._current_job = {
            "status": "training",
            "progress_pct": 0,
            "current_epoch": 0,
            "total_epochs": epochs,
            "epoch_loss": 0.0,
            "val_accuracy": 0.0,
            "error": None
        }

        try:
            # Gather samples
            all_samples = []
            for class_idx, c_name in enumerate(classes):
                c_dir = os.path.join(dataset_path, c_name)
                if os.path.isdir(c_dir):
                    for f in os.scandir(c_dir):
                        if os.path.splitext(f.name)[1].lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                            all_samples.append((f.path, class_idx))

            if len(all_samples) < 6:
                raise ValueError(f"Too few samples ({len(all_samples)}) to run transfer learning.")

            # Simple random train/val split (80/20)
            np.random.seed(42)
            np.random.shuffle(all_samples)
            split_idx = int(len(all_samples) * 0.8)
            train_samples = all_samples[:split_idx]
            val_samples = all_samples[split_idx:] if split_idx < len(all_samples) else all_samples

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

            train_ds = LocalImageDataset(train_samples, transform=transform)
            val_ds = LocalImageDataset(val_samples, transform=transform)

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

            # Load MobileNetV3-Small
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            model = models.mobilenet_v3_small(weights=weights)

            # Freeze backbone features
            for param in model.features.parameters():
                param.requires_grad = False

            # Replace classifier
            num_classes = len(classes)
            in_features = model.classifier[0].in_features
            model.classifier = nn.Sequential(
                nn.Linear(in_features, 1024),
                nn.Hardswish(),
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(1024, num_classes)
            )

            device = torch.device("cpu")
            model.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

            for epoch in range(epochs):
                model.train()
                running_loss = 0.0
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * images.size(0)

                epoch_loss = running_loss / max(1, len(train_ds))

                # Validation accuracy
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        _, preds = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (preds == labels).sum().item()

                val_acc = float(correct) / max(1, total)

                cls._current_job["current_epoch"] = epoch + 1
                cls._current_job["epoch_loss"] = round(epoch_loss, 4)
                cls._current_job["val_accuracy"] = round(val_acc, 3)
                cls._current_job["progress_pct"] = int(((epoch + 1) / epochs) * 100)

            # Save checkpoint
            version_tag = f"mobilenetv3-custom-{int(time.time())}"
            ckpt_filename = f"{version_tag}.pt"
            ckpt_path = os.path.join(MODELS_DIR, ckpt_filename)
            torch.save(model.state_dict(), ckpt_path)

            version_id = f"ver_{int(time.time())}"
            metrics_data = {
                "val_accuracy": round(val_acc, 3),
                "final_loss": round(epoch_loss, 4),
                "epochs": epochs,
                "classes": classes,
                "training_samples": len(train_samples)
            }

            conn = get_db()
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO model_versions (id, model_name, version, is_active, checkpoint_path, metrics_json, created_at)
            VALUES (?, ?, ?, 0, ?, ?, ?)
            """, (
                version_id,
                f"MobileNetV3 Fine-Tuned ({len(classes)} classes)",
                version_tag,
                ckpt_path,
                json.dumps(metrics_data),
                time.time()
            ))
            conn.commit()
            conn.close()

            cls._current_job["status"] = "completed"
            return {
                "version_id": version_id,
                "version_tag": version_tag,
                "val_accuracy": round(val_acc, 3),
                "checkpoint_path": ckpt_path
            }

        except Exception as e:
            cls._current_job["status"] = "failed"
            cls._current_job["error"] = str(e)
            raise e
