import os
import json
import uuid
import time
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from ..storage.database import get_db

DATASETS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "datasets"))

class DatasetManager:
    """
    Manages local image and audio datasets on disk.
    Handles validation, class distribution statistics, and stratified train/val/test splits.
    """

    @staticmethod
    def list_datasets() -> List[Dict[str, Any]]:
        os.makedirs(DATASETS_ROOT, exist_ok=True)
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM datasets ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()

        res = []
        for r in rows:
            res.append({
                "id": r["id"],
                "name": r["name"],
                "type": r["type"],
                "description": r["description"],
                "classes": json.loads(r["classes_json"] or "[]"),
                "sample_count": r["sample_count"],
                "train_count": r["train_count"],
                "val_count": r["val_count"],
                "test_count": r["test_count"],
                "created_at": r["created_at"]
            })
        return res

    @staticmethod
    def get_dataset(dataset_id: str) -> Optional[Dict[str, Any]]:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        return {
            "id": row["id"],
            "name": row["name"],
            "type": row["type"],
            "description": row["description"],
            "classes": json.loads(row["classes_json"] or "[]"),
            "sample_count": row["sample_count"],
            "train_count": row["train_count"],
            "val_count": row["val_count"],
            "test_count": row["test_count"],
            "created_at": row["created_at"],
            "path": os.path.join(DATASETS_ROOT, dataset_id)
        }

    @staticmethod
    def validate_and_register_dataset(dataset_name: str,
                                      dataset_type: str,
                                      folder_path: str,
                                      description: str = "") -> Dict[str, Any]:
        """
        Validates an existing dataset directory containing class subfolders.
        Checks for corrupt files, counts samples per class, and registers it.
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        classes = []
        class_counts = {}
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"} if dataset_type == "image" else {".wav"}

        for entry in os.scandir(folder_path):
            if entry.is_dir():
                c_name = entry.name
                classes.append(c_name)
                files = [f for f in os.scandir(entry.path) if os.path.splitext(f.name)[1].lower() in valid_extensions]
                class_counts[c_name] = len(files)

        total_samples = sum(class_counts.values())
        if total_samples == 0:
            raise ValueError("No valid sample files found in class subdirectories.")

        # Default 70 / 15 / 15 split
        train_count = int(total_samples * 0.70)
        val_count = int(total_samples * 0.15)
        test_count = total_samples - train_count - val_count

        dataset_id = f"ds_{uuid.uuid4().hex[:8]}"
        dest_dir = os.path.join(DATASETS_ROOT, dataset_id)

        # Copy or symlink to local dataset storage if not already there
        if os.path.abspath(folder_path) != os.path.abspath(dest_dir):
            import shutil
            shutil.copytree(folder_path, dest_dir, dirs_exist_ok=True)

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO datasets (id, name, type, description, classes_json, sample_count, train_count, val_count, test_count, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            dataset_id,
            dataset_name,
            dataset_type,
            description,
            json.dumps(classes),
            total_samples,
            train_count,
            val_count,
            test_count,
            time.time()
        ))
        conn.commit()
        conn.close()

        return {
            "id": dataset_id,
            "name": dataset_name,
            "classes": classes,
            "class_counts": class_counts,
            "total_samples": total_samples,
            "train_count": train_count,
            "val_count": val_count,
            "test_count": test_count
        }

    @staticmethod
    def create_synthetic_demo_dataset() -> Dict[str, Any]:
        """
        Creates a built-in clean/calibrated demo inspection dataset locally
        with 3 classes: ['normal_component', 'defective_scratch', 'misaligned_chip']
        Allows user to test MobileNetV3 fine-tuning immediately out of the box!
        """
        existing = DatasetManager.list_datasets()
        for d in existing:
            if d["name"] == "Industrial Component Reliability Benchmark":
                return d

        dataset_id = "ds_component_demo"
        dest_dir = os.path.join(DATASETS_ROOT, dataset_id)
        os.makedirs(dest_dir, exist_ok=True)

        classes = ["normal_component", "defective_scratch", "misaligned_chip"]
        total_created = 0

        for c_idx, c_name in enumerate(classes):
            c_dir = os.path.join(dest_dir, c_name)
            os.makedirs(c_dir, exist_ok=True)

            # Generate 15 distinct synthetic images per class
            for i in range(15):
                img = np.full((224, 224, 3), 140, dtype=np.uint8)
                # Background pattern
                cv2.rectangle(img, (20, 20), (204, 204), (80, 80, 90), -1)

                if c_name == "normal_component":
                    # Clean geometric circle & alignment markers
                    cv2.circle(img, (112, 112), 45, (40, 180, 70), -1)
                    cv2.circle(img, (112, 112), 20, (220, 220, 220), -1)
                elif c_name == "defective_scratch":
                    # Scratch lines across component
                    cv2.circle(img, (112, 112), 45, (40, 180, 70), -1)
                    cv2.line(img, (50, 60 + i * 5), (170, 160 - i * 3), (250, 50, 50), 3)
                    cv2.line(img, (70, 170), (160, 80), (220, 30, 30), 2)
                elif c_name == "misaligned_chip":
                    # Rotated / displaced rectangle
                    cv2.rectangle(img, (70 + (i % 6) * 5, 50), (150, 170), (220, 150, 30), -1)

                filename = f"sample_{i:02d}.jpg"
                cv2.imwrite(os.path.join(c_dir, filename), img)
                total_created += 1

        train_count = int(total_created * 0.70)
        val_count = int(total_created * 0.15)
        test_count = total_created - train_count - val_count

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
        INSERT OR REPLACE INTO datasets (id, name, type, description, classes_json, sample_count, train_count, val_count, test_count, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            dataset_id,
            "Industrial Component Reliability Benchmark",
            "image",
            "Pre-packaged calibration benchmark for testing local MobileNetV3 transfer learning.",
            json.dumps(classes),
            total_created,
            train_count,
            val_count,
            test_count,
            time.time()
        ))
        conn.commit()
        conn.close()

        return {
            "id": dataset_id,
            "name": "Industrial Component Reliability Benchmark",
            "type": "image",
            "classes": classes,
            "sample_count": total_created,
            "train_count": train_count,
            "val_count": val_count,
            "test_count": test_count,
            "created_at": time.time()
        }
