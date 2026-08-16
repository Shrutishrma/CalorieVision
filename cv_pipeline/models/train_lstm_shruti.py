"""
train_lstm.py
=============
Trains an LSTM on the windowed keypoint sequences.
The class list is derived dynamically from data/windowed/split.json,
sorted alphabetically for a stable label index. The mapping is saved
to models/label_map.json.
"""

import json
import sys
from pathlib import Path
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from models.seq_utils import WindowedKeypointDataset, compute_class_weights, train_model, plot_curves, evaluate_and_report

class KeypointLSTM(nn.Module):
    def __init__(self, input_size=132, hidden_size=64, num_layers=2, num_classes=None, dropout=0.2):
        """
        num_classes is required and must be passed explicitly —
        derived from the label_map at runtime so it always matches the data.
        """
        if num_classes is None:
            raise ValueError(
                "num_classes must be passed explicitly. "
                "Derive it from models/label_map.json, not hardcoded."
            )
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x, lengths=None):
        # x is (batch, T, 132)
        out, (hn, cn) = self.lstm(x)
        # Take the last timestep
        last_out = out[:, -1, :] 
        return self.fc(last_out)

def load_label_map(split_path: Path, models_dir: Path) -> tuple[list, dict]:
    """
    Derive the class list dynamically from split.json.
    Returns (sorted_classes, class_to_idx) and saves models/label_map.json.
    """
    with open(split_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    all_classes: set = set()
    for split_name in ["train", "val", "test"]:
        for item in split_data.get(split_name, []):
            all_classes.add(item[1])  # item = [win_id, cls, orig_clip_id]

    sorted_classes = sorted(all_classes)  # alphabetical = stable label index
    class_to_idx = {c: i for i, c in enumerate(sorted_classes)}

    label_map = {
        "classes": sorted_classes,
        "class_to_idx": class_to_idx,
        "num_classes": len(sorted_classes),
    }
    label_map_path = models_dir / "label_map.json"
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)
    print(f"[leMON] label_map.json saved: {label_map_path}")
    print(f"[leMON] {len(sorted_classes)} classes: {sorted_classes}")
    return sorted_classes, class_to_idx


def main():
    windowed_dir = Path("data/windowed")
    reports_dir = Path("reports")
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    split_path = windowed_dir / "split.json"

    # --- Derive class list dynamically ---
    sorted_classes, class_to_idx = load_label_map(split_path, models_dir)
    num_classes = len(sorted_classes)

    # Load Data (seq_utils will use the dynamic class list via label_map.json)
    train_dataset = WindowedKeypointDataset(split_path, "train", windowed_dir)
    val_dataset   = WindowedKeypointDataset(split_path, "val",   windowed_dir)
    test_dataset  = WindowedKeypointDataset(split_path, "test",  windowed_dir)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

    # Weights
    class_weights = compute_class_weights(train_dataset)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # num_classes derived from data — never hardcoded
    model = KeypointLSTM(num_classes=num_classes).to(device)
    print(f"KeypointLSTM: input=132, hidden=64, layers=2, num_classes={num_classes}")

    print("\nTraining LSTM...")
    model, history = train_model(
        model, train_loader, val_loader, class_weights,
        num_epochs=40, lr=1e-3, patience=10, device=device
    )

    torch.save(model.state_dict(), str(models_dir / "lstm.pt"))
    plot_curves(history, reports_dir / "learning_curves_lstm.png", "LSTM")

    print("\nEvaluating LSTM on Test Set...")
    clip_acc, clip_macro_f1, n_zero = evaluate_and_report(
        model, test_loader, device, "LSTM", reports_dir
    )

    # Append/write to CSV (models listed after baselines)
    csv_path = reports_dir / "baseline_results.csv"
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["LSTM", f"{clip_acc:.6f}", f"{clip_macro_f1:.6f}", n_zero])

    print("\nDone training LSTM.")

if __name__ == "__main__":
    main()
