import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, optimizer, criterion, dataloader):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total

def evaluate(model, criterion, dataloader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item() * x.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return total_loss / total, correct / total

def calculate_metrics(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    acc = np.mean(preds == labels)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    return acc, precision, recall, f1

def two_stage_training(dataset, model_class, model_kwargs,
                       epochs_stage1=200, epochs_stage2=20,
                       batch_size=32, lr1=0.001, lr2=0.0001):
    outer_kf = KFold(n_splits=10, shuffle=True, random_state=42)
    all_subject_metrics = []

    for subj_idx in range(len(dataset)):
        print(f"Subject {subj_idx + 1} / {len(dataset)}")
        subject_fold_metrics = []

        data = dataset.data_per_subject[subj_idx]
        label = dataset.label_per_subject[subj_idx]

        data_np = data.cpu().numpy()
        label_np = label.cpu().numpy()

        for outer_fold, (train_val_idx, test_idx) in enumerate(outer_kf.split(data_np)):
            print(f" Outer Fold {outer_fold + 1} / 10")

            test_data = data[test_idx]
            test_label = label[test_idx]

            train_val_data_np = data_np[train_val_idx]
            train_val_label_np = label_np[train_val_idx]

            inner_kf = KFold(n_splits=3, shuffle=True, random_state=42)
            best_val_acc = 0
            best_model_state = None

            for inner_fold, (train_idx, val_idx) in enumerate(inner_kf.split(train_val_data_np)):
                print(f"  Inner Fold {inner_fold + 1} / 3")

                train_data = torch.tensor(train_val_data_np[train_idx], dtype=torch.float32)
                train_label = torch.tensor(train_val_label_np[train_idx], dtype=torch.long)
                val_data = torch.tensor(train_val_data_np[val_idx], dtype=torch.float32)
                val_label = torch.tensor(train_val_label_np[val_idx], dtype=torch.long)

                train_loader = DataLoader(TensorDataset(train_data, train_label), batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(TensorDataset(val_data, val_label), batch_size=batch_size, shuffle=False)

                model = model_class(**model_kwargs).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=lr1)

                for epoch in range(epochs_stage1):
                    train_loss, train_acc = train_one_epoch(model, optimizer, criterion, train_loader)
                    val_loss, val_acc = evaluate(model, criterion, val_loader)
                    #print(f"   Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
                    print(
                        f"   Epoch {epoch + 1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f},Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = model.state_dict()

            print(f"  Best val acc in inner folds: {best_val_acc:.4f}")

            # Fine-tune
            full_train_data = torch.tensor(train_val_data_np, dtype=torch.float32)
            full_train_label = torch.tensor(train_val_label_np, dtype=torch.long)
            full_train_loader = DataLoader(TensorDataset(full_train_data, full_train_label), batch_size=batch_size, shuffle=True)

            model = model_class(**model_kwargs).to(device)
            model.load_state_dict(best_model_state)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr2)

            for epoch in range(epochs_stage2):
                train_loss, train_acc = train_one_epoch(model, optimizer, criterion, full_train_loader)
                print(f"   Fine-tune Epoch {epoch+1}: Train Acc={train_acc:.4f}")
                if train_acc >= 1.0:
                    print("   Train acc reached 100%, early stop fine-tune")
                    break

            test_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=batch_size, shuffle=False)
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(device)
                    outputs = model(x)
                    preds = outputs.argmax(dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(y.numpy())

            acc, prec, rec, f1 = calculate_metrics(all_preds, all_labels)
            subject_fold_metrics.append((acc, prec, rec, f1))
            print(f" Outer Fold {outer_fold + 1} Test Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")


            if outer_fold == 9:
                model_path = f"subject_{subj_idx + 1}_final_model.pth"
                torch.save(model.state_dict(), model_path)
                print(f" Saved final fine-tuned model for Subject {subj_idx + 1} to {model_path}")

            print("-" * 50)

        acc_mean = np.mean([m[0] for m in subject_fold_metrics])
        prec_mean = np.mean([m[1] for m in subject_fold_metrics])
        rec_mean = np.mean([m[2] for m in subject_fold_metrics])
        f1_mean = np.mean([m[3] for m in subject_fold_metrics])
        print(f"\n>>> Subject {subj_idx+1} 10-Fold Average - Acc: {acc_mean:.4f}, Prec: {prec_mean:.4f}, Recall: {rec_mean:.4f}, F1: {f1_mean:.4f}")
        all_subject_metrics.append({
            "Subject": f"Subject_{subj_idx + 1}",
            "Accuracy": acc_mean,
            "Precision": prec_mean,
            "Recall": rec_mean,
            "F1": f1_mean
        })

    print("\n======= All Subjects Finished =======")
    for m in all_subject_metrics:
        print(f"{m['Subject']}: Acc={m['Accuracy']:.4f}, Prec={m['Precision']:.4f}, Recall={m['Recall']:.4f}, F1={m['F1']:.4f}")

    overall = {
        "Subject": "Overall_Avg",
        "Accuracy": np.mean([m["Accuracy"] for m in all_subject_metrics]),
        "Precision": np.mean([m["Precision"] for m in all_subject_metrics]),
        "Recall": np.mean([m["Recall"] for m in all_subject_metrics]),
        "F1": np.mean([m["F1"] for m in all_subject_metrics]),
    }
    print(f"\nOverall Avg - Acc: {overall['Accuracy']:.4f}, Prec: {overall['Precision']:.4f}, Recall: {overall['Recall']:.4f}, F1: {overall['F1']:.4f}")
    all_subject_metrics.append(overall)

    pd.DataFrame(all_subject_metrics).to_csv("subject_metrics.csv", index=False)
    print("\nResults saved to 'subject_metrics.csv'")
if __name__ == "__main__":
    from networks import MB_STFormer
    from eeg_dataset import EEGdataset

    data_path = "C:/Users/SLL/dataset/data_eeg_FATIG_FTG"
    dataset = EEGdataset(data_path)

    model_kwargs = dict(
        in_planes=30,
        out_planes=64,
        radix=5,
        patch_size=32,
        time_points=384,
        num_classes=2
    )

    two_stage_training(dataset, MB_STFormer, model_kwargs)