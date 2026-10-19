"""Utilities shared by the fracture notebooks; labels follow ImageFolder class order."""
from pathlib import Path
import json
import random

import numpy as np
import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Preserve and load the one verified truncated JPEG; inspect it before any data cleanup.
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "Bone_Fracture_Binary_Classification"
RESULTS = ROOT / "results"
SEED = 42

def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def data_loaders(image_size=64, batch_size=64, validation_split=False):
    tfm = transforms.Compose([transforms.Grayscale(), transforms.Resize((image_size,image_size)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_ds = datasets.ImageFolder(DATA / "train", transform=tfm)
    val_ds = datasets.ImageFolder(DATA / "val", transform=tfm)
    test_ds = datasets.ImageFolder(DATA / "test", transform=tfm)
    assert train_ds.classes == val_ds.classes == test_ds.classes
    if validation_split:
        # Preserve official validation and test sets; carve a deterministic tuning split from training.
        from torch.utils.data import random_split
        n_val = max(1, int(0.2 * len(train_ds)))
        train_ds, val_ds = random_split(train_ds, [len(train_ds)-n_val, n_val], generator=torch.Generator().manual_seed(SEED))
    return (DataLoader(train_ds,batch_size=batch_size,shuffle=True), DataLoader(val_ds,batch_size=batch_size,shuffle=False), DataLoader(test_ds,batch_size=batch_size,shuffle=False), train_ds.classes)

def network(dropout=0.0):
    layers=[nn.Linear(64*64,256),nn.ReLU()]
    if dropout: layers.append(nn.Dropout(dropout))
    layers += [nn.Linear(256,128),nn.ReLU()]
    if dropout: layers.append(nn.Dropout(dropout))
    layers += [nn.Linear(128,2)]
    return nn.Sequential(nn.Flatten(),*layers)

def epoch(model, loader, criterion, optimizer=None, device="cpu"):
    training=optimizer is not None; model.train(training); total=correct=loss_sum=0
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        if training: optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits=model(x); loss=criterion(logits,y)
            if training: loss.backward(); optimizer.step()
        loss_sum += loss.item()*len(y); correct += (logits.argmax(1)==y).sum().item(); total += len(y)
    return loss_sum/total, correct/total

def evaluate(model, loader, classes, device="cpu"):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for x,y in loader: ps.extend(model(x.to(device)).argmax(1).cpu().tolist()); ys.extend(y.tolist())
    cm=confusion_matrix(ys,ps,labels=[0,1]); tn,fp,fn,tp=cm.ravel()
    return {"accuracy":accuracy_score(ys,ps),"class_names_by_index":classes,"positive_class":"fractured","fractured_precision":precision_score(ys,ps,pos_label=classes.index("fractured"),zero_division=0),"fractured_sensitivity":recall_score(ys,ps,pos_label=classes.index("fractured"),zero_division=0),"not_fractured_specificity":tn/(tn+fp) if tn+fp else 0.0,"confusion_matrix_rows_true_cols_pred":cm.tolist(),"n":len(ys)}

def save_history(history, name):
    RESULTS.joinpath("metrics").mkdir(parents=True,exist_ok=True)
    RESULTS.joinpath("metrics",f"{name}.json").write_text(json.dumps(history,indent=2),encoding="utf-8")

def inspect_images():
    rows=[]
    for split in ["train","val","test"]:
        for cls in sorted(p for p in (DATA/split).iterdir() if p.is_dir()):
            for f in cls.iterdir():
                if f.is_file():
                    try:
                        with Image.open(f) as im: im.verify()
                        with Image.open(f) as im:
                            im.load()
                            rows.append({"split":split,"class":cls.name,"format":im.format,"size":list(im.size),"mode":im.mode,"path":str(f.relative_to(DATA))})
                    except Exception as e: rows.append({"split":split,"class":cls.name,"path":str(f.relative_to(DATA)),"error":str(e)})
    return rows
