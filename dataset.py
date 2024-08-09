
import torch
from PIL import Image
import os 
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_labels = {}

        # Create a mapping of class labels to integers
        self.class_labels = {}
        class_idx = 0

        # Iterate over sub-directories
        for class_dir in sorted(os.listdir(self.root_dir)):
            class_dir_path = os.path.join(self.root_dir, class_dir)
            if os.path.isdir(class_dir_path):
                self.class_labels[class_dir] = class_idx
                class_idx += 1

                # Iterate over images in the sub-directory
                for img_filename in os.listdir(class_dir_path):
                    if img_filename.endswith(".jpg"):
                        img_path = os.path.join(class_dir_path, img_filename)
                        self.images.append(img_path)
                        self.labels.append(self.class_labels[class_dir])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx])        
        label = self.labels[idx]
        
        if image.mode == "L":
            image = Image.merge("RGB", (image, image, image))
        if self.transform:
            image = self.transform(image)
        return image, label
    

# pytorch dataloader
def model_dataloader(weights, transform):
    
    weights = weights
    
    data_folder = "./nycu-2023-deep-learning-final-project-1/achieve/"

    train_folder = data_folder + "/train"
    val_folder = data_folder + "/valid"
    test_folder = data_folder + "/test"
    
    # pytorch dataset
    train_dataset = ImageDataset(train_folder, transform = transform)
    val_dataset = ImageDataset(val_folder, transform = transform)
    test_dataset = ImageDataset(test_folder, transform = transform)
    
    # pytorch dataloader
    train_dataloader = DataLoader(dataset = train_dataset, batch_size = 32, shuffle = True)
    val_dataloader = DataLoader(dataset = val_dataset, batch_size = 32, shuffle = False)
    test_dataloader = DataLoader(dataset = test_dataset, batch_size = 32, shuffle = False)
    
    return train_dataloader, val_dataloader, test_dataloader


# Train -> train_loss, train_acc
def train (model, dataloader, loss_fn, optimizer, device):
    train_loss, train_acc = 0, 0
    
    model.to(device)
    model.train()
    
    for batch, (x, y) in enumerate (dataloader):
        x, y = x.to(device), y.to(device)
        
        train_pred = model(x)
        
        loss = loss_fn(train_pred, y)
        train_loss = train_loss + loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_pred_label = torch.argmax(torch.softmax(train_pred, dim = 1), dim = 1)
        train_acc = train_acc + (train_pred_label == y).sum().item() / len(train_pred)
    
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    
    return train_loss, train_acc

# Validation -> val_loss, val_acc
def val (model, dataloader, loss_fn, device):
    val_loss, val_acc = 0, 0
    
    model.to(device)
    model.eval()
    
    with torch.inference_mode():
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            val_pred = model(x)
            
            loss = loss_fn(val_pred, y)
            val_loss = val_loss + loss.item()
            
            val_pred_label = torch.argmax(torch.softmax(val_pred, dim = 1), dim = 1)
            val_acc = val_acc + (val_pred_label == y).sum().item() / len(val_pred)
        
        val_loss = val_loss / len(dataloader)
        val_acc = val_acc / len(dataloader)
        
        return val_loss, val_acc
