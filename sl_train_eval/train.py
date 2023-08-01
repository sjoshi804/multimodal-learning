import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from dataset import ImageLabelDataset
from torchvision.transforms import InterpolationMode
from tqdm import tqdm 

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set hyperparameters
num_epochs = 30
batch_size = 256
learning_rate = 0.1

# Load and preprocess data
transform = transforms.Compose([
    transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

train_dataset = ImageLabelDataset(root="", csv_path="sl_imagenet_like_cc.csv", transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Load pre-trained ResNet50 model
print(f"Dataset: num-classes={train_dataset.num_classes}, size={len(train_dataset)}")
model = models.resnet50()
num_classes = train_dataset.num_classes

# Replace the last fully connected layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
print("Training")
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}"):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}")

# Save the model checkpoint
torch.save(model.state_dict(), "resnet50_conceptual_captions.ckpt")
