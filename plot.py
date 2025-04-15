import matplotlib.pyplot as plt
import logging
from models.fcs import FCS
from utils.data_manager import DataManager
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Define temperatures to test
temperatures = [0.07, 0.1, 0.5]

# Initialize DataManager and FCS model
args = {
    "contrast_T": 0.1,  # Default temperature
    "lambda_contrast": 1.0,
    "batch_size": 64,
    "device": ["cuda"],
    "epochs": 10,
    "lr": 0.001,
    "weight_decay": 1e-4,
    "step_size": 30,
    "gamma": 0.1,
    "dataset": "cifar100",
    "memory_size": 2000,
    "init_cls": 10,
    "increment": 10,
    "log_dir": "./logs",
    "log_name": "temperature_analysis",
    "model_name": "FCS",
    "convnet_type": "resnet18_cbam",
}

data_manager = DataManager(
    dataset_name=args["dataset"], shuffle=True, seed=42, init_cls=args["init_cls"], increment=args["increment"]
)

fcs_model = FCS(args)

# Prepare data loaders
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.FakeData(transform=transform)  # Replace FakeData with actual dataset
train_loader = DataLoader(dataset, batch_size=args["batch_size"], shuffle=True)
test_loader = DataLoader(dataset, batch_size=args["batch_size"], shuffle=False)

# Ensure train_loader and test_loader are not None
if train_loader is None or test_loader is None:
    raise ValueError("train_loader and test_loader must be initialized with valid DataLoader objects.")

# Train with varying temperatures
results = fcs_model.train_with_temperatures(train_loader, test_loader, temperatures)

# Plot results
temps = list(results.keys())
forgetting = list(results.values())

plt.figure(figsize=(8, 6))
plt.plot(temps, forgetting, marker='o', linestyle='-', color='b')
plt.title('Temperature vs. Average Forgetting')
plt.xlabel('Temperature')
plt.ylabel('Average Forgetting')
plt.grid(True)
plt.savefig('temperature_vs_forgetting.png')
plt.show()
