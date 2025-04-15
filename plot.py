import matplotlib.pyplot as plt
import logging
from models.fcs import FCS
from utils.data_manager import DataManager

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
}

data_manager = DataManager(
    dataset_name=args["dataset"], shuffle=True, seed=42, init_cls=args["init_cls"], increment=args["increment"]
)

fcs_model = FCS(args)

# Prepare data loaders
train_loader = None  # Replace with actual DataLoader for training
test_loader = None   # Replace with actual DataLoader for testing

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
