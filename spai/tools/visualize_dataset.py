import pandas as pd
import matplotlib.pyplot as plt
import random
import os
from PIL import Image

# Load CSV file
csv_path = "/home/pnair/spai/datasets/ldm_train_val_subset.csv"  # Update if needed
df = pd.read_csv(csv_path)

# Number of samples to visualize
num_samples = 9

# Randomly sample rows
samples = df.sample(n=num_samples).reset_index(drop=True)

# Create subplot
cols = 3
rows = (num_samples + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

for idx, ax in enumerate(axes.flatten()):
    if idx >= num_samples:
        ax.axis("off")
        continue
    row = samples.iloc[idx]
    img_path = row['image']  
    label = row['class'] 

    if not os.path.exists(img_path):
        ax.text(0.5, 0.5, "Missing image", ha='center', va='center')
        ax.axis("off")
        continue

    try:
        img = Image.open(img_path).convert('RGB')
        ax.imshow(img)
        ax.set_title(f"Class: {label}")
        ax.axis("off")
    except Exception as e:
        ax.text(0.5, 0.5, f"Error loading image:\n{e}", ha='center', va='center')
        ax.axis("off")

plt.tight_layout()
plt.savefig("/home/pnair/spai/job_outputs/vis_data.png")
plt.show()
