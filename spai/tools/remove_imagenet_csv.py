import pandas as pd
from tqdm import tqdm
import os
dataset_path = "/home/pnair/spai/datasets/"

# Get all the files in the dataset path
files = os.listdir(dataset_path)

# Get all the files in the dataset path that are csv files and do not contain "ldm"
csv_files = [file for file in files if file.endswith(".csv") and "ldm" not in file]

# Remove the lines containing "imagenet" in the csv files
for file in tqdm(csv_files, desc="Processing files"):
    df = pd.read_csv(os.path.join(dataset_path, file))
    # Go through each row in the dataframe
    for index, row in df.iterrows():
        # If the row contains "imagenet" in the image_path, remove the row
        if "imagenet" in row["image"]:
            df = df.drop(index)
    # New file name
    new_file_name = file.replace(".csv", "_no_imagenet.csv")
    # Save the dataframe to a new csv file
    df.to_csv(os.path.join(dataset_path, new_file_name), index=False)








