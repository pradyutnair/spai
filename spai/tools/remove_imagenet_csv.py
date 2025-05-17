import pandas as pd
from tqdm import tqdm
import os
dataset_path = "/home/pnair/spai/datasets/"

# Get all the files in the dataset path
files = os.listdir(dataset_path)

# Get all the files in the dataset path that are csv files and do not contain "ldm"
#csv_files = [file for file in files if file.endswith(".csv") and "ldm" not in file]
#csv_files = ["test_set_sd3_fixed.csv"]
# Remove the lines containing "imagenet" in the csv files
# for file in tqdm(csv_files, desc="Processing files"):
#     df = pd.read_csv(os.path.join(dataset_path, file))
#     # Go through each row in the dataframe
#     for index, row in df.iterrows():
#         # If the row contains "imagenet" in the image_path, remove the row
#         if "imagenet" in row["image"]:
#             df = df.drop(index)
#     # New file name
#     new_file_name = file.replace(".csv", "_no_imagenet.csv")
#     # Save the dataframe to a new csv file
#     df.to_csv(os.path.join(dataset_path, new_file_name), index=False)

ldm_train_val = pd.read_csv(os.path.join(dataset_path, "ldm_train_val_subset.csv"))
#ldm_test = pd.read_csv(os.path.join(dataset_path, "ldm_test.csv"))
lsun_train_val = pd.read_csv(os.path.join(dataset_path, "lsun_train_val.csv"))
##lsun_test = pd.read_csv(os.path.join(dataset_path, "lsun_test.csv"))

# Add lsun_train_val to ldm_train_val and lsun_test to ldm_test
ldm_train_val = pd.concat([ldm_train_val, lsun_train_val])
#ldm_test = pd.concat([ldm_test, lsun_test])
ldm_train_val.to_csv(os.path.join(dataset_path, "ldm_lsun_train_val_subset.csv"), index=False)
#ldm_test.to_csv(os.path.join(dataset_path, "ldm_lsun_test.csv"), index=False)









