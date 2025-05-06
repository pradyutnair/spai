from pathlib import Path

import tensorflow_datasets as tfds

# lsun_categories = [
#     "classroom", "bedroom", "bridge", "church_outdoor", "conference_room",
#     "dining_room", "kitchen", "living_room", "restaurant", "tower",
#     "airplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
#     "chair", "cow", "dining_table", "dog", "horse", "motorbike", "potted_plant",
#     "sheep", "sofa", "train", "tv-monitor"
# ]
lsun_categories = ["bedroom"]

data_dir = "/scratch-shared/dl2_spai/datasets/LSUN"

for category in lsun_categories:
    try:
        print(f"⬇️ Downloading and loading 1% of lsun/{category}...")
        ds = tfds.load(f'lsun/{category}', split='train[:1%]', data_dir=data_dir, shuffle_files=False)
        print(f"✅ Loaded lsun/{category} sample successfully.\\n")
    except Exception as e:
        print(f"❌ Failed to load lsun/{category}: {e}\\n")

