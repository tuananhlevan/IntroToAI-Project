import os
from torchvision.datasets import ImageFolder

# 1. Define your dataset roots
train_dir = "IntroToAI-Project/data_split/KidneyCancer/train"
val_dir = "IntroToAI-Project/data_split/KidneyCancer/val"

# 2. Create your datasets (no loaders needed yet)
# We don't even need to pass transforms
train_dataset = ImageFolder(root=train_dir)
val_dataset = ImageFolder(root=val_dir)

# 3. Get the list of file paths from each dataset
# train_dataset.samples is a list of tuples: [('path/to/file.jpg', 0), ...]
train_paths = [sample[0] for sample in train_dataset.samples]
val_paths = [sample[0] for sample in val_dataset.samples]

# 4. Get just the filenames (the safest check)
# This finds 'img1.jpg' even if one is in 'train/cat/' and one is in 'val/cat/'
train_filenames = set(os.path.basename(p) for p in train_paths)
val_filenames = set(os.path.basename(p) for p in val_paths)

# 5. Check for overlap
overlap = train_filenames.intersection(val_filenames)

if not overlap:
    print("✅ No overlap detected. Your folders are clean.")
else:
    print(f"⚠️ DANGER: Found {len(overlap)} overlapping files!")
    print("Overlapping files:")
    for filename in overlap:
        print(filename)