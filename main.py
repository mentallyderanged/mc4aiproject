from PIL import Image
import os
import numpy as np

ds_path = input("Enter the path to the sample dataset directory: ").strip()
folders = os.listdir(ds_path)
label_mapping = {folder: i for i, folder in enumerate(folders)}
X,y = [],[]

# read files in folder
for folder in folders:
  files = os.listdir(os.path.join(ds_path, folder))
  for f in files:
    if f.endswith('.png'):
      img = Image.open(os.path.join(ds_path, folder, f))
      img = np.array(img)
      X.append(img)
      y.append(label_mapping[folder])

print(X,y)
