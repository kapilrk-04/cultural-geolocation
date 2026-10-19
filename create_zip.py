import os
import random
import zipfile

# CONFIG
base_dir = "attention_vizls"
targets = ["bg_bg", "es_arg"]
num_samples = 10
output_zip = "sampled_folders.zip"

selected_folders = []

# Collect folders
for t in targets:
    matching = [f for f in os.listdir(base_dir)
                if f.startswith(t) and os.path.isdir(os.path.join(base_dir, f))]
    
    if len(matching) < num_samples:
        print(f"Warning: only {len(matching)} folders found for {t}")
    
    sampled = random.sample(matching, min(num_samples, len(matching)))
    selected_folders.extend(sampled)

# Create zip
with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as z:
    for folder in selected_folders:
        folder_path = os.path.join(base_dir, folder)
        for root, _, files in os.walk(folder_path):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, base_dir)
                z.write(full_path, arcname)

print(f"Created {output_zip} with {len(selected_folders)} folders.")