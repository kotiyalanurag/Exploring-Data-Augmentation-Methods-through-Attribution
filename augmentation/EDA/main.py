import os
import subprocess

folder_path = "/Users/anuragkotiyal/Desktop/Master Thesis/Original Subsets Text"  
augment_script_path = "/Users/anuragkotiyal/Desktop/Master Thesis/augmentation/Easy Data Augmentation/code/augment.py" 

output_dir = "/Users/anuragkotiyal/Desktop/Master Thesis/EDA"

for filename in os.listdir(folder_path):
    if filename.endswith(".txt") and not any(f"_aug_samples_" in filename for f in [filename]):
        input_path = os.path.join(folder_path, filename)
        base_name = filename.replace(".txt", "")

        for i in range(1, 4):
            output_filename = f"{base_name}_aug_samples_{i}.txt"
            output_path = os.path.join(output_dir, output_filename)

            command = [
                "python",
                augment_script_path,
                f"--input={input_path}",
                f"--output={output_path}",
                f"--num_aug={i}"
            ]
    
            print(f"🚀 Running: {' '.join(command)}")
            subprocess.run(command)
            print(f"✅ Created: {output_filename}")