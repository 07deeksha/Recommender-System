import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split
import shutil
from PIL import Image


if __name__ == '__main__':
    # Load tag_list.csv
    tag_list_path = r'C:\Users\Deeksha\Desktop\TrendPulse\Tp\new_txt_lst.csv'
    tag_data = pd.read_csv(tag_list_path)

    # Extract image names and tags
    image_names = tag_data['file'].tolist()
    tags_associated = tag_data['tags'].tolist()

    # Shuffle data
    shuffle_with_order = list(zip(image_names, tags_associated))
    random.shuffle(shuffle_with_order)
    image_names, tags_associated = zip(*shuffle_with_order)

    # Split data into train, validation, and test sets
    img_train_val, img_test, tags_train_val, tags_test = train_test_split(image_names, tags_associated, test_size=0.2, random_state=0)
    img_train, img_val, tags_train, tags_val = train_test_split(img_train_val, tags_train_val, test_size=0.25, random_state=0)  # 0.25 x 0.8 = 0.2

    # Destination directories
    dest_train = r'C:\Users\Deeksha\Desktop\TrendPulse\Tp\TRAIN'
    dest_val = r'C:\Users\Deeksha\Desktop\TrendPulse\Tp\VALIDATION'
    dest_test = r'C:\Users\Deeksha\Desktop\TrendPulse\Tp\TEST'

    # Create train, validation, and test directories if not exist
    for directory in [dest_train, dest_val, dest_test]:
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Copy images to respective directories
        for img, tags, dest_dir in zip([img_train, img_val, img_test], [tags_train, tags_val, tags_test],
                                       [dest_train, dest_val, dest_test]):
            for image_name, tags_associated in zip(img, tags):
                full_name = os.path.join(r'C:\Users\Deeksha\Desktop\TrendPulse\Tp\images', image_name)
                image_details = [image_name, tags_associated, full_name]

                try:
                    # Copy image to destination directory
                    shutil.copy(full_name, dest_dir)

                    # Display image details
                    with Image.open(full_name) as img:
                        if hasattr(img, 'bits'):
                            print(img.bits, img.size, img.format)
                        else:
                            print("Bits: N/A", img.size, img.format)

                except FileNotFoundError:
                    print(f"File not found: {full_name}. Removing corresponding instance from CSV.")
                    # Remove corresponding instance (file, tag) from the CSV
                    tag_data = tag_data[tag_data['file'] != image_name]
                    tag_data.to_csv(tag_list_path, index=False)