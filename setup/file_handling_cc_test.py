import h5py
import os
import json
import argparse
from tqdm import tqdm
import logging
from multiprocessing import Pool
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import csv
import pandas as pd
logging.basicConfig(level=logging.INFO)

def has_annotations(hdf5_file_path):
    try:
        with h5py.File(hdf5_file_path, 'r', libver='latest', swmr=True) as hdf5:
            label_data = hdf5['label']['bounding_box_annotations'][...]
            if label_data.ndim == 0:
                label_data_str = label_data.item()
            else:
                label_data_str = label_data.decode('utf-8') if isinstance(label_data, bytes) else label_data
            annotations = json.loads(label_data_str)
            for annotation in annotations:
                bbox = annotation['bbox']
                category_id = annotation['category_id']
                if bbox != [0, 0, 0, 0] or category_id != 0:
                    return True
        return False
    except:
        print(hdf5_file_path)
        return False
    

def update_dataframe(csv_file):
    df = pd.read_csv(csv_file)
    df['has_annotations'] = df['hdf5_file_path'].apply(has_annotations)
    filtered_df = df[df['has_annotations']].reset_index(drop=True)
    filtered_csv_file = csv_file.replace('.csv', '_filtered.csv')
    filtered_df.to_csv(filtered_csv_file, index=False)
    print(f"Filtered DataFrame saved to {filtered_csv_file}")


def parse_bounding_boxes(json_data):
    processed_boxes = []
    for box in json_data:
        processed_boxes.append({
            "id": box["id"],
            "image_id": box["image_id"],
            "category_id": box["category_id"],
            "bbox": box["bbox"],
            "iscrowd": box["iscrowd"]
        })
    return processed_boxes

def convert_image(args):
    image_id, hdf5_file_path, output_dir = args
    try:
        with h5py.File(hdf5_file_path, 'r') as hdf5:
            if image_id not in hdf5['image']:
                logging.warning(f"Image ID {image_id} not found in the HDF5 file.")
                return None

            image = hdf5['image'][image_id][...]
            image = np.array(image, dtype=np.float32)

            labels = {}
            for label_type in hdf5['label']:
                try:
                    if label_type == "bounding_box_annotations":
                        label_data = hdf5['label'][label_type][image_id][...]
                        if label_data.ndim == 0:
                            label_data_str = label_data.item()
                        else:
                            label_data_str = label_data.decode('utf-8') if isinstance(label_data, bytes) else label_data
                        label_data = parse_bounding_boxes(json.loads(label_data_str))
                        labels[label_type] = label_data
                    else:
                        label_data = hdf5['label'][label_type][image_id][...]
                        labels[label_type] = np.array(label_data, dtype=np.int64)
                except KeyError as e:
                    logging.warning(f"Error loading label {label_type} for image {image_id}: {e}")
                    labels[label_type] = None

            metadata = {key: hdf5['image'][image_id].attrs[key] for key in hdf5['image'][image_id].attrs}

        output_file = os.path.join(output_dir, f"{image_id}.h5")
        with h5py.File(output_file, 'w') as output_hdf5:
            output_hdf5.create_dataset('image', data=image)
            for label_type, label_data in labels.items():
                if label_data is not None:
                    if label_type == "bounding_box_annotations":
                        json_str = json.dumps(label_data)
                        dt = h5py.special_dtype(vlen=bytes)
                        output_hdf5.create_dataset(f'label/{label_type}', data=json.dumps(label_data).encode('utf-8'), dtype=dt)
                    else:
                        output_hdf5.create_dataset(f'label/{label_type}', data=label_data)

            for key, value in metadata.items():
                output_hdf5['image'].attrs[key] = value

        return image_id, output_file
    except Exception as e:
        logging.error(f"Error processing image {image_id}: {e}")
        return None

def convert_hdf5_to_individual_hdf5(hdf5_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with h5py.File(hdf5_file, 'r') as hdf5:
        image_ids = list(hdf5['image'].keys())
    results = []
    with Pool() as pool:
        for result in tqdm(pool.imap_unordered(convert_image, [(image_id, hdf5_file, output_dir) for image_id in image_ids]), total=len(image_ids), desc='Processing images'):
            if result:
                results.append(result)
    return results

def write_csv(file_path, data):
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_id', 'hdf5_file_path'])
        writer.writerows(data)

def main():
    parser = argparse.ArgumentParser(description="Read directory of h5 files big")
    parser.add_argument('file_path', type=str, help='directory of h5 files')
    args = parser.parse_args()

    print("Test Set")
    hdf5_file = os.path.join(args.file_path,'downloads','TestSet.h5')
    output_dir = os.path.join(args.file_path,'downloads','adw-files-split','test')
    os.makedirs(output_dir, exist_ok=True)
    converted_files = convert_hdf5_to_individual_hdf5(hdf5_file, output_dir)
    logging.info(f"Number of converted test files: {len(converted_files)}")
    output_dir_csv = os.path.join(args.file_path,'downloads','adw-files-split','test_csv_file.csv')
    write_csv(output_dir_csv, converted_files)
    df = pd.read_csv(output_dir_csv)
    print(len(df))
    update_dataframe(output_dir_csv)
    output_dir_csv = os.path.join(args.file_path,'downloads','adw-files-split','test_csv_file_filtered.csv')
    df = pd.read_csv(output_dir_csv)
    print(len(df))
    print("Splitting Complete")

if __name__ == "__main__":
    main()

