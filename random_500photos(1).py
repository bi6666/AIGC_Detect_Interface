import os
import random
import shutil
from pathlib import Path

CURRENT_DATASET = "cyclegan"  # 当前数据集前缀
target_dataset = CURRENT_DATASET

def create_random_sample(target_dataset="ADM", sample_size=500):
    """
    从指定数据集中随机抽取图片
    
    Args:
        target_dataset: 目标数据集前缀 (例如: "ADM", "DDPM", "GLIDE" 等)
        sample_size: 每个类别要抽取的图片数量
    """
    # Define source and destination paths
    source_real = "0_real"
    source_fake = "1_fake"
    dest_real = "real_test"
    dest_fake = "fake_test"
    
    # Create destination directories if they don't exist
    os.makedirs(dest_real, exist_ok=True)
    os.makedirs(dest_fake, exist_ok=True)
    
    # Clear existing files in destination directories
    for file in os.listdir(dest_real):
        os.remove(os.path.join(dest_real, file))
    for file in os.listdir(dest_fake):
        os.remove(os.path.join(dest_fake, file))
    
    print(f"Extracting images from dataset: {target_dataset}")
    print(f"Sample size per category: {sample_size}")
    
    # Get list of image files (common image extensions)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    # Get real images from target dataset
    if os.path.exists(source_real):
        all_real_files = [f for f in os.listdir(source_real) 
                         if Path(f).suffix.lower() in image_extensions]
        
        # Filter files by dataset prefix
        target_real_files = [f for f in all_real_files 
                           if f.lower().startswith(f"{target_dataset}_")]
        
        print(f"Found {len(all_real_files)} total real images")
        print(f"Found {len(target_real_files)} real images from {target_dataset} dataset")
        
        if target_real_files:
            # Randomly select images
            selected_real = random.sample(target_real_files, min(sample_size, len(target_real_files)))
            
            # Copy selected images to destination
            for img in selected_real:
                src_path = os.path.join(source_real, img)
                dst_path = os.path.join(dest_real, img)
                shutil.copy2(src_path, dst_path)
            
            print(f"Copied {len(selected_real)} real images from {target_dataset} to {dest_real}")
        else:
            print(f"No real images found for dataset {target_dataset}")
    else:
        print(f"Source folder {source_real} not found!")
    
    # Get fake images from target dataset
    if os.path.exists(source_fake):
        all_fake_files = [f for f in os.listdir(source_fake) 
                         if Path(f).suffix.lower() in image_extensions]
        
        # Filter files by dataset prefix
        target_fake_files = [f for f in all_fake_files 
                           if f.lower().startswith(f"{target_dataset}_")]
        
        print(f"Found {len(all_fake_files)} total fake images")
        print(f"Found {len(target_fake_files)} fake images from {target_dataset} dataset")
        
        if target_fake_files:
            # Randomly select images
            selected_fake = random.sample(target_fake_files, min(sample_size, len(target_fake_files)))
            
            # Copy selected images to destination
            for img in selected_fake:
                src_path = os.path.join(source_fake, img)
                dst_path = os.path.join(dest_fake, img)
                shutil.copy2(src_path, dst_path)
            
            print(f"Copied {len(selected_fake)} fake images from {target_dataset} to {dest_fake}")
        else:
            print(f"No fake images found for dataset {target_dataset}")
    else:
        print(f"Source folder {source_fake} not found!")
    
    print(f"Random sampling from {target_dataset} dataset completed!")


if __name__ == "__main__":

    # 列出可用的数据集
    available_datasets = ['progan', 'stylegan', 'biggan', 'cyclegan', 'stargan', 'gaugan',
                         'stylegan2', 'whichfaceisreal', 'adm', 'glide', 'midjourney', 'stable_diffusion_v_1_4',
                         'stable_diffusion_v_1_5', 'vqdm', 'wukong', 'dalle2', 'chameleon']
    print("Available datasets:")
    for i, dataset in enumerate(available_datasets, 1):
        print(f"  {i}. {dataset}")
    
    print("\n" + "="*50)
    
    # 可以在这里修改要抽取的数据集和样本大小
    target_dataset = CURRENT_DATASET.lower()
    sample_size = 500       # 修改这里来改变抽取的样本数量
    
    # 检查目标数据集是否可用
    if target_dataset in available_datasets:
        create_random_sample(target_dataset, sample_size)
    else:
        print(f"Dataset '{target_dataset}' not found!")
        print(f"Available datasets: {available_datasets}")
