import torch
import clip
from PIL import Image
import os
from pathlib import Path
import numpy as np
from datetime import datetime
import json

def load_clip_model():
    """加载CLIP模型"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 加载CLIP模型
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

def get_image_files(directory):
    """获取目录下所有图片文件"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    for file_path in directory.iterdir():
        if file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)
    
    return image_files

def extract_image_features(image_path, model, preprocess, device):
    """提取单张图片的特征向量"""
    try:
        # 打开并预处理图片
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        # 提取特征
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            # 标准化特征向量
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
        return image_features.cpu().numpy()
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def extract_and_save_vectors(directories, output_dir="clip_vectors"):
    """
    为多个目录提取CLIP向量并保存
    
    Args:
        directories: 目录路径列表或字典 {名称: 路径}
        output_dir: 输出目录
    """
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 加载CLIP模型
    model, preprocess, device = load_clip_model()
    
    # 如果directories是列表，转换为字典
    if isinstance(directories, list):
        directories = {Path(d).name: d for d in directories}
    
    # 处理每个目录
    for dir_name, dir_path in directories.items():
        print(f"\nProcessing directory: {dir_name}")
        print(f"Path: {dir_path}")
        
        # 为每个目录创建子文件夹
        dir_output_path = output_path / dir_name
        dir_output_path.mkdir(exist_ok=True)
        
        # 获取图片文件
        image_files = get_image_files(dir_path)
        print(f"Found {len(image_files)} images")
        
        if len(image_files) == 0:
            print(f"No images found in {dir_path}")
            continue
        
        # 存储向量和文件名
        vectors = []
        filenames = []
        failed_files = []
        
        # 处理每张图片
        for i, image_path in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {image_path.name}")
            
            # 提取特征向量
            features = extract_image_features(image_path, model, preprocess, device)
            
            if features is not None:
                vectors.append(features.flatten())  # 展平为1D向量
                filenames.append(image_path.name)
            else:
                failed_files.append(image_path.name)
        
        # 转换为numpy数组
        if vectors:
            vectors_array = np.array(vectors)
            print(f"Extracted {len(vectors)} vectors, shape: {vectors_array.shape}")
            
            # 保存向量数组到对应的子文件夹
            vector_file = dir_output_path / f"{dir_name}_vectors.npy"
            np.save(vector_file, vectors_array)
            print(f"Vectors saved to: {vector_file}")
            
            # 保存文件名列表到对应的子文件夹
            filename_file = dir_output_path / f"{dir_name}_filenames.json"
            with open(filename_file, 'w', encoding='utf-8') as f:
                json.dump(filenames, f, ensure_ascii=False, indent=2)
            print(f"Filenames saved to: {filename_file}")
            
            # 保存失败文件列表（如果有）到对应的子文件夹
            if failed_files:
                failed_file = dir_output_path / f"{dir_name}_failed.json"
                with open(failed_file, 'w', encoding='utf-8') as f:
                    json.dump(failed_files, f, ensure_ascii=False, indent=2)
                print(f"Failed files saved to: {failed_file}")
        else:
            print(f"No vectors extracted from {dir_name}")
    
    # 保存处理信息到主输出目录
    info_file = output_path / "processing_info.json"
    processing_info = {
        "processing_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "clip_model": "ViT-B/32",
        "directories_processed": list(directories.keys()),
        "vector_dimension": 512,  # ViT-B/32的特征维度
        "device_used": device
    }
    
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(processing_info, f, ensure_ascii=False, indent=2)
    
    print(f"\nProcessing complete!")
    print(f"All results saved to: {output_path}")
    print(f"Processing info saved to: {info_file}")

def main():
    """主函数"""
    # 定义要处理的目录
    directories = {
        "0_real": "0_real",
        "1_fake": "1_fake",
    }
    
    print("=" * 60)
    print("CLIP Vector Extraction and Storage")
    print("=" * 60)
    
    try:
        # 提取并保存向量
        extract_and_save_vectors(directories)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
