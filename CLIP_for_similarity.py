import torch
import clip
from PIL import Image
import os
from pathlib import Path
import numpy as np
from datetime import datetime
import shutil
import json

def load_clip_model():
    """加载CLIP模型"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 加载CLIP模型
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

def load_precomputed_vectors(vector_dir="clip_vectors"):
    """
    加载预计算的向量 - 适应子目录结构
    
    Args:
        vector_dir: 向量存储目录
    
    Returns:
        dict: {目录名: {"vectors": numpy数组, "filenames": 文件名列表}}
    """
    vector_path = Path(vector_dir)
    if not vector_path.exists():
        raise FileNotFoundError(f"Vector directory not found: {vector_path}")
    
    datasets = {}
    
    # 查找子目录中的向量文件
    for subdir in vector_path.iterdir():
        if subdir.is_dir():
            dataset_name = subdir.name
            vector_file = subdir / f"{dataset_name}_vectors.npy"
            filename_file = subdir / f"{dataset_name}_filenames.json"
            
            if vector_file.exists() and filename_file.exists():
                # 加载向量
                vectors = np.load(vector_file)
                
                # 加载文件名
                with open(filename_file, 'r', encoding='utf-8') as f:
                    filenames = json.load(f)
                
                datasets[dataset_name] = {
                    "vectors": vectors,
                    "filenames": filenames
                }
                print(f"Loaded {dataset_name}: {len(vectors)} vectors")
    
    return datasets

def extract_query_features(image_path, model, preprocess, device):
    """提取查询图片的特征向量"""
    try:
        # 打开并预处理图片
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        # 提取特征
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            # 标准化特征向量
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
        return image_features.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def find_similar_images_fast(query_image_path, search_datasets, top_k=10):
    """
    使用预计算向量快速寻找相似图片
    
    Args:
        query_image_path: 查询图片路径
        search_datasets: 预计算的数据集字典
        top_k: 返回最相似的图片数量
    
    Returns:
        List of tuples: (dataset_name, filename, similarity_score)
    """
    # 加载CLIP模型（仅用于查询图片）
    model, preprocess, device = load_clip_model()
    
    # 检查查询图片是否存在
    query_path = Path(query_image_path)
    if not query_path.exists():
        raise FileNotFoundError(f"Query image not found: {query_image_path}")
    
    print(f"Query image: {query_path.name}")
    
    # 提取查询图片的特征
    print("Extracting query image features...")
    query_features = extract_query_features(query_path, model, preprocess, device)
    if query_features is None:
        raise ValueError("Failed to extract query image features")
    
    # 计算与所有预计算向量的相似度
    all_similarities = []
    
    for dataset_name, dataset_data in search_datasets.items():
        print(f"Computing similarities with {dataset_name}...")
        vectors = dataset_data["vectors"]
        filenames = dataset_data["filenames"]
        
        # 计算相似度 (向量化操作，非常快)
        similarities = np.dot(vectors, query_features)
        
        # 收集结果
        for i, similarity in enumerate(similarities):
            # 跳过查询图片本身
            if filenames[i] != query_path.name:
                all_similarities.append((dataset_name, filenames[i], float(similarity)))
    
    # 按相似度排序，取top_k
    all_similarities.sort(key=lambda x: x[2], reverse=True)
    top_similarities = all_similarities[:top_k]
    
    return top_similarities

def save_fast_results(query_image_path, top_similarities, base_dirs, output_dir="similarity_results"):
    """保存快速相似度搜索结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(output_dir) / f"similarity_search_fast_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制查询图片到结果目录
    query_path = Path(query_image_path)
    shutil.copy2(query_path, results_dir / f"query_{query_path.name}")
    
    # 保存结果文本文件
    results_file = results_dir / "similarity_results.txt"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write(f"CLIP Image Similarity Search Results (Fast Mode)\n")
        f.write(f"Search Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"=" * 60 + "\n\n")
        
        f.write(f"Query Image: {query_path.name}\n")
        f.write(f"Total Results: {len(top_similarities)}\n\n")
        
        f.write(f"Top {len(top_similarities)} Most Similar Images:\n")
        f.write(f"-" * 50 + "\n")
        
        for i, (dataset_name, filename, similarity) in enumerate(top_similarities, 1):
            f.write(f"{i:2d}. {filename} (from {dataset_name})\n")
            f.write(f"    Similarity Score: {similarity:.4f}\n")
            if dataset_name in base_dirs:
                full_path = Path(base_dirs[dataset_name]) / filename
                f.write(f"    Path: {full_path}\n")
            f.write("\n")
    
    # 复制相似图片到结果目录
    similar_images_dir = results_dir / "similar_images"
    similar_images_dir.mkdir(exist_ok=True)
    
    for i, (dataset_name, filename, similarity) in enumerate(top_similarities, 1):
        if dataset_name in base_dirs:
            source_path = Path(base_dirs[dataset_name]) / filename
            if source_path.exists():
                # 重命名文件，包含排名和相似度分数
                new_name = f"{i:02d}_score_{similarity:.4f}_{dataset_name}_{filename}"
                shutil.copy2(source_path, similar_images_dir / new_name)
    
    print(f"\nResults saved to: {results_dir}")
    print(f"Query image copied to: {results_dir / f'query_{query_path.name}'}")
    print(f"Similar images copied to: {similar_images_dir}")
    print(f"Results summary saved to: {results_file}")

def main():
    """主函数"""
    # 配置参数
    query_image_path = "test1.png"
    vector_dir = "clip_vectors"  # 现在指向包含子目录的主目录
    top_k = 10

    # 基础目录映射（用于复制文件）
    base_dirs = {
        "0_real": "0_real",
        "1_fake": "1_fake",
    }
    
    print("=" * 60)
    print("CLIP Image Similarity Search (Fast Mode)")
    print("=" * 60)
    
    try:
        # 加载预计算的向量
        print("Loading precomputed vectors...")
        search_datasets = load_precomputed_vectors(vector_dir)
        
        if not search_datasets:
            print("No precomputed vectors found. Please run save_vector_2_numpy.py first.")
            return
        
        print(f"Available datasets: {list(search_datasets.keys())}")
        
        # 执行快速相似度搜索
        top_similarities = find_similar_images_fast(query_image_path, search_datasets, top_k)
        
        if not top_similarities:
            print("No similar images found.")
            return
        
        # 显示结果
        print(f"\nTop {len(top_similarities)} Most Similar Images:")
        print("-" * 50)
        
        for i, (dataset_name, filename, similarity) in enumerate(top_similarities, 1):
            print(f"{i:2d}. {filename} (from {dataset_name})")
            print(f"    Similarity Score: {similarity:.4f}")
            
            # 简单的相似度解读
            if similarity > 0.95:
                interpretation = "几乎一模一样"
            elif similarity > 0.85:
                interpretation = "高度相似"
            elif similarity > 0.75:
                interpretation = "相似"
            elif similarity > 0.65:
                interpretation = "较为相似"
            else:
                interpretation = "相关性较低"
            
            print(f"    Interpretation: {interpretation}")
            print()
        
        # 保存结果
        save_fast_results(query_image_path, top_similarities, base_dirs)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
