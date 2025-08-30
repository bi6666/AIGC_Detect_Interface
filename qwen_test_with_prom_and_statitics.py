import os
import torch
import clip
from openai import OpenAI
from pathlib import Path
import random
from PIL import Image
import base64
import numpy as np
import json
import time
import shutil
from datetime import datetime
from collections import defaultdict

# Qwen API configuration
API_KEY = "sk-c08fe5d86cd048bda352ed7aed6168cb"  # Replace with your actual API key
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # Qwen API base URL

# Configure the OpenAI client for Qwen
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

def load_clip_model():
    """加载CLIP模型"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 加载CLIP模型
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

def load_precomputed_vectors(vector_dir="clip_vectors"):
    """加载预计算的向量 - 适应子目录结构"""
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

def find_similar_images_for_prompt(query_image_path, search_datasets, base_dirs, image_source_mapping, top_k=10):
    """找到相似图片用于提示 - 从0_real和1_fake中各选择5个，并返回来源信息"""
    # 加载CLIP模型
    model, preprocess, device = load_clip_model()
    
    # 检查查询图片是否存在
    query_path = Path(query_image_path)
    if not query_path.exists():
        return []
    
    # 提取查询图片的特征
    query_features = extract_query_features(query_path, model, preprocess, device)
    if query_features is None:
        return []
    
    # 分别计算与0_real和1_fake的相似度
    real_similarities = []
    fake_similarities = []
    
    for dataset_name, dataset_data in search_datasets.items():
        vectors = dataset_data["vectors"]
        filenames = dataset_data["filenames"]
        
        # 计算相似度
        similarities = np.dot(vectors, query_features)
        
        # 收集结果，确定标签和来源
        for i, similarity in enumerate(similarities):
            # 跳过查询图片本身
            if filenames[i] != query_path.name and similarities[i] < 0.99:
                # 获取完整路径
                full_path = Path(base_dirs.get(dataset_name, '')) / filenames[i]
                if full_path.exists():
                    # 获取图片来源信息
                    image_source = get_image_source_from_mapping(filenames[i], 
                                                               'real' if dataset_name == '0_real' else 'fake', 
                                                               image_source_mapping)
                    
                    if dataset_name == '0_real':
                        real_similarities.append((dataset_name, filenames[i], float(similarity), 'real', str(full_path), image_source))
                    elif dataset_name == '1_fake':
                        fake_similarities.append((dataset_name, filenames[i], float(similarity), 'fake', str(full_path), image_source))
    
    # 从真实图片中选择前5个最相似的
    real_similarities.sort(key=lambda x: x[2], reverse=True)
    top_real = real_similarities[:5]
    
    # 从假图片中选择前5个最相似的
    fake_similarities.sort(key=lambda x: x[2], reverse=True)
    top_fake = fake_similarities[:5]
    
    # 合并结果并按相似度排序
    combined_results = top_real + top_fake
    combined_results.sort(key=lambda x: x[2], reverse=True)
    
    return combined_results[:top_k]

def image_to_base64(image_path):
    """Convert image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def classify_image_with_qwen_prompt(image_path, similar_examples):
    """使用相似图片作为提示来调用Qwen API分类图片 - 上传实际图片"""
    try:
        # Convert target image to base64
        target_image_b64 = image_to_base64(image_path)
        
        # Determine mime type for target image
        target_mime_type = "image/jpeg" if image_path.suffix.lower() in ['.jpg', '.jpeg'] else "image/png"
        
        # 构建提示信息
        prompt_text = """I will show you a series of example images with their correct labels (real or fake), followed by a target image that you need to classify.

EXAMPLE IMAGES WITH LABELS:
Below are example images similar to the target image, each clearly labeled as either 'real' (authentic photograph) or 'fake' (AI-generated/synthetic):

"""
        
        # 准备消息内容，从提示文本开始
        message_content = [
            {
                "type": "text",
                "text": prompt_text
            }
        ]
        
        # 添加相似图片作为实际的图像输入，每张图片都有清楚的标签
        example_count = 0
        for i, (dataset_name, filename, similarity, label, full_path) in enumerate(similar_examples, 1):
            if Path(full_path).exists() and example_count < 10:  # 限制最多10张示例图片
                try:
                    example_b64 = image_to_base64(Path(full_path))
                    example_mime = "image/jpeg" if Path(full_path).suffix.lower() in ['.jpg', '.jpeg'] else "image/png"
                    
                    # 为每张示例图片添加清楚的标签说明
                    label_text = f"\n--- EXAMPLE {i} ---\nFilename: {filename}\nLabel: {label.upper()}\nThis image is {label.upper()} ({'a real photograph' if label == 'real' else 'AI-generated/fake'})\nSimilarity to target: {similarity:.3f}\n"
                    
                    message_content.append({
                        "type": "text",
                        "text": label_text
                    })
                    
                    message_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{example_mime};base64,{example_b64}"
                        }
                    })
                    
                    example_count += 1
                    
                except Exception as e:
                    print(f"Warning: Could not load example image {full_path}: {e}")
        
        # 添加分析指令和目标图片
        analysis_instruction = f"""

--- TARGET IMAGE FOR CLASSIFICATION ---
Now analyze the following target image based on the {example_count} labeled examples above.

Look for:
1. Patterns and characteristics you observed in the REAL examples
2. Patterns and characteristics you observed in the FAKE examples  
3. Visual artifacts, inconsistencies, or unnatural elements
4. Lighting, shadows, and texture consistency
5. Overall image quality and generation artifacts

Based on your analysis of the examples and the target image, determine if the target image is:
- 'real' (authentic photograph)
- 'fake' (AI-generated/synthetic)

TARGET IMAGE TO CLASSIFY:
"""
        
        message_content.append({
            "type": "text",
            "text": analysis_instruction
        })
        
        # 添加目标图片
        message_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{target_mime_type};base64,{target_image_b64}"
            }
        })
        
        # 添加最终指令
        final_instruction = "\nBased on your analysis of the examples and this target image, respond with only 'real'(authentic photograph) or 'fake'(AI-generated/synthetic)."
        message_content.append({
            "type": "text",
            "text": final_instruction
        })
        
        # Create the message with all images
        response = client.chat.completions.create(
            model="qwen-vl-max",  # 使用支持视觉的模型
            messages=[
                {
                    "role": "user",
                    "content": message_content
                }
            ],
            max_tokens=50,
            temperature=0
        )
        
        text_response = response.choices[0].message.content.lower().strip()
        return 'real' if 'real' in text_response else 'fake'
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def load_image_source_mapping(mapping_file="image_source_mapping.json"):
    """加载图片来源映射信息"""
    mapping_path = Path(mapping_file)
    if mapping_path.exists():
        with open(mapping_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        print(f"Warning: Image source mapping file not found: {mapping_path}")
        return {}

def get_image_source_from_mapping(image_name, true_label, image_source_mapping):
    """从映射中获取图片来源信息"""
    # 默认先尝试从文件名前缀提取数据集名称
    dataset_name = "Unknown"
    
    # 先尝试从文件名前缀提取数据集名称
    if '_' in image_name:
        if "_real" in image_name:
            potential_dataset = image_name.split('_real')[0]
        elif "_fake" in image_name:
            potential_dataset = image_name.split('_fake')[0]
        # 检查是否是已知的数据集名称 - 转换为小写进行比较
        known_datasets = ['progan', 'stylegan', 'biggan', 'cyclegan', 'stargan', 'gaugan', 
                         'stylegan2', 'whichfaceisreal', 'adm', 'glide', 'midjourney', 'stable_diffusion_v_1_4', 
                         'stable_diffusion_v_1_5', 'vqdm', 'wukong', 'dalle2', 'chameleon']
        if potential_dataset.lower() in known_datasets:
            dataset_name = potential_dataset
        else:
            print(f"Warning: Unknown dataset name in image name: {potential_dataset}")
            exit()
    
    # 如果有映射文件，则使用映射文件中的信息
    if image_source_mapping:
        mapping_key = 'real_train' if true_label == 'real' else 'fake_train'
        
        if mapping_key in image_source_mapping and image_name in image_source_mapping[mapping_key]:
            source_info = image_source_mapping[mapping_key][image_name]
            dataset_name = source_info.get('dataset_name', dataset_name)
    
    return dataset_name

def main():
    # 配置参数
    vector_dir = "clip_vectors"
    
    # 基础目录映射
    base_dirs = {
        "0_real": "0_real",
        "1_fake": "1_fake",
        "real_test": "real_test",
        "fake_test": "fake_test"
    }
    
    # Define test directory paths
    real_test_dir = Path("real_test")
    fake_test_dir = Path("fake_test")
    
    # Check if test directories exist
    if not real_test_dir.exists() or not fake_test_dir.exists():
        print("Error: 'real_test' and 'fake_test' directories must exist in the current directory.")
        return
    
    # 加载预计算的向量
    try:
        print("Loading precomputed vectors...")
        search_datasets = load_precomputed_vectors(vector_dir)
        
        if not search_datasets:
            print("No precomputed vectors found. Please run save_vector_2_numpy.py first.")
            return
        
        # 检查是否有0_real和1_fake数据集
        if '0_real' not in search_datasets or '1_fake' not in search_datasets:
            print("Warning: Missing 0_real or 1_fake datasets. Available datasets:", list(search_datasets.keys()))
            
    except Exception as e:
        print(f"Error loading vectors: {e}")
        return
    
    # Get image files from test directories
    real_images = (list(real_test_dir.glob("*.jpg")) + 
                   list(real_test_dir.glob("*.jpeg")) + 
                   list(real_test_dir.glob("*.png")))
    fake_images = (list(fake_test_dir.glob("*.jpg")) + 
                   list(fake_test_dir.glob("*.jpeg")) + 
                   list(fake_test_dir.glob("*.png")))
    # Create test dataset
    test_data = []
    for img_path in real_images:
        test_data.append((img_path, 'real'))
    for img_path in fake_images:
        test_data.append((img_path, 'fake'))
    test_data = test_data[:250] + test_data[-250:]
        
    # Shuffle the test data
    random.shuffle(test_data)
    
    
    print(f"Testing {len(test_data)} images with similarity-based prompts...")
    print(f"Real images: {len(real_images)}, Fake images: {len(fake_images)}")
    print("-" * 70)
    
    correct_predictions = 0
    total_predictions = 0
    similar_source_statistics = defaultdict(lambda: {'count': 0, 'total_similarity': 0.0})
    results_by_source = []
    count = 0
    
    # 加载图片来源映射
    print("Loading image source mapping...")
    image_source_mapping = load_image_source_mapping()
    
    for i, (image_path, true_label) in enumerate(test_data, 1):
        count += 1
        print(f"Processing image {i}/{len(test_data)}: {image_path.name}")
        
        # 找到相似的图片作为提示 - 从0_real和1_fake中各选5个，包含来源信息
        print("  Finding similar images for prompt (5 real + 5 fake)...")
        similar_examples = find_similar_images_for_prompt(image_path, search_datasets, base_dirs, image_source_mapping, top_k=10)
        
        if similar_examples:
            print(f"  Found {len(similar_examples)} similar examples")
            # 统计真实和假图片的数量
            real_count = sum(1 for item in similar_examples if item[3] == 'real')
            fake_count = sum(1 for item in similar_examples if item[3] == 'fake')
            print(f"    Real examples: {real_count}, Fake examples: {fake_count}")
            
            # 统计相似图片的来源
            similar_sources_for_this_query = defaultdict(int)
            for j, (dataset_name, filename, similarity, label, full_path, source) in enumerate(similar_examples, 1):
                # 统计来源
                similar_source_statistics[source]['count'] += 1
                similar_source_statistics[source]['total_similarity'] += similarity
                similar_sources_for_this_query[source] += 1
                
                if j <= 3:  # 显示前3个最相似的例子
                    print(f"    {j}. {filename} ({label}, similarity: {similarity:.3f}, source: {source})")
            
            # 显示本次查询的来源分布
            print(f"    Source distribution for this query: {dict(similar_sources_for_this_query)}")
            
            # 检查示例图片是否存在
            valid_examples = []
            for example in similar_examples:
                if Path(example[4]).exists():
                    valid_examples.append(example)
            
            print(f"    Valid example images found: {len(valid_examples)}")
            
            if len(valid_examples) < 5:
                print("    Warning: Less than 5 valid example images found")
        else:
            print("  No similar examples found, proceeding without prompt")
            similar_sources_for_this_query = {}
        
        # 使用相似图片作为实际图像输入进行分类
        print("  Uploading example images and target image to Qwen...")
        # 转换similar_examples格式以适配classify_image_with_qwen_prompt函数
        examples_for_classification = [(item[0], item[1], item[2], item[3], item[4]) for item in similar_examples]
        prediction = classify_image_with_qwen_prompt(image_path, examples_for_classification)
        
        if prediction is not None:
            total_predictions += 1
            is_correct = prediction == true_label
            
            if is_correct:
                correct_predictions += 1
                result = "✓"
            else:
                result = "✗"
            
            # 记录详细结果
            results_by_source.append({
                'image_name': image_path.name,
                'true_label': true_label,
                'predicted_label': prediction,
                'is_correct': is_correct,
                'similar_examples': len(similar_examples),
                'similar_sources': dict(similar_sources_for_this_query)
            })
            
            print(f"  True: {true_label}, Predicted: {prediction} {result}")
        else:
            print(f"  Failed to get prediction")
        
        print()
    
    # Calculate and display results
    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        print("=" * 70)
        print(f"Results (With Similarity-Based Prompts):")
        print(f"Total images processed: {total_predictions}")
        print(f"Correct predictions: {correct_predictions}")
        print(f"Accuracy: {accuracy:.2f}%")
        
        # 显示检索到的相似图片的来源统计
        print(f"\nSimilar Images Source Statistics:")
        print(f"(Statistics of the top-10 most similar retrieved images)")
        for source, stats in similar_source_statistics.items():
            avg_similarity = stats['total_similarity'] / stats['count'] if stats['count'] > 0 else 0
            print(f"  {source}: {stats['count']} images (avg similarity: {avg_similarity:.3f})")
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = Path(f"qwen_test_withprompt_results_{timestamp}.txt")
        
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write(f"Qwen Image Classification Test Results (With Similarity-Based Prompts)\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"=" * 70 + "\n\n")
            
            f.write(f"Summary:\n")
            f.write(f"Total images processed: {total_predictions}\n")
            f.write(f"Correct predictions: {correct_predictions}\n")
            f.write(f"Incorrect predictions: {total_predictions - correct_predictions}\n")
            f.write(f"Accuracy: {accuracy:.2f}%\n\n")
            
            # 检索到的相似图片来源统计
            f.write(f"Similar Images Source Statistics:\n")
            f.write(f"(Statistics of the top-10 most similar retrieved images)\n")
            f.write(f"-" * 50 + "\n")
            for source, stats in similar_source_statistics.items():
                avg_similarity = stats['total_similarity'] / stats['count'] if stats['count'] > 0 else 0
                f.write(f"{source}:\n")
                f.write(f"  Count: {stats['count']} images\n")
                f.write(f"  Average Similarity: {avg_similarity:.3f}\n\n")
            
            # 详细结果记录
            f.write(f"Detailed Results:\n")
            f.write(f"-" * 50 + "\n")
            for i, result in enumerate(results_by_source, 1):
                f.write(f"{i}. Image: {result['image_name']}\n")
                f.write(f"   True Label: {result['true_label']}\n")
                f.write(f"   Predicted: {result['predicted_label']}\n")
                f.write(f"   Result: {'Correct' if result['is_correct'] else 'Wrong'}\n")
                f.write(f"   Similar Examples Used: {result['similar_examples']}\n")
                f.write(f"   Similar Sources: {result['similar_sources']}\n\n")
        
        # 保存JSON格式的详细结果
        json_results_file = Path(f"qwen_test_withprompt_results_{timestamp}.json")
        with open(json_results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': {
                    'total_processed': total_predictions,
                    'correct_predictions': correct_predictions,
                    'accuracy': accuracy,
                    'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                },
                'similar_source_statistics': dict(similar_source_statistics),
                'detailed_results': results_by_source
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        print(f"JSON results saved to: {json_results_file}")
    else:
        print("No successful predictions made.")

if __name__ == "__main__":
    print("Starting Qwen Image Classification Test with Prompts and Statistics...")
    # Check if test directories exist
    if not os.path.exists("real_test") or not os.path.exists("fake_test"):
        print("Error: Please ensure 'real_test' and 'fake_test' directories exist in the current directory.")
    elif not API_KEY or API_KEY == "YOUR_QWEN_API_KEY_HERE":
        print("Error: Please set your Qwen API key in the API_KEY variable.")
    else:
        main()
