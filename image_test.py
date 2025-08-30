import os
import random
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# 导入主脚本中的相关函数
from qwen_test_with_prom_and_statitics import (
    load_precomputed_vectors,
    load_clip_model,
    extract_query_features,
    find_similar_images_for_prompt,
    classify_image_with_qwen_prompt,
    load_image_source_mapping
)

def main():
    # 配置参数
    vector_dir = "clip_vectors"
    base_dirs = {
        "0_real": "0_real",
        "1_fake": "1_fake",
        "real_test": "real_test",
        "fake_test": "fake_test"
    }
    data_test_dir = Path("data_test")
    if not data_test_dir.exists():
        print("Error: 'data_test' directory must exist in the current directory.")
        return

    # 加载预计算的向量
    print("Loading precomputed vectors...")
    search_datasets = load_precomputed_vectors(vector_dir)
    if not search_datasets:
        print("No precomputed vectors found. Please run save_vector_2_numpy.py first.")
        return

    # 加载图片来源映射
    print("Loading image source mapping...")
    image_source_mapping = load_image_source_mapping()

    # 获取data_test目录下所有图片
    test_images = list(data_test_dir.glob("*.jpg")) + \
                  list(data_test_dir.glob("*.jpeg")) + \
                  list(data_test_dir.glob("*.png"))
    print(f"Found {len(test_images)} images in 'data_test'.")

    # 随机打乱
    random.shuffle(test_images)

    results = []
    similar_source_statistics = defaultdict(lambda: {'count': 0, 'total_similarity': 0.0})

    for i, image_path in enumerate(test_images, 1):
        print(f"Processing image {i}/{len(test_images)}: {image_path.name}")
        similar_examples = find_similar_images_for_prompt(
            image_path, search_datasets, base_dirs, image_source_mapping, top_k=10
        )
        similar_sources_for_this_query = defaultdict(int)
        for j, (dataset_name, filename, similarity, label, full_path, source) in enumerate(similar_examples, 1):
            similar_source_statistics[source]['count'] += 1
            similar_source_statistics[source]['total_similarity'] += similarity
            similar_sources_for_this_query[source] += 1

        examples_for_classification = [(item[0], item[1], item[2], item[3], item[4]) for item in similar_examples]
        prediction = classify_image_with_qwen_prompt(image_path, examples_for_classification)
        results.append({
            'image_name': image_path.name,
            'predicted_label': prediction,
            'similar_examples': len(similar_examples),
            'similar_sources': dict(similar_sources_for_this_query)
        })
        print(f"  Predicted: {prediction}")

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(f"image_test_results_{timestamp}.txt")
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write(f"Image Test Results (data_test)\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total images processed: {len(results)}\n\n")
        f.write(f"Similar Images Source Statistics:\n")
        for source, stats in similar_source_statistics.items():
            avg_similarity = stats['total_similarity'] / stats['count'] if stats['count'] > 0 else 0
            f.write(f"{source}: {stats['count']} images (avg similarity: {avg_similarity:.3f})\n")
        f.write("\nDetailed Results:\n")
        for i, result in enumerate(results, 1):
            f.write(f"{i}. Image: {result['image_name']}\n")
            f.write(f"   Predicted: {result['predicted_label']}\n")
            f.write(f"   Similar Examples Used: {result['similar_examples']}\n")
            f.write(f"   Similar Sources: {result['similar_sources']}\n\n")

    json_results_file = Path(f"image_test_results_{timestamp}.json")
    with open(json_results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total_processed': len(results),
                'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'similar_source_statistics': dict(similar_source_statistics),
            'detailed_results': results
        }, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {results_file}")
    print(f"JSON results saved to: {json_results_file}")

if __name__ == "__main__":
    # print("Starting image test for data_test directory...")
    main()
