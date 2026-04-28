import os
import pickle
from datetime import datetime, timedelta
import numpy as np
import scipy.io
from torchvision import transforms
from tqdm import tqdm

import conf.config as config

OUTPUT_PKL = config.PICKLE_FILE_STR

def transfer_age(dob, photo_taken):
    """
    IMDB-WIKI 年龄计算方法：
    dob: MATLAB datenum (公元1年1月1日对应的 matlab datenum = 366)
    photo_taken: 拍摄年份（整数）
    返回: 整数年龄
    """
    days_since_1ad = dob - 366
    # 出生年份（考虑闰年）
    birth_year = np.floor(days_since_1ad / 365.25).astype(int)
    age = photo_taken - birth_year
    return age

def main():
    # === 第一步: 加载mat文件 ===
    print("=== 第一步: 加载mat文件 ===")
    mat = scipy.io.loadmat(config.MAT_PATH)
    print("Top-level keys:", list(mat.keys()))
    
    if 'wiki' in mat:
        main = mat['wiki'][0, 0]
        print("'wiki.mat' detected.")
    else:
        raise KeyError("No 'wiki' in .mat")
    
    # WIKI数据集标签。更换数据集需要修改此处。
    dob = main['dob'][0]
    photo_taken = main['photo_taken'][0]
    full_path = main['full_path'][0]
    gender = main['gender'][0]
    face_score = main['face_score'][0]
    
    total = len(dob)
    print(f"\nTotal raw samples: {total}\n")
    
    # === 第二步: 转换年龄 ===
    print("=== 第二步: 转换年龄 ===")
    age_int = transfer_age(dob, photo_taken)
    
    print(f"Age range in dataset (computed): {age_int.min()} ~ {age_int.max()}")
    if config.MIN_AGE is not None:
        age_valid = (age_int >= config.MIN_AGE) & (age_int <= config.MAX_AGE)
        print(f"Number of samples with age in [{config.MIN_AGE}, {config.MAX_AGE}]: {age_valid.sum()} / {total}")
        if age_valid.sum() > 0:
            ages_in_range = age_int[age_valid]
            print(f"  - Min in range: {ages_in_range.min()}, Max: {ages_in_range.max()}, Mean: {ages_in_range.mean():.2f}")
    else:
        age_valid = np.ones(total, dtype=bool)
    
    # === 第三步: 统计性别 ===
    print("\n=== 第三步: 统计性别 ===")
    gender_valid = np.isfinite(gender) & (gender >= 0) & (gender <= 1)
    print(f"Valid gender samples: {gender_valid.sum()} / {total}")
    if gender_valid.sum() > 0:
        male_count = np.sum(gender[gender_valid] == 1)
        female_count = np.sum(gender[gender_valid] == 0)
        print(f"  男性: {male_count} ({male_count/gender_valid.sum()*100:.1f}%)")
        print(f"  女性: {female_count} ({female_count/gender_valid.sum()*100:.1f}%)")
    
    # === 第四步: 统计人脸分数 ===
    print("\n=== 第四步: 统计人脸分数 ===")
    valid_face = np.isfinite(face_score)
    print(f"Finite face_score samples: {valid_face.sum()} / {total}")
    if valid_face.sum() > 0:
        face_vals = face_score[valid_face]
        print(f"  Min: {face_vals.min():.4f}, Max: {face_vals.max():.4f}, Mean: {face_vals.mean():.4f}")
        if config.MIN_FACE_SCORE is not None:
            above_thresh = (face_vals > config.MIN_FACE_SCORE).sum()
            print(f"  Face score > {config.MIN_FACE_SCORE}: {above_thresh} / {valid_face.sum()} ({above_thresh/valid_face.sum()*100:.1f}%)")
    
    # === 第五步: 检查图像 ===
    print("\n=== 第五步: 检查图像 ===")
    image_exists = []
    missing_paths_examples = []
    for idx in tqdm(range(total), desc="Checking images"):
        rel_path = full_path[idx][0]
        full_img_path = os.path.join(config.IMG_DIR_STR, rel_path)
        exists = os.path.exists(full_img_path)
        image_exists.append(exists)
        if not exists and len(missing_paths_examples) < 5:
            missing_paths_examples.append(full_img_path)
    image_exists = np.array(image_exists)
    print(f"Images existing: {image_exists.sum()} / {total}")
    if image_exists.sum() < total:
        print("Examples of missing paths:")
        for p in missing_paths_examples:
            print(f"  {p}")
    
    # === 第六步: 清洗 ===
    print("\n=== 第六步: 清洗 ===")
    final_mask = np.ones(total, dtype=bool)
    final_mask &= age_valid
    final_mask &= gender_valid
    if config.MIN_FACE_SCORE is not None:
        final_mask &= valid_face & (face_score > config.MIN_FACE_SCORE)
    final_mask &= image_exists
    
    final_count = final_mask.sum()
    print(f"Final valid samples after all filters: {final_count} / {total}\n")
    
    # 详细过滤损失
    print("Filtering breakdown (note: overlaps may cause double count):")
    print(f"  - Age out of [{config.MIN_AGE}, {config.MAX_AGE}]          : {(~age_valid).sum()}")
    print(f"  - Invalid gender (NaN or not 0/1)            : {(~gender_valid).sum()}")
    if config.MIN_FACE_SCORE is not None:
        face_filter = ~(valid_face & (face_score > config.MIN_FACE_SCORE))
        print(f"  - Face score <= {config.MIN_FACE_SCORE} or non-finite: {face_filter.sum()}")
    print(f"  - Image file missing                         : {(~image_exists).sum()}")
    
    if final_count == 0:
        print("No samples survived. Exiting without saving.")
        # 打印一个样本示例帮助调试
        print("\nSample raw data (first index):")
        print(f"  dob: {dob[0]}, photo_taken: {photo_taken[0]}, age computed: {age_int[0]}")
        print(f"  gender: {gender[0]}, face_score: {face_score[0]}")
        print(f"  full_path: {full_path[0][0]}")
        return
    
    # === 第七步: 构建样本列表 ===
    print("\n=== 第七步: 构建样本列表 ===")
    samples = []
    for idx in tqdm(range(total), desc="Building samples"):
        if not final_mask[idx]:
            continue
        rel_path = full_path[idx][0] 
        full_img_path = os.path.join(config.IMG_DIR_STR, rel_path) # 把IMG_DIR_STR和从 mat 中读取的相对路径进行拼接
        samples.append({
            'image_path': full_img_path,
            'age': int(age_int[idx]),
            'gender': int(gender[idx])
        })
    
    print(f"Successfully built {len(samples)} samples.")
    print("Example sample:", samples[0])
    
    # === 第八步: 保存PKL文件 ===
    print("\n=== 第八步: 保存PKL文件 ===")
    answer = input(f"Do you want to save the PKL file to {OUTPUT_PKL}? (y/n): ")
    if answer.lower() == 'y':
        with open(OUTPUT_PKL, 'wb') as f:
            pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved to {OUTPUT_PKL}")
    else:
        print("PKL not saved.")

if __name__ == "__main__":
    main()