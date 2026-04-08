import os
import shutil
# 图片分类
data_path = "data"
test = "data/test"
train = "data/train"
true_categories = ['African people and villages', 'Beach', 'Historical buildings',
              'Buses', 'Dinosaurs', 'Elephants', 'Flowers', 'Horses',
              'Mountains and glaciers', 'Food', 'Dogs', 'Lizards', 'Fashion',
              'Sunsets', 'Cars', 'Waterfalls', 'Antiques', 'Battle ships',
              'Skiing', 'Desserts']
categories = [str(i) for i in range(0, 20)]
def processing():
    n = 0
    # 按数字排序
    files = sorted(os.listdir(data_path), key=lambda x:int(x[:-4]))
    for category in categories:
        # 创建目录
        test_category_path = os.path.join(test, category)
        train_category_path = os.path.join(train, category)
        os.makedirs(test_category_path, exist_ok=True)
        os.makedirs(train_category_path, exist_ok=True)

        # 移动文件
        train_files = files[n:n+50]
        test_files = files[n+50:n+100]
        for file in test_files:
            shutil.move(os.path.join(data_path, file), os.path.join(test_category_path, file))
        for file in train_files:
            shutil.move(os.path.join(data_path, file), os.path.join(train_category_path, file))

        n += 100



if __name__ == "__main__":
    processing()