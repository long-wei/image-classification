import os
os.environ["OMP_NUM_THREADS"] = "1"
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.metrics import accuracy_score


class Img_classify():
    def __init__(self, split="test"):
        torch.cuda.set_device(0)  # 初始化GPU设备:ml-citation{ref="4" data="citationList"}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.features = []
        self.labels = []
        self.img_name = []
        self.split = split
        self.kmeans = None


    # SIFT(Bow)特征
    def __BOW_SIFT(self, image_paths):
        sift = cv2.SIFT_create(contrastThreshold=0.03, edgeThreshold=10)
        # 训练阶段
        bow_trainer = cv2.BOWKMeansTrainer(500)
        for path in os.listdir(image_paths):
            class_path = os.path.join(image_paths, path)
            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path,img_file)
                img = cv2.imread(img_path, 0)
                _, des = sift.detectAndCompute(img, None)
                if des is not None:
                    bow_trainer.add(des)
        vocabulary = bow_trainer.cluster()
        kmeans = KMeans(n_clusters=500)
        kmeans.fit(vocabulary)
        self.kmeans = kmeans


    # 2. 遍历目录提取特征
    def extract_features(self, root_dir):
        features = []
        LBP_features = []
        labels = []
        if self.split == "train":
            self.__BOW_SIFT(root_dir)

        for class_dir in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_dir)
            for img_file in os.listdir(class_path):
                self.img_name.append(img_file)

                img_path = os.path.join(class_path, img_file)
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

                # cnn特征
                model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(self.device)
                model.eval()
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                img_tensor = transform(img).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    cnn_feat = model(img_tensor).cpu().numpy().flatten()


                # GLCM灰度共生矩阵特征
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                glcm = graycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)
                glcm_feat = np.hstack([
                                graycoprops(glcm, 'contrast').ravel(),
                                graycoprops(glcm, 'energy').ravel(),
                                graycoprops(glcm, 'homogeneity').ravel()
                                ])


                # SIFT(Bow)特征
                sift = cv2.SIFT_create()
                _, des = sift.detectAndCompute(img, None)
                sift_hist = np.zeros(self.kmeans.n_clusters)
                if des is not None:
                    lab = self.kmeans.predict(des)
                    for i in lab:
                        sift_hist[i] += 1


                # LBP纹理特征
                lbp_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                lbp = local_binary_pattern(lbp_img, P=8, R=1, method='uniform')
                hist_lbp, bin_edges = np.histogram(lbp, bins=256)
                LBP_features.append(hist_lbp)

                # 颜色特征
                hist_r = cv2.calcHist([img], [0], None, [256], [0, 256])     # 计算各通道直方图
                hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
                hist_b = cv2.calcHist([img], [2], None, [256], [0, 256])
                hist_r = cv2.normalize(hist_r, None).flatten()   # 直方图归一化
                hist_g = cv2.normalize(hist_g, None).flatten()
                hist_b = cv2.normalize(hist_b, None).flatten()
                color_feat = np.concatenate([hist_r, hist_g, hist_b])


                labels.append(int(class_dir))
                # # cnn_feat  glcm_feat  sift_hist   hist_lbp  color_feat

                # features.append(cnn_feat)  # 0.83/0.81
                # features.append(glcm_feat)  # 0.37/0.40
                # features.append(sift_hist)  # 0.34/0.33
                # features.append(hist_lbp)  # 0.40/0.40
                # features.append(color_feat)  # 0.49/0.51
                # features.append(np.concatenate([cnn_feat, glcm_feat]))    # 0.83/0.81
                # features.append(np.concatenate([cnn_feat, sift_hist]))    # 0.83/0.83
                features.append(np.concatenate([cnn_feat, color_feat]))    # 0.84/0.82
                # features.append(np.concatenate([cnn_feat, hist_lbp]))       # 0.83/0.82
                # features.append(np.concatenate([glcm_feat, hist_lbp]))      # 0.46/0.52
                # features.append(np.concatenate([cnn_feat, glcm_feat, color_feat]))    # 0.84/0.82
                # features.append(np.concatenate([cnn_feat, glcm_feat, hist_lbp, color_feat]))    # 0.84/0.82
                # features.append(np.concatenate([cnn_feat, glcm_feat, hist_lbp, sift_hist]))    # 0.82/0.83
                # features.append(np.concatenate([cnn_feat, glcm_feat, color_feat, sift_hist]))    # 0.83/0.83
                # features.append(np.concatenate([cnn_feat, glcm_feat, sift_hist, hist_lbp, color_feat]))   # 0.83/0.84

        self.labels = np.reshape(np.array(labels), (-1, 1))
        self.features = np.array(features)



class Train_classify():
    svm_model_rbf = SVC(kernel='rbf')
    svm_model_linear = SVC(kernel='linear')

    def __init__(self, train_features, train_labels, test_features, test_labels, img_name):
        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.test_labels = test_labels
        self.img_name = img_name

    # SVM分类
    def SVM_train(self, model):
        model.fit(self.train_features, self.train_labels)

    def SVM_fit(self, model):
        # 预测测试集
        y_pred = model.predict(self.test_features)

        # 计算准确率
        accuracy = accuracy_score(self.test_labels, y_pred)
        print(f"分类准确率: {accuracy:.2f}")
        return y_pred


    def result_text(self, predictions):
        with open('classification_results.txt', 'w') as f:
            for i in range(1000):
                f.write(f"{self.img_name[i][:-4]} {int(self.test_labels[i].item())} {predictions[i]}\n")

    def Confusion_matrix(self, pre, name):
        arr = np.zeros((20, 20))
        for i in range(1000):
            r = int(self.test_labels[i].item())
            c = pre[i]
            arr[r][c] += 1
        normalized_arr = arr / 50
        import seaborn as sns
        plt.figure(figsize=(12, 10))
        sns.heatmap(normalized_arr, annot=True, fmt='.2f', cmap='Blues')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'{name} Confusion Matrix')
        plt.show()

    def main(self):
        # rbf
        self.SVM_train(self.svm_model_rbf)
        predictions_1 = self.SVM_fit(self.svm_model_rbf)
        self.result_text(predictions_1)
        self.Confusion_matrix(predictions_1, "Rbf")
        # linear
        self.SVM_train(self.svm_model_linear)
        predictions_2 = self.SVM_fit(self.svm_model_linear)
        self.result_text(predictions_2)
        self.Confusion_matrix(predictions_2, "Linear")


import DP
# 对解压的图片进行分类
DP.processing()
train, test = DP.train, DP.test

train_data = Img_classify("train")
train_data.extract_features(train)

test_data = Img_classify("test")
test_data.kmeans = train_data.kmeans
test_data.extract_features(test)

scaler = StandardScaler()
x_train, x_test = scaler.fit_transform(train_data.features), scaler.transform(test_data.features)
y_train, y_test = train_data.labels, test_data.labels

model = Train_classify(x_train, y_train, x_test, y_test, test_data.img_name)
model.main()
