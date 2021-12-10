# H.D.A.I 2021 
## 주제 2. 심전도 데이터셋을 이용한 부정맥 진단 AI 모델
### Development status
	GPU : Quadro RTX 6000 (2개)
	tensorflow 2.3.1
	python 3.6.9
### Dependencies
	Requires the following libraries:
	1. numpy
	2. pandas 
	3. matplotlib
	4. sklearn
	5. tensorflow
	
### Usage
	1. Data_load.ipynb (Data load 및 padding)
	2. model.py 
	3. Data_Normalization & Training.ipynb 
	(1. 에서 저장한 데이터를 불러와 정규화, 2.의 model import 하여 학습 및 model weight 저장)
	4. Prediction.ipynb
	(1. 에서 저장한 validation 데이터와 3.에서 학습한 model을 load하여 , model의 성능 평가)
	5. model file (1. best_model.h5 (model+weight file) 2. best_weights.h5 (weights values) 3. best_model.csv (model accuracy, loss values))
	
### Results on our model (AUC & ROC Curve)
Our model AUC = 0.9960
![savefig_default (1)](https://user-images.githubusercontent.com/62556038/145504513-97a8dbaa-127b-4f64-99c4-3d5b39a82ecd.png)
