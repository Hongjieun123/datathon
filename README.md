# H.D.A.I 2021 
## 주제 2. 심전도 데이터셋을 이용한 부정맥 진단 AI 모델
### Development status
	GPU Quadro RTX 6000 (2개)
	tensorflow 2.3.1
	python 3.6.9
### Dependencies
	Requires the following libraries:
	1. numpy
	2. pandas 
	3. matplotlib
	4. sklearn
	5. tensorflow
	6. xmldict
	7. joblib
	8. base64
	9. os
	
	
	
### Usage
1. 전체 code를 아래 순서대로 실행
2. 참가자가 전달한 폴더 경로를 raw_path 변수로 설정
3. test dataset이 저장된 폴더 경로를 base_dir 변수로 설정
4. 최종 ROC, AUC 확인

	4_Evaluation.ipynb 이용하여 평가 
	※ Test set 폴더 구성이 아래와 같이 생긴 것을 가정함. 
	
	./data/test
	
	./data/test/arrhythmia
	
	./data/test/normal
	
	
### Results on our model (AUC & ROC Curve)
Our model AUC = 0.9959
![savefig_default (1)](https://user-images.githubusercontent.com/62556038/145504513-97a8dbaa-127b-4f64-99c4-3d5b39a82ecd.png)
