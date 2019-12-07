# Fashion_MNIST
> CNN을 이용한 Keras fashion_mnist 모델 생성 및 학습

## 실행 방법
```
python3 FashionMNIST_CNN.py
```

## 개발 내용

* task0
    * Adam Optimizer사용 (91.73%)
    
* task3
    * task0에서 epoch를 10으로 변환 (90.64%)
    
* task4
    * task0의 conv에 maxpool을 한층 더쌓음 (91.13%)
    
* task5
    * task4에서 epoch를 10으로 줄임 (91.21%)
    
* task6
    * task4에서 커널 사이즈 5*5로 조정 (91.66%)
    
* task7
    * conv2d층 늘린상태에서 커널사이즈 5로 변경 , epoch 15로 변경 (91.32%)
    
* task8
    * 각 conv층 사이에 BN층 하나씩 삽입. 편차가 너무 커 batch-size를 128에서 64로 수정 (90.19%)

* task9
    * fc층 사이에 Dropout(0.25) 삽입 (91.1%)
    
* task10
    * task9에서 conv층 하나로 줄임 (90.58%)
    
* task11
    * optimizer를 sgd로 바꿔봄 (91.29%)
    
* task12
    * task11에서 다시 두층으로 (91.2%)
    
* task13
    * 다시 한층으로 줄이고 optimizef를 Adagrad로 바꿈 (91.93%)
    
* task14
    * Adadelta optimizer도 써봄 (91.06%)
    
* task15
    * 제일 근접했던 sgd에서 learning rate와 momentum그리고 nesterov를 적용시킴 (90.73%)
    
* task1
    * task0에서 relu dense를 256으로 올림, dropout(0.5)적용 (92.07%)
    
* task2           
    * task0에서 optimizer를 vggnet로 변형 (93.42%)
    

## 개발자

강신혁

김애은 - https://github.com/aeeunkim

안유진 - https://github.com/ahnyujin

유안지 - https://github.com/yooanji

이수연
