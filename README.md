# Fashion_MNIST
> 목표 : CNN을 이용한 fashion_mnist 최적의 모델 생성 및 학습//  
기존의 MNIST를 대체할만한 fashion_mnist를 고안해둔 프로젝트 < https://github.com/zalandoresearch/fashion-mnist
에서 fashion MNIST의 데이터셋을 사용해 간단한 CNN모델로 어떤 적용을 하면 어떻게 변하는지 다양한 시도로 인한 loss 및 accuracy 측정

## 실행 방법
```
python3 FashionMNIST_CNN.py
```

## 개발 내용 (시도시점순으로 나열)

* task0//                                           
최초의 기존 CNN conv + max pool 한층 그리고 fc층 두개로 첫 fc는 128 relu를 사용, 두번째 fc는 10으로 softmax를 사용 
optimizer로는 Adam Optimizer사용 (91.73%)
    
* task3//                                           
task0에서 시도횟수가 많아져 overfitting 되는것 같아 epoch를 10으로 변환 (90.64%)
    
* task4//                                           
conv층이 한층 더 늘어나면 어떻게 될까 싶어서 task0에서 conv와 maxpool을 똑같은 구조로 한층 더쌓음 (91.13%)
    
* task5//                                           
그리고 conv층이 두층인 상태, task4에서 epoch를 10으로 줄임 (91.21%)
    
* task6//                                           
변함이 없길래, conv 한층의 커널사이즈를, task4에서 5*5로 조정 (91.66%)
    
* task7//                                           
이번에는 conv 두층의 커널사이즈를 task5에서 5*5로 조정\, epoch 15로 변경 (91.32%)
    
여기서부터 epoch를 15로 고정

* task8//                                           
batch normalization을 적용시키면 어떨까 싶어,
task7에서 각 conv층 사이에 BN층 하나씩 삽입. 편차가 너무 커 batch-size를 128에서 64로 수정 (90.19%)

* task9//                                           
그리고 dropout도 적용해보고 싶어, fc층 사이에 Dropout(0.25) 삽입 (91.1%)
    
* task10//                                           
여전히 변함이 없길래 지금까지 적용한것들을 그대로, task9에서 conv층 하나로 줄임 (90.58%)
    
* task11//                                           
이번에는 optimizer가 문제인가 싶어, task optimizer를 sgd로 바꿔봄 (91.29%)
    
* task12//                                           
sgd가 다시 loss function이 괜찮길래 conv층을, task11에서 다시 두층으로 (91.2%)
    
* task13//                                           
영향이 없길래, 다시 한층으로 줄이고 optimizer를 Adagrad로 바꿈 (91.93%)
    
* task14//                                           
Adadelta optimizer도 써봄 (91.06%)
    
* task15//                                           
제일 근접했던 sgd에서 learning rate와 momentum그리고 nesterov를 적용시킴 (90.73%)
    
* task1//                                           
task0에서 relu dense를 256으로 올림, dropout(0.5)적용 (92.07%)
    
* task2//                                                   
task0에서 모델을 vggnet로 변형 (93.42%)
    
## 결과

   task2의 vggnet optimizer로 최대 93.42%의 accuracy를 얻을 수 있었음                               
   pytorch로 구현한 다른 코드에서 < 최대 93%의 accuracy를 얻었기에 그 구조를 따라가며 적용했으나                               
   전혀 영향이 없었고, 오히려 기존 코드에서 요소들을 하나씩만 바꿔볼걸 괜히 계속 겹치며 적용을 했나 싶음                               
   모델에 이런저런 변형을 시도하더라도, loss와 accuracy가 눈에 확 띄게 결과가 나오지 않아서 어떤식으로 개량해야할지 매우 난감했음                
   결국 기존의 모델 여러가지를 시도해보던중 mini vggnet이 가장 좋은 accuracy가 나오게 되어서 가장 좋은 accuracy로 멈추게 됨                    
   
## Reference

   https://www.pyimagesearch.com/2019/02/11/fashion-mnist-with-keras-and-deep-learning/
   https://github.com/zalandoresearch/fashion-mnist
   
## 개발자

강신혁 - https://github.com/dtc02040

김애은 - https://github.com/aeeunkim

안유진 - https://github.com/ahnyujin

유안지 - https://github.com/yooanji

이수연
