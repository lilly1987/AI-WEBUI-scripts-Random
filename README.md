# AI-WEBUI-scripts-Random

# 설치법

Random.py 를 받아서 WEBUI\stable-diffusion-webui\scripts 에 넣으면 된다

![2022-10-22 22 24 22](https://user-images.githubusercontent.com/20321215/197341519-b2537b68-99e3-4acb-8363-738787bb596d.png)  
![2022-10-22 22 24 37](https://user-images.githubusercontent.com/20321215/197341513-6b0c09f0-636d-4567-ac1f-f5eb1160af58.png)  
# 사용법

기본 로직은 x/y plot과 같음  
단지 내부적으로 x종류를 step로 고정,y종류를 cfg로 고정 시킴  
첫번째 스샷의 경우와 같이 설정할 경우  
step1|2 값의 범위(10-30) 안에서 step cnt 갯수(10)만큼 x값 생성  
cfg1|2 값의 범위(6-15) 안에서 cfg cnt 갯수(10)만큼 x값 생성  
1|2의 범위갑을 거꾸로 넣어도 알아서 바궈 넣도록 처리함  
cfg의 값의 경우 int형으로 처리해서 소수점값을 읽지 않음.(귀찬으니 전문가가 있으면 알아서 수정해줘)  

두번째 스샷처럼 x,y값이 랜덤으로 생성됨  
x:[27, 25, 15, 28, 21, 13, 30, 21, 13, 24]  
y:[13, 14, 7, 14, 10, 7, 10, 10, 15, 9]  

![2022-10-22 22 19 48](https://user-images.githubusercontent.com/20321215/197341552-ecfe787a-4643-4a6f-a32f-593de23b96be.png)  
![2022-10-22 22 22 00](https://user-images.githubusercontent.com/20321215/197341554-306e9384-9d1d-45c0-833f-55c03edec5fc.png)  
