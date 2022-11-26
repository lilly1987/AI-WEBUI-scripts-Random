더이상 유지보수 하지 않음. No more maintenance.  

AI-WEBUI-scripts-Random -> https://github.com/lilly1987/sd-random-extensions/

# AI-WEBUI-scripts-Random

더이상 유지보수 하지 않음. No more maintenance.  
랜덤 다이나믹은 https://github.com/lilly1987/sd-dynamic-prompting 로 분리  
마음대로 개조 및 배포 가능  


# 설치법

![2022-11-15 19 39 35](https://user-images.githubusercontent.com/20321215/201926877-6279b138-a1a8-49e1-8db1-31121e872cd4.png)


# Random loop

그리드 없이 단순 횟수 반복

![2022-10-22 23 57 47](https://user-images.githubusercontent.com/20321215/197346617-0ed1cd09-0ddd-48ad-8161-bc1540d628ad.png)  

![2022-10-23 00 10 10](https://user-images.githubusercontent.com/20321215/197346739-84835f11-3eea-4df5-b091-a57d4b0c0b51.png)  


## 항목

![2022-11-15 22 08 01](https://user-images.githubusercontent.com/20321215/201927356-73cefcd3-882c-4ab8-a410-d35dd6e063c9.png)

- loop = 반복 횟수
- step
- CFG = int 형으로 처리. 즉 소수점 처리 안함.(누가 개조해줘)
- width = width1 ~ width2 사이의 랜덧값. 64의 배수로 자동 처리
- height = height1 ~ height2
- fix width height direction = 특정 방향으로 회전. 예를들어 가로보다 세로가 길 경우 가로방향으로 회전.
- Sampling Random = 샘플링 선택한것중 랜덤


## 참조

https://gist.github.com/camenduru/9ec5f8141db9902e375967e93250860f  
https://github.com/adieyal/sd-dynamic-prompting  


# Random grid

더이상 유지보수 하지 않음. No more maintenance.  

xy grid 값 임의로 넣는게 너무 귀찬아서만든것. 
xy_grid.py를 대충 개조함  

![2022-10-23 00 09 20](https://user-images.githubusercontent.com/20321215/197346726-f93b7e84-f808-4167-9969-dc42763eeff1.png)  

![2022-10-22 22 22 00](https://user-images.githubusercontent.com/20321215/197341554-306e9384-9d1d-45c0-833f-55c03edec5fc.png)  

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


## 참조

https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/scripts/xy_grid.py  






