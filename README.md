# MyProject1

|프로젝트명|SNS(인스타그램)을 이용한 상품 추천 서비스|
|:---|:---|
|목표|특정 사용자의 인스타그램 게시글 이미지를 크롤링하고, 크롤링한 이미지를 kakao vision api를 이용하여 쇼핑몰 내 상품 추천|
|사용자 시나리오|1. 사용자는 어플 실행 후 로그인을 한다. <br/> 2. 사용자는 인스타그램을 통해 상품 추천받기를 할 것인지, 사진 업로드를 통해 상품 추천 받기를 할 것인지 선택한다.<br/>3-1. 인스타그램을 선택할 경우 BeautifulSoup라이브러리와 Selenium 라이브러리, ChormeDrive를 이용하여 인스타그램 내 이미지를 가져와 웹 서버 내에 저장한다.<br/>3-2. 사진 업로드를 선택할 경우 사용자가 직접 업로드를 통해 웹 서버에 이미지를 저장한다. <br/>4. 수집된 이미지는 Kakao Vision API 중 하나인, 상품 탐색 API를 이용하여 이미지 내 패션 상품, 가방, 신 발 등의 상품들을 검출한 결과 사진을 웹 서버에 저장하여 사용자에게 보여준다. <br/>5. 사용자는 이 결과 사진 중 추천받고 싶은 상품을 선택하면, 웹 서버에서 Mahotas 오픈소스를 이용하여 가장 유사한 이미지 3개와 함께 해당 상품의 쇼핑몰 링크를 사용자에게 출력해준다.|

## 프로젝트 구조
![structure](https://github.com/T2us/MyProject1/blob/master/markdownImg/%EA%B5%AC%EC%A1%B0.PNG)

## 프로젝트 작동 예시
![exam](https://github.com/T2us/MyProject1/blob/master/markdownImg/%EC%9E%91%EB%8F%99%EC%98%88%EC%8B%9C.PNG)
