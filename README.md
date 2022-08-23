# Goorm NLP Project 3 - Neural Machine Translation
Goorm 자연어 처리 전문가 양성과정 4기 세 번째 프로젝트
1. 기간 : 2022.08.04~2022.08.18
2. 주제 : 신조어 번역이 가능한 한영, 영한 번역 모델 개발
3. 목표
    1) 외국인들에게 한국어 신조어를 이해할 수 있도록 하는 번역 모델 개발
    2) Monolingual corpus의 효과적인 사용
4. 성과
    1) Tagged Back Translation을 통한 신조어 / 표준어 제어, Monolingual corpus 활용
    2) 배포 환경에 따른 모델 경량화 (Quantization)
5. 환경 : Google Colab Pro+, Goorm IDE
6. 주요 라이브러리 : transformers, datasets, pandas, torch, re, flask, gunicorn, bootstrap
7. 구성원 및 역할
    * 박정민 (팀장)
        * 팀 프로젝트 관리 감독 및 총괄
        * flask와 gunicorn을 활용한 모델 배포 환경 개발
        * 모델 학습 및 추론 코드 작성
        * 발표자료 작성
    * 이예인
        * 영어 신조어 데이터 크롤링
        * flask와g gunicorn을 활용한 모델 배포 환경 개발
        * 아이디어 및 최종 발표
        * 발표자료 작성
    * 이용호
        * 자료 조사
        * 개발 환경 탐색
    * 임영서
        * 신조어 데이터 탐색 및 분포 확인
        * 신조어를 활용한 학습 방법 제안
        * 발표자료 작성
    * 정재빈
        * 한국어 신조어 데이터 크롤링
        * Back Translation 기법 조사, 공유
        * 발표자료 작성 및 취합
8. 핵심 아이디어 및 기술
    * mBART모델의 Embedding layer pruning을 통한 모델 경량화
    * Tagged Back Translation을 통한 monolingual corpus 활용 및 신조어 / 표준어 제어
