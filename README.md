# AI 모델 개발 프로젝트

## 프로젝트 개요
LLM/챗봇 개발과 예측 모델을 결합한 통합 AI 시스템 개발 프로젝트입니다.

## 프로젝트 구조

```
AI_MODEL/
├── data/
│   ├── raw/           # 원본 데이터
│   ├── processed/     # 전처리된 데이터
│   └── external/      # 외부 데이터 소스
├── notebooks/         # Jupyter 실험 노트북
├── src/
│   ├── llm/           # LLM/챗봇 관련 소스코드
│   ├── prediction/    # 예측 모델 관련 소스코드
│   ├── utils/         # 공통 유틸리티
│   └── api/           # API 서버 코드
├── models/
│   ├── llm/           # 저장된 LLM 모델
│   └── prediction/    # 저장된 예측 모델
├── configs/           # 설정 파일 (YAML, JSON)
├── tests/             # 단위 테스트 및 통합 테스트
├── docs/              # 프로젝트 문서
├── scripts/           # 실행 스크립트
└── logs/              # 로그 파일
```

## 기술 스택
- **LLM/챗봇**: LangChain, HuggingFace Transformers, OpenAI API
- **예측 모델**: scikit-learn, XGBoost, PyTorch
- **데이터 처리**: pandas, numpy
- **API**: FastAPI
- **실험 추적**: MLflow, Weights & Biases

## 시작하기

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

## 문의
- 담당자: 박창선
- 이메일: chaosen0521@gmail.com
