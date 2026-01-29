# 교통량 예측 (Deep Learning)

서울시 교통량·기상 데이터를 활용한 **시계열 교통량 예측** 딥러닝 프로젝트입니다. LSTM/GRU/XGBoost 등으로 롤링 백테스트 및 모델 선택을 지원합니다.

## 주요 기능

- **전처리**: `build_dataset`, `make_windows` 로 시계열 윈도우 생성
- **모델**: LSTM, GRU, XGBoost 등 (TensorFlow/Keras, scikit-learn)
- **평가**: 롤링 백테스트(`rolling_backtest`), MAE/RMSE/MAPE
- **데이터 수집**: `src/collect/seoul_open_data.py` 로 서울 열린데이터 API 연동 (선택)

## 환경 요구사항

- Python 3.10+
- GPU 사용 시: CUDA 지원 TensorFlow 설치 권장

## 설치 및 실행

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 학습·평가 (CLI)

```bash
# 데이터 전처리·윈도우 생성 후 모델 학습
python -m src.train_models --help

# 롤링 백테스트
python -m src.rolling_backtest --help

# 최적 모델 선택
python -m src.select_model --help
```

### Streamlit 앱 (app.py)

```bash
streamlit run app.py
```

(앱이 있는 경우) 브라우저에서 `http://localhost:8501` 접속

## 프로젝트 구조

```
traffic-volume(DL)/
├── app.py                 # Streamlit 앱 (교통량 예측·시각화)
├── requirements.txt
├── data/
│   ├── raw/               # 원본 CSV (교통량·기상 등)
│   └── processed/         # 전처리 결과
├── models/                # .keras, .joblib 등 저장 모델
├── reports/               # 백테스트 결과, 메트릭 CSV
├── traffic_volume.ipynb   # 실험 노트북
└── src/
    ├── preprocess/        # build_dataset, make_windows
    ├── collect/           # seoul_open_data (API)
    ├── utils/             # io, metrics
    ├── train_models.py
    ├── eval_rolling.py
    ├── rolling_backtest.py
    └── select_model.py
```

## 데이터

- `data/raw/` 에 교통량·기상 CSV 배치
- 서울 열린데이터 API 사용 시 `.env` 에 API 키 설정

## 라이선스

프로젝트용·학습용으로 자유롭게 사용 가능합니다.
