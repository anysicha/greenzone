# GreenCity AI - 사용자용 MVP

자연어 입력 → Gemini API 파라미터 추출 → Hard Constraint Filter → PuLP 최적화 Solver → 사용자용 리포트 화면으로 이어지는 Streamlit MVP입니다.

## 실행

```bash
cd greencity_ai_gemini_mvp_clean
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
export GEMINI_API_KEY="새_API_KEY"
python -m streamlit run app.py
```

## 사용자 입력 예시

```text
예산은 5천만원이고 면적은 200㎡인 관리 쉬운 옥상 정원을 만들고 싶어. 토심은 30cm, 한 수종은 30% 이하로 다양하게 구성해줘.
```
