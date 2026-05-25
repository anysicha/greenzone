"""
GreenCity AI - Dual Mode MVP

사용자 흐름 A: 자연어 직접 입력
- 사용자가 문장으로 조경 조건을 입력
- Gemini API 또는 로컬 백업 파서가 내부 파라미터로 변환
- Hard Constraint Filter + PuLP 최적화 Solver 실행
- 사용자용 리포트 제공

사용자 흐름 B: 버튼/슬라이더 선택
- 사용자가 폼 UI로 조건을 선택
- 동일한 최적화 엔진 실행
- 사용자용 리포트 제공
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from optimization_model import LandscapingOptimizationModel, ScenarioConstraints
from optimization_solver import LandscapingOptimizationSolver, OptimizationResult

try:
    from google import genai
    from google.genai import types
    GEMINI_SDK_AVAILABLE = True
except Exception:
    GEMINI_SDK_AVAILABLE = False


# =========================
# 기본 설정
# =========================
APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "master_table.xlsx"

DEFAULT_VALUES = {
    "target_co2": 0.0,
    "min_hardiness_zone": 4,
    "diversity_ratio": 0.30,
    "max_soil_depth_by_location": {
        "Rooftop": 30,
        "Wall": 25,
        "Park": 100,
    },
    "location_label": {
        "Rooftop": "옥상 정원",
        "Wall": "건물 외벽/수직정원",
        "Park": "공원·평지 조경",
    },
}

REQUIRED_FIELDS = ["location_type", "total_budget", "total_area"]


# =========================
# UI 스타일
# =========================
def inject_css() -> None:
    st.markdown(
        """
        <style>
        .main .block-container { padding-top: 2rem; max-width: 1180px; }
        .hero {
            padding: 28px 30px;
            border-radius: 24px;
            background: linear-gradient(135deg, #eefbea 0%, #e8f7ff 100%);
            border: 1px solid #d7efd4;
            margin-bottom: 22px;
        }
        .hero h1 { margin: 0; font-size: 2.2rem; }
        .hero p { margin: 10px 0 0 0; font-size: 1.05rem; color: #3f4b3f; }
        .soft-card {
            border: 1px solid #e6e6e6;
            border-radius: 18px;
            padding: 18px 18px;
            background-color: #ffffff;
            box-shadow: 0 2px 10px rgba(0,0,0,0.035);
            margin-bottom: 12px;
        }
        .small-muted { color: #6b7280; font-size: 0.92rem; }
        .pill {
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            background: #eef8ee;
            color: #256029;
            font-size: 0.88rem;
            margin-right: 6px;
            margin-bottom: 6px;
            border: 1px solid #d7ecd4;
        }
        .warning-box {
            padding: 16px 18px;
            border-radius: 16px;
            background: #fff8e6;
            border: 1px solid #ffe3a3;
            color: #5d4300;
        }
        div[data-testid="stMetricValue"] { font-size: 1.55rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================
# 유틸 함수
# =========================
def clean_json_text(text: str) -> str:
    """Gemini 응답에서 JSON만 안전하게 추출."""
    text = (text or "").strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    return text or "{}"


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def normalize_parsed_params(data: Dict[str, Any]) -> Dict[str, Any]:
    """LLM 출력값 또는 UI 입력값을 최적화 엔진에 넣기 좋은 형태로 보정."""
    data = dict(data or {})

    location_type = data.get("location_type")
    if isinstance(location_type, str):
        location_type = location_type.strip()
    if location_type not in ["Rooftop", "Wall", "Park", None]:
        raw = str(location_type)
        if any(k in raw for k in ["옥상", "루프", "roof", "Rooftop"]):
            location_type = "Rooftop"
        elif any(k in raw for k in ["외벽", "벽면", "수직", "wall", "Wall"]):
            location_type = "Wall"
        elif any(k in raw for k in ["공원", "평지", "마당", "park", "Park"]):
            location_type = "Park"
        else:
            location_type = None
    data["location_type"] = location_type

    data["location_label"] = data.get("location_label") or DEFAULT_VALUES["location_label"].get(location_type)
    data["total_budget"] = safe_float(data.get("total_budget"), None)
    data["total_area"] = safe_float(data.get("total_area"), None)
    data["target_co2"] = safe_float(data.get("target_co2"), DEFAULT_VALUES["target_co2"])
    data["min_hardiness_zone"] = safe_int(data.get("min_hardiness_zone"), DEFAULT_VALUES["min_hardiness_zone"])
    data["diversity_ratio"] = safe_float(data.get("diversity_ratio"), DEFAULT_VALUES["diversity_ratio"])

    if data["diversity_ratio"] is not None and data["diversity_ratio"] > 1:
        data["diversity_ratio"] = data["diversity_ratio"] / 100

    max_soil_depth = safe_float(data.get("max_soil_depth"), None)
    if max_soil_depth is None and location_type in DEFAULT_VALUES["max_soil_depth_by_location"]:
        max_soil_depth = DEFAULT_VALUES["max_soil_depth_by_location"][location_type]
    data["max_soil_depth"] = max_soil_depth

    data["maint_level_max"] = safe_int(data.get("maint_level_max"), None)

    preferences = data.get("preferences")
    if preferences is None:
        preferences = []
    if isinstance(preferences, str):
        preferences = [preferences]
    data["preferences"] = preferences

    missing = []
    for field in REQUIRED_FIELDS:
        if data.get(field) is None:
            missing.append(field)
    llm_missing = data.get("missing_fields") or []
    if isinstance(llm_missing, str):
        llm_missing = [llm_missing]
    data["missing_fields"] = sorted(set(llm_missing + missing))

    return data


def won_to_label(value: float) -> str:
    if value >= 100_000_000:
        return f"{value / 100_000_000:.1f}억 원".replace(".0", "")
    return f"{value / 10_000:.0f}만 원"


def location_icon(location_type: Optional[str]) -> str:
    return {"Rooftop": "🏢", "Wall": "🧱", "Park": "🌳"}.get(location_type, "🌿")


def maint_label_from_value(value: Optional[int]) -> str:
    if value == 2:
        return "쉬운 관리"
    if value == 3:
        return "보통 관리"
    if value == 5:
        return "전문 관리 가능"
    return "제한 없음"


# =========================
# Gemini 자연어 파싱
# =========================
def parse_with_gemini(user_text: str, api_key: str, model_name: str = "gemini-2.5-flash") -> Dict[str, Any]:
    """Gemini API를 사용해 자연어 요청을 최적화 파라미터 JSON으로 변환."""
    if not GEMINI_SDK_AVAILABLE:
        raise RuntimeError("google-genai 패키지가 설치되어 있지 않습니다. `pip install google-genai`를 실행하세요.")
    if not api_key:
        raise ValueError("Gemini API Key가 없습니다.")

    client = genai.Client(api_key=api_key)

    prompt = f"""
너는 도시녹화 최적화 서비스 'GreenCity AI'의 입력 파서다.
사용자의 한국어 자연어 요청을 읽고, 아래 JSON 스키마에 맞춰 최적화 엔진 입력값만 반환하라.
반드시 JSON 객체만 출력하고, 설명 문장/마크다운/코드블록은 출력하지 마라.

[반환 JSON 스키마]
{{
  "location_type": "Rooftop | Wall | Park | null",
  "location_label": "옥상 정원 | 건물 외벽/수직정원 | 공원 및 평지 조경 | null",
  "total_budget": number 또는 null,
  "total_area": number 또는 null,
  "target_co2": number,
  "max_soil_depth": number 또는 null,
  "min_hardiness_zone": number,
  "diversity_ratio": number,
  "maint_level_max": number 또는 null,
  "preferences": string 배열,
  "missing_fields": string 배열,
  "reasoning_summary": string
}}

[해석 규칙]
- 옥상, 루프탑, roof, rooftop → location_type = "Rooftop", 기본 max_soil_depth = 30
- 외벽, 벽면, 수직정원, wall → location_type = "Wall", 기본 max_soil_depth = 25
- 공원, 평지, 마당, 공터, park → location_type = "Park", 기본 max_soil_depth = 100
- 1평 = 3.3058㎡로 변환하라.
- 예산 단위 변환: 1억=100000000원, 5천만원=50000000원, 2천만원=20000000원, 100만원=1000000원
- 관리 쉬움, 저관리, 손이 덜 감, 관리 어렵지 않음 → maint_level_max = 2
- 관리 보통 → maint_level_max = 3
- 전문 관리 가능, 관리 많이 가능 → maint_level_max = 5
- 다양하게, 골고루, 한 종류만 몰리지 않게 → diversity_ratio = 0.3
- 다양성 조건이 숫자로 주어지면 반영하라. 예: 한 수종 40% 이하 → diversity_ratio = 0.4
- 탄소 목표가 없으면 target_co2 = 0
- 내한성 정보가 없으면 min_hardiness_zone = 4
- 필수 필드는 location_type, total_budget, total_area다.
- 필수 필드가 없거나 추론하기 어려우면 missing_fields에 해당 필드명을 넣어라.
- reasoning_summary에는 한 문장으로 어떤 조건을 어떻게 해석했는지 한국어로 써라.

[사용자 입력]
{user_text}
""".strip()

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )

    raw = response.text or "{}"
    data = json.loads(clean_json_text(raw))
    return normalize_parsed_params(data)


# =========================
# 로컬 백업 파서
# =========================
def parse_budget_fallback(text: str) -> Optional[float]:
    t = text.replace(",", "").replace(" ", "")
    m = re.search(r"(\d+(?:\.\d+)?)억", t)
    if m:
        return float(m.group(1)) * 100_000_000
    m = re.search(r"(\d+(?:\.\d+)?)천만원", t)
    if m:
        return float(m.group(1)) * 10_000_000
    m = re.search(r"(\d+(?:\.\d+)?)만원", t)
    if m:
        return float(m.group(1)) * 10_000
    m = re.search(r"(\d{6,})원", t)
    if m:
        return float(m.group(1))
    return None


def parse_area_fallback(text: str) -> Optional[float]:
    t = text.replace(",", "").replace(" ", "")
    m = re.search(r"(\d+(?:\.\d+)?)㎡", t)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)m2", t, flags=re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)평", t)
    if m:
        return float(m.group(1)) * 3.3058
    return None


def parse_fallback(user_text: str) -> Dict[str, Any]:
    """Gemini API가 안 될 때 앱이 완전히 죽지 않게 하는 백업 파서."""
    text = user_text.lower()

    location_type = None
    if any(k in user_text for k in ["옥상", "루프탑"]) or "roof" in text:
        location_type = "Rooftop"
    elif any(k in user_text for k in ["외벽", "벽면", "수직정원"]) or "wall" in text:
        location_type = "Wall"
    elif any(k in user_text for k in ["공원", "평지", "마당", "공터"]) or "park" in text:
        location_type = "Park"

    maint_level_max = None
    if any(k in user_text for k in ["관리 쉬", "저관리", "손이 덜", "어렵지"]):
        maint_level_max = 2
    elif "관리 보통" in user_text:
        maint_level_max = 3
    elif any(k in user_text for k in ["전문 관리", "관리 많이"]):
        maint_level_max = 5

    diversity_ratio = DEFAULT_VALUES["diversity_ratio"]
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", user_text)
    if m and any(k in user_text for k in ["수종", "한 종류", "다양"]):
        diversity_ratio = float(m.group(1)) / 100

    data = {
        "location_type": location_type,
        "location_label": DEFAULT_VALUES["location_label"].get(location_type),
        "total_budget": parse_budget_fallback(user_text),
        "total_area": parse_area_fallback(user_text),
        "target_co2": 0,
        "max_soil_depth": DEFAULT_VALUES["max_soil_depth_by_location"].get(location_type),
        "min_hardiness_zone": 4,
        "diversity_ratio": diversity_ratio,
        "maint_level_max": maint_level_max,
        "preferences": [],
        "reasoning_summary": "입력 문장에서 공간·예산·면적 조건을 간단히 추출했습니다.",
    }
    return normalize_parsed_params(data)


# =========================
# 버튼/슬라이더 입력 변환
# =========================
def build_params_from_controls(
    location_type: str,
    total_budget_million: int,
    total_area: float,
    max_soil_depth: float,
    min_hardiness_zone: int,
    diversity_percent: int,
    maint_label: str,
    target_co2: float,
    preferences: list[str],
) -> Dict[str, Any]:
    maint_map = {
        "쉬운 관리": 2,
        "보통 관리": 3,
        "전문 관리 가능": 5,
        "제한 없음": None,
    }
    data = {
        "location_type": location_type,
        "location_label": DEFAULT_VALUES["location_label"].get(location_type),
        "total_budget": total_budget_million * 10_000,
        "total_area": total_area,
        "target_co2": target_co2,
        "max_soil_depth": max_soil_depth,
        "min_hardiness_zone": min_hardiness_zone,
        "diversity_ratio": diversity_percent / 100,
        "maint_level_max": maint_map.get(maint_label),
        "preferences": preferences,
        "missing_fields": [],
        "reasoning_summary": "버튼과 슬라이더에서 선택한 조건을 바탕으로 녹화 설계 조건을 구성했습니다.",
    }
    return normalize_parsed_params(data)


# =========================
# 최적화 실행
# =========================
def run_optimization(parsed: Dict[str, Any]) -> Dict[str, Any]:
    constraints = ScenarioConstraints(
        location_type=parsed["location_type"],
        total_budget=float(parsed["total_budget"]),
        total_area=float(parsed["total_area"]),
        target_co2=float(parsed.get("target_co2") or 0),
        max_soil_depth=float(parsed.get("max_soil_depth") or DEFAULT_VALUES["max_soil_depth_by_location"].get(parsed["location_type"], 100)),
        min_hardiness_zone=int(parsed.get("min_hardiness_zone") or 4),
        diversity_ratio=float(parsed.get("diversity_ratio") or 0.3),
    )

    model = LandscapingOptimizationModel(str(DATA_PATH))

    log_buffer = io.StringIO()
    with contextlib.redirect_stdout(log_buffer):
        feasible = model.apply_hard_constraints(constraints)

        maint_level_max = parsed.get("maint_level_max")
        if maint_level_max is not None and "maint_cost_idx" in model.feasible_plants.columns:
            model.feasible_plants = model.feasible_plants[
                model.feasible_plants["maint_cost_idx"] <= int(maint_level_max)
            ].copy()
            feasible = model.feasible_plants

        if len(feasible) == 0:
            return {
                "constraints": constraints,
                "feasible": feasible,
                "result": None,
                "status": "NoFeasiblePlants",
                "logs": log_buffer.getvalue(),
            }

        solver = LandscapingOptimizationSolver(model)
        solver.build_problem(problem_name="GreenCity_AI_Optimization")
        result = solver.solve()

    return {
        "constraints": constraints,
        "feasible": feasible,
        "result": result,
        "status": result.status if result else "NoResult",
        "logs": log_buffer.getvalue(),
    }


def add_master_columns(detail_df: pd.DataFrame) -> pd.DataFrame:
    plant_df = pd.read_excel(DATA_PATH)
    cols = ["plant_id"]
    for col in ["maint_cost_idx", "soil_depth_req(cm)", "hardiness_zone"]:
        if col in plant_df.columns:
            cols.append(col)
    return detail_df.merge(plant_df[cols], on="plant_id", how="left")


# =========================
# 리포트 생성
# =========================
def build_report_text(parsed: Dict[str, Any], result: OptimizationResult) -> str:
    pine_equivalent = result.total_carbon / 5.4 if result.total_carbon else 0
    car_equivalent = result.total_carbon / 2000 if result.total_carbon else 0
    budget_use = result.total_cost / parsed["total_budget"] * 100
    area_use = result.total_area / parsed["total_area"] * 100

    report = f"""
GreenCity AI 조경 추천 리포트

- 공간 유형: {parsed.get('location_label') or parsed.get('location_type')}
- 가용 예산: {parsed['total_budget']:,.0f}원
- 조경 가능 면적: {parsed['total_area']:,.2f}㎡
- 연간 예상 탄소흡수량: {result.total_carbon:,.2f} kg CO₂/년
- 총 소요 비용: {result.total_cost:,.0f}원 (예산 사용률 {budget_use:.1f}%)
- 총 사용 면적: {result.total_area:,.2f}㎡ (면적 사용률 {area_use:.1f}%)
- 추천 수종 수: {len(result.plant_quantities)}종
- 총 식재 본수: {sum(result.plant_quantities.values()):,}본

환경 기여도 환산
- 소나무 약 {pine_equivalent:,.0f}그루를 심은 것과 유사한 연간 탄소흡수 효과입니다.
- 승용차 약 {car_equivalent:,.2f}대의 연간 CO₂ 배출량에 해당하는 양입니다.

추천 방식
- 먼저 토심, 위치 유형, 내한성 조건에 맞지 않는 식물을 제외했습니다.
- 이후 예산, 면적, 생태 다양성 조건을 만족하면서 연간 탄소흡수량이 최대가 되는 식재 조합을 계산했습니다.
""".strip()
    return report


# =========================
# 입력/결과 렌더링
# =========================
def show_missing_fields(parsed: Dict[str, Any]) -> None:
    labels = {
        "location_type": "공간 유형(옥상/외벽/공원)",
        "total_budget": "예산",
        "total_area": "면적",
    }
    missing_labels = [labels.get(x, x) for x in parsed["missing_fields"]]
    st.markdown(
        f"""
        <div class="warning-box">
            <b>조금 더 정보가 필요해요.</b><br>
            {', '.join(missing_labels)} 정보가 빠져 있습니다.<br>
            예: “예산은 5천만원, 면적은 200㎡인 옥상 정원을 만들고 싶어.”
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_condition_summary(parsed: Dict[str, Any]) -> None:
    st.markdown("### AI가 이해한 녹화 조건")
    cond1, cond2, cond3, cond4 = st.columns(4)
    cond1.metric("공간 유형", f"{location_icon(parsed.get('location_type'))} {parsed.get('location_label')}")
    cond2.metric("예산", won_to_label(parsed["total_budget"]))
    cond3.metric("면적", f"{parsed['total_area']:,.1f}㎡")
    cond4.metric("관리 수준", maint_label_from_value(parsed.get("maint_level_max")))

    if parsed.get("preferences"):
        pref_html = "".join([f'<span class="pill">{p}</span>' for p in parsed["preferences"]])
        st.markdown(f"<div style='margin-top: -8px;'>{pref_html}</div>", unsafe_allow_html=True)


def render_results(parsed: Dict[str, Any]) -> None:
    render_condition_summary(parsed)

    try:
        output = run_optimization(parsed)
    except Exception as e:
        st.error(f"추천 계산 중 문제가 발생했습니다. 조건을 조금 완화해서 다시 시도해주세요.\n\n오류: {e}")
        return

    feasible = output["feasible"]
    result = output["result"]

    if output["status"] == "NoFeasiblePlants":
        st.warning("현재 조건을 만족하는 식물이 없습니다. 토심, 위치 유형, 관리 난이도 조건을 완화해보세요.")
        return

    if result is None or result.status != "Optimal":
        st.warning("현재 조건에서는 최적 조합을 찾기 어렵습니다. 예산이나 면적을 늘리거나 제약 조건을 완화해보세요.")
        return

    st.markdown("---")
    st.markdown("### 추천 결과")
    st.success("입력 조건을 만족하는 최적 식재 포트폴리오를 찾았습니다.")

    budget_use = result.total_cost / parsed["total_budget"] * 100 if parsed["total_budget"] else 0
    area_use = result.total_area / parsed["total_area"] * 100 if parsed["total_area"] else 0
    pine_equivalent = result.total_carbon / 5.4 if result.total_carbon else 0
    car_equivalent = result.total_carbon / 2000 if result.total_carbon else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("연간 탄소흡수량", f"{result.total_carbon:,.1f} kg CO₂")
    m2.metric("예상 비용", f"{result.total_cost/10000:,.0f}만 원", f"예산 {budget_use:.1f}% 사용")
    m3.metric("사용 면적", f"{result.total_area:,.1f}㎡", f"면적 {area_use:.1f}% 사용")
    m4.metric("추천 식물", f"{len(result.plant_quantities)}종", f"총 {sum(result.plant_quantities.values()):,}본")

    st.markdown("### 추천 식재 조합")
    detail_df = add_master_columns(result.plant_details.copy())

    display_cols = {
        "plant_name": "식물명",
        "category": "분류",
        "quantity": "추천 수량",
        "total_carbon": "예상 탄소흡수량(kg/년)",
        "total_cost": "예상 비용(원)",
        "total_area": "점유 면적(㎡)",
    }

    table_df = detail_df[list(display_cols.keys())].rename(columns=display_cols)
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    st.markdown("### 왜 이 조합을 추천했나요?")
    st.markdown(
        f"""
        <div class="soft-card">
            <b>1. 공간 조건에 맞지 않는 식물은 먼저 제외했습니다.</b><br>
            <span class="small-muted">{parsed.get('location_label')} 조건, 최대 토심 {parsed.get('max_soil_depth'):.0f}cm, 내한성 Zone {parsed.get('min_hardiness_zone')} 기준을 적용했습니다.</span><br><br>
            <b>2. 예산과 면적 안에서 탄소흡수량이 가장 커지는 조합을 계산했습니다.</b><br>
            <span class="small-muted">총 {len(feasible)}개 후보 식물 중에서 비용·면적·다양성 조건을 동시에 만족하는 식재 조합을 선택했습니다.</span><br><br>
            <b>3. 한 식물이 너무 많이 몰리지 않도록 구성했습니다.</b><br>
            <span class="small-muted">한 수종이 전체 면적의 {parsed.get('diversity_ratio', 0.3)*100:.0f}%를 넘지 않도록 제한했습니다.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        fig_carbon = px.bar(
            detail_df,
            x="plant_name",
            y="total_carbon",
            title="식물별 연간 탄소흡수량",
            labels={"plant_name": "식물", "total_carbon": "kg CO₂/년"},
        )
        st.plotly_chart(fig_carbon, use_container_width=True)

    with right:
        fig_area = px.pie(
            detail_df,
            names="plant_name",
            values="total_area",
            title="식물별 면적 구성비",
        )
        st.plotly_chart(fig_area, use_container_width=True)

    st.markdown("### 예상 유지관리 비용 시뮬레이션")
    if "maint_cost_idx" in detail_df.columns:
        base_annual = (detail_df["total_cost"] * detail_df["maint_cost_idx"].fillna(3) * 0.015).sum()
    else:
        base_annual = result.total_cost * 0.05

    years = list(range(1, 11))
    lcc_df = pd.DataFrame({
        "연도": years,
        "누적 유지관리 비용(원)": [base_annual * y for y in years],
    })
    fig_lcc = px.line(
        lcc_df,
        x="연도",
        y="누적 유지관리 비용(원)",
        markers=True,
        title="10년 누적 유지관리 비용 추정",
    )
    st.plotly_chart(fig_lcc, use_container_width=True)

    st.markdown("### 환경 효과 환산")
    st.markdown(
        f"""
        <div class="soft-card">
            🌲 소나무 약 <b>{pine_equivalent:,.0f}그루</b>를 심은 것과 유사한 연간 탄소흡수 효과입니다.<br>
            🚗 승용차 약 <b>{car_equivalent:,.2f}대</b>의 연간 CO₂ 배출량에 해당하는 양입니다.<br>
            <span class="small-muted">※ 환산값은 사용자 이해를 돕기 위한 단순 비교 지표입니다.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    report = build_report_text(parsed, result)
    st.download_button(
        "📄 추천 리포트 다운로드",
        data=report.encode("utf-8-sig"),
        file_name="greencity_ai_report.txt",
        mime="text/plain",
        use_container_width=True,
    )

    csv_bytes = table_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📊 식재 조합 CSV 다운로드",
        data=csv_bytes,
        file_name="greencity_ai_planting_plan.csv",
        mime="text/csv",
        use_container_width=True,
    )


# =========================
# Streamlit 앱
# =========================
st.set_page_config(page_title="GreenCity AI", page_icon="🌿", layout="wide")
inject_css()

with st.sidebar:
    st.markdown("### ⚙️ 운영자 설정")
    env_key = os.getenv("GEMINI_API_KEY", "")
    api_key_input = st.text_input(
        "Gemini API Key",
        type="password",
        value=env_key,
        placeholder="환경변수 GEMINI_API_KEY 또는 직접 입력",
    )
    model_name = st.text_input("Gemini 모델명", value="gemini-2.5-flash")
    use_fallback = st.toggle("API 오류 시 임시 분석 사용", value=True)
    st.divider()
    st.caption("자연어 직접 입력 모드에서만 Gemini API를 사용합니다.")

st.markdown(
    """
    <div class="hero">
        <h1>🌿 GreenCity AI</h1>
        <p>원하는 녹화 공간을 설명하거나 조건을 선택하면, AI와 최적화 엔진이 최적의 식재 포트폴리오를 추천합니다.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

mode_nl, mode_controls = st.tabs(["✍️ 자연어로 직접 입력", "🎛️ 버튼·슬라이더로 선택"])

with mode_nl:
    st.markdown("### 문장으로 조건을 입력하세요")
    st.caption("예산, 면적, 공간 유형을 문장으로 편하게 입력하면 AI가 조건을 이해합니다.")

    example_prompts = {
        "옥상 정원 예시": "예산은 5천만원이고 면적은 200㎡인 관리 쉬운 옥상 정원을 만들고 싶어. 토심은 30cm, 한 수종은 30% 이하로 다양하게 구성해줘.",
        "수직정원 예시": "건물 외벽에 수직정원을 만들고 싶어. 예산 2천만원, 면적 80㎡, 관리가 쉬운 식물 위주로 추천해줘.",
        "공원 조경 예시": "공원에 평지 조경을 하려고 해. 예산은 1억원이고 면적은 500㎡야. 탄소흡수량이 높은 식재 조합으로 추천해줘.",
        "20평 옥상 예시": "20평 정도 되는 옥상에 손이 많이 안 가는 정원을 만들고 싶어. 예산은 3천만원 정도야.",
    }

    selected_example = st.selectbox("빠른 예시", ["직접 입력"] + list(example_prompts.keys()), key="nl_example")
    default_text = example_prompts.get(selected_example, "")

    user_request = st.text_area(
        "요청 내용",
        value=default_text,
        height=150,
        placeholder="예: 예산은 5천만원이고 면적은 200㎡인 관리 쉬운 옥상 정원을 만들고 싶어.",
        label_visibility="collapsed",
        key="nl_text",
    )

    run_nl = st.button("🌱 자연어 조건으로 녹화 설계 받기", type="primary", use_container_width=True, key="run_nl")

    if not run_nl:
        st.markdown(
            """
            <div class="soft-card">
                <b>입력 예시</b><br>
                <span class="small-muted">“예산은 5천만원이고 면적은 200㎡인 관리 쉬운 옥상 정원을 만들고 싶어.”</span><br><br>
                <span class="pill">옥상 정원</span><span class="pill">수직정원</span><span class="pill">공원 조경</span><span class="pill">저관리</span><span class="pill">탄소흡수량 최대화</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        if not user_request.strip():
            st.warning("먼저 원하는 조경 조건을 문장으로 입력해주세요.")
        else:
            with st.spinner("AI가 요청을 이해하고 최적 식재 조합을 계산하는 중입니다..."):
                try:
                    parsed = parse_with_gemini(user_request, api_key_input, model_name=model_name)
                except Exception:
                    if use_fallback:
                        parsed = parse_fallback(user_request)
                    else:
                        st.error("AI가 요청을 분석하지 못했습니다. API Key 또는 모델 설정을 확인해주세요.")
                        parsed = None

            if parsed:
                if parsed.get("missing_fields"):
                    show_missing_fields(parsed)
                else:
                    render_results(parsed)

with mode_controls:
    st.markdown("### 조건을 선택하세요")
    st.caption("시연 중 빠르게 조건을 바꾸고 싶을 때 사용하는 선택형 모드입니다. Gemini API 없이 바로 계산합니다.")

    with st.form("control_form"):
        loc_choice = st.radio(
            "공간 유형",
            options=["Rooftop", "Wall", "Park"],
            format_func=lambda x: f"{location_icon(x)} {DEFAULT_VALUES['location_label'][x]}",
            horizontal=True,
        )

        col_a, col_b = st.columns(2)
        with col_a:
            total_budget_million = st.slider("예산", min_value=500, max_value=20_000, value=5_000, step=500, help="단위: 만 원")
            total_area = st.slider("조경 가능 면적", min_value=10.0, max_value=1000.0, value=200.0, step=10.0, help="단위: ㎡")
            target_co2 = st.number_input("최소 탄소흡수 목표량(선택)", min_value=0.0, value=0.0, step=50.0, help="0이면 목표 제약을 적용하지 않습니다. 단위: kg CO₂/년")
        with col_b:
            default_depth = DEFAULT_VALUES["max_soil_depth_by_location"][loc_choice]
            max_soil_depth = st.slider("최대 허용 토심", min_value=10.0, max_value=120.0, value=float(default_depth), step=5.0, help="단위: cm")
            min_hardiness_zone = st.slider("최소 내한성 Zone", min_value=1, max_value=7, value=4, step=1)
            diversity_percent = st.slider("한 수종 최대 면적 비율", min_value=10, max_value=60, value=30, step=5, help="생태 다양성 제약")

        maint_choice = st.radio(
            "관리 가능 수준",
            options=["쉬운 관리", "보통 관리", "전문 관리 가능", "제한 없음"],
            horizontal=True,
        )
        preferences = st.multiselect(
            "선호 요소",
            options=["저관리", "탄소흡수", "단풍", "꽃", "상록", "허브", "도시텃밭", "미세먼지 저감"],
            default=["탄소흡수"],
        )

        run_controls = st.form_submit_button("🌿 선택 조건으로 녹화 설계 받기", type="primary", use_container_width=True)

    if run_controls:
        parsed = build_params_from_controls(
            location_type=loc_choice,
            total_budget_million=total_budget_million,
            total_area=total_area,
            max_soil_depth=max_soil_depth,
            min_hardiness_zone=min_hardiness_zone,
            diversity_percent=diversity_percent,
            maint_label=maint_choice,
            target_co2=target_co2,
            preferences=preferences,
        )
        render_results(parsed)
