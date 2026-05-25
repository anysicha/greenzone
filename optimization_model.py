"""
도심 외부 조경 최적화 모델 - 수학적 수식 정의
Linear Programming 기반 최적화 엔진
"""

import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ScenarioConstraints:
    """사용자 입력 시나리오 제약 조건"""
    location_type: str  # 'Rooftop', 'Wall', 'Park'
    total_budget: float  # 원
    total_area: float    # m²
    target_co2: float = 0  # kg/년 (최소 목표, 옵션)
    max_soil_depth: float = 100  # cm (옥상은 30cm 등으로 제한)
    min_hardiness_zone: int = 3  # 지역 최저 내한성
    diversity_ratio: float = 0.3  # 생태 다양성 제약 (30%)


class LandscapingOptimizationModel:
    """
    조경 최적화 수리 모델 클래스
    
    목적 함수: Maximize Z = Σ(C_i × x_i)
    - Z: 총 탄소 흡수량
    - C_i: 수종 i의 연간 탄소 흡수 계수
    - x_i: 수종 i의 식재 본수
    
    제약 조건:
    1. 예산 제약: Σ(P_i × x_i) ≤ B
    2. 면적 제약: Σ(A_i × x_i) ≤ S
    3. 다양성 제약: (A_i × x_i) ≤ S × 0.3 (각 수종)
    4. 환경 적합성: Hard constraint (사전 필터링)
    """
    
    def __init__(self, master_data_path: str):
        """
        Args:
            master_data_path: 식물 마스터 데이터 엑셀 파일 경로
        """
        self.master_df = pd.read_excel(master_data_path)
        self.feasible_plants = None
        self.constraints = None
        
    def apply_hard_constraints(self, constraints: ScenarioConstraints) -> pd.DataFrame:
        """
        환경 적합성 필터 (Hard Constraint)
        물리적으로 생존 불가능한 식물을 사전 필터링
        
        N = { i | soil_depth_i ≤ User_Depth AND hardiness_i >= User_Zone }
        
        Args:
            constraints: 사용자 입력 제약 조건
            
        Returns:
            필터링된 후보 식물 DataFrame
        """
        df = self.master_df.copy()
        
        print(f"\n{'='*60}")
        print(f"[Hard Constraint 필터링 시작]")
        print(f"{'='*60}")
        print(f"초기 후보 수종: {len(df)}개")
        
        # 1. 토심 제약
        initial_count = len(df)
        df = df[df['soil_depth_req(cm)'] <= constraints.max_soil_depth]
        print(f"✓ 토심 제약 ({constraints.max_soil_depth}cm 이하): {initial_count}개 → {len(df)}개")
        
        # 2. 내한성 제약
        initial_count = len(df)
        df = df[df['hardiness_zone'] >= constraints.min_hardiness_zone]
        print(f"✓ 내한성 제약 (Zone {constraints.min_hardiness_zone} 이상): {initial_count}개 → {len(df)}개")
        
        # 3. 위치별 제약
        if constraints.location_type == 'Rooftop':
            initial_count = len(df)
            # 옥상: 관목, 초본, 지피식물만 가능 (교목 제외)
            df = df[df['category'] != '교목']
            print(f"✓ 옥상 제약 (교목 제외): {initial_count}개 → {len(df)}개")
            
        elif constraints.location_type == 'Wall':
            initial_count = len(df)
            # 벽면: 관목, 지피식물만 가능
            df = df[df['category'].isin(['관목', '지피식물'])]
            print(f"✓ 벽면 제약 (관목/지피식물만): {initial_count}개 → {len(df)}개")
        
        print(f"\n최종 후보 수종: {len(df)}개")
        print(f"{'='*60}\n")
        
        self.feasible_plants = df
        self.constraints = constraints
        
        return df
    
    def get_optimization_matrices(self) -> Dict:
        """
        최적화 엔진에 투입할 행렬 데이터 추출
        
        Returns:
            {
                'objective_coeffs': 목적 함수 계수 (탄소 흡수량),
                'cost_coeffs': 예산 제약 계수,
                'area_coeffs': 면적 제약 계수,
                'plant_ids': 식물 ID 리스트,
                'plant_names': 식물명 리스트
            }
        """
        if self.feasible_plants is None:
            raise ValueError("apply_hard_constraints()를 먼저 실행해야 합니다.")
        
        df = self.feasible_plants
        
        matrices = {
            'objective_coeffs': df['carbon_factor'].tolist(),  # C_i
            'cost_coeffs': df['cost_per_unit'].tolist(),      # P_i
            'area_coeffs': df['space_req_m2'].tolist(),       # A_i
            'plant_ids': df['plant_id'].tolist(),
            'plant_names': df['plant_name'].tolist(),
            'n_plants': len(df)
        }
        
        return matrices
    
    def print_model_formulation(self):
        """수학적 수식 정의를 보기 좋게 출력"""
        
        print("\n" + "="*80)
        print("📐 선형 계획법 모델 수식 정의 (Linear Programming Formulation)")
        print("="*80)
        
        print("\n1️⃣  결정 변수 (Decision Variables)")
        print("   x_i : 수종 i를 식재할 본수 (그루)")
        print(f"   i ∈ {{1, 2, ..., {self.feasible_plants['plant_id'].nunique()}}} (후보 식물 ID)")
        print("   x_i ≥ 0, x_i는 정수")
        
        print("\n2️⃣  목적 함수 (Objective Function)")
        print("   Maximize Z = Σ(C_i × x_i)")
        print("   ")
        print("   여기서:")
        print("   - Z: 연간 총 탄소 흡수량 (kg CO₂/년)")
        print("   - C_i: 수종 i의 연간 탄소 흡수 계수 (kg CO₂/그루/년)")
        
        print("\n3️⃣  제약 조건 (Constraints)")
        
        print("\n   ① 예산 제약 (Budget Constraint)")
        print(f"      Σ(P_i × x_i) ≤ {self.constraints.total_budget:,.0f} 원")
        print("      - P_i: 수종 i의 식재 단가 (원/그루)")
        
        print("\n   ② 면적 제약 (Area Constraint)")
        print(f"      Σ(A_i × x_i) ≤ {self.constraints.total_area:,.1f} m²")
        print("      - A_i: 수종 i의 단위 점유 면적 (m²/그루)")
        
        print("\n   ③ 생태 다양성 제약 (Diversity Constraint)")
        print(f"      (A_i × x_i) ≤ {self.constraints.total_area * self.constraints.diversity_ratio:,.1f} m²  (모든 i에 대해)")
        print(f"      - 각 수종이 전체 면적의 {self.constraints.diversity_ratio*100:.0f}% 이하 차지")
        
        if self.constraints.target_co2 > 0:
            print("\n   ④ 최소 탄소 목표 제약 (Minimum Carbon Target)")
            print(f"      Σ(C_i × x_i) ≥ {self.constraints.target_co2:,.1f} kg CO₂/년")
        
        print("\n4️⃣  환경 적합성 (Hard Constraints - 사전 필터링 완료)")
        print(f"   - 최대 토심: {self.constraints.max_soil_depth} cm 이하")
        print(f"   - 최소 내한성: Zone {self.constraints.min_hardiness_zone} 이상")
        print(f"   - 위치 타입: {self.constraints.location_type}")
        
        print("\n" + "="*80 + "\n")
    
    def get_feasible_summary(self) -> pd.DataFrame:
        """필터링된 후보 식물 요약 정보"""
        if self.feasible_plants is None:
            return None
        
        summary = self.feasible_plants[[
            'plant_id', 'plant_name', 'category', 
            'carbon_factor', 'cost_per_unit', 'space_req_m2',
            'soil_depth_req(cm)', 'hardiness_zone'
        ]].copy()
        
        summary = summary.sort_values('carbon_factor', ascending=False)
        
        return summary


if __name__ == "__main__":
    # 테스트 실행
    print("🌳 조경 최적화 모델 - 수학적 정의 테스트\n")
    
    # 모델 초기화
    model = LandscapingOptimizationModel('/mnt/project/master_table.xlsx')
    
    # 시나리오 1: 옥상 정원
    print("\n[시나리오 1: 옥상 정원 조성]")
    scenario1 = ScenarioConstraints(
        location_type='Rooftop',
        total_budget=50_000_000,  # 5천만원
        total_area=200,           # 200m²
        max_soil_depth=30,        # 옥상 제약
        min_hardiness_zone=4
    )
    
    feasible = model.apply_hard_constraints(scenario1)
    model.print_model_formulation()
    
    print("\n후보 수종 목록 (탄소 흡수량 내림차순):")
    print(model.get_feasible_summary().to_string(index=False))
    
    matrices = model.get_optimization_matrices()
    print(f"\n\n최적화 행렬 데이터:")
    print(f"- 목적 함수 계수 (탄소): {matrices['objective_coeffs'][:5]}...")
    print(f"- 비용 계수: {matrices['cost_coeffs'][:5]}...")
    print(f"- 면적 계수: {matrices['area_coeffs'][:5]}...")
