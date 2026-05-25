"""
도심 외부 조경 최적화 엔진
PuLP 라이브러리 기반 선형 계획법(Linear Programming) 솔버
"""

import pulp
import pandas as pd
from typing import Dict, List, Optional
from optimization_model import LandscapingOptimizationModel, ScenarioConstraints
from dataclasses import dataclass


@dataclass
class OptimizationResult:
    """최적화 결과 데이터 클래스"""
    status: str  # 'Optimal', 'Infeasible', 'Unbounded'
    total_carbon: float  # kg CO₂/년
    total_cost: float  # 원
    total_area: float  # m²
    plant_quantities: Dict[str, int]  # {plant_id: quantity}
    plant_details: pd.DataFrame  # 상세 정보
    
    
class LandscapingOptimizationSolver:
    """
    조경 최적화 솔버 클래스
    PuLP 라이브러리를 사용하여 정수 선형 계획법(Integer Linear Programming) 문제 해결
    """
    
    def __init__(self, model: LandscapingOptimizationModel):
        """
        Args:
            model: 수학적 모델이 정의된 LandscapingOptimizationModel 인스턴스
        """
        self.model = model
        self.prob = None
        self.variables = {}
        self.result = None
        
    def build_problem(self, problem_name: str = "LandscapingOptimization"):
        """
        PuLP 문제 객체 생성 및 변수/제약조건 설정
        
        Args:
            problem_name: 문제 이름
        """
        if self.model.feasible_plants is None:
            raise ValueError("모델에 apply_hard_constraints()를 먼저 실행해야 합니다.")
        
        # 1. 문제 정의: 최대화 문제
        self.prob = pulp.LpProblem(problem_name, pulp.LpMaximize)
        
        # 2. 최적화 행렬 데이터 추출
        matrices = self.model.get_optimization_matrices()
        
        plant_ids = matrices['plant_ids']
        carbon_coeffs = matrices['objective_coeffs']
        cost_coeffs = matrices['cost_coeffs']
        area_coeffs = matrices['area_coeffs']
        
        print(f"\n{'='*70}")
        print(f"🔧 PuLP 최적화 문제 구성 중...")
        print(f"{'='*70}\n")
        
        # 3. 결정 변수 생성: x_i (각 식물의 식재 본수)
        print(f"✓ 결정 변수 생성: {len(plant_ids)}개 수종")
        self.variables = pulp.LpVariable.dicts(
            "x",
            plant_ids,
            lowBound=0,
            cat=pulp.LpInteger  # 정수 제약
        )
        
        # 4. 목적 함수 설정: Maximize Σ(C_i × x_i)
        print(f"✓ 목적 함수 설정: 탄소 흡수량 최대화")
        self.prob += pulp.lpSum([
            carbon_coeffs[i] * self.variables[plant_ids[i]]
            for i in range(len(plant_ids))
        ]), "Total_Carbon_Absorption"
        
        # 5. 제약 조건 추가
        constraints = self.model.constraints
        
        # 제약 1: 예산 제약
        print(f"✓ 예산 제약 추가: ≤ {constraints.total_budget:,.0f}원")
        self.prob += pulp.lpSum([
            cost_coeffs[i] * self.variables[plant_ids[i]]
            for i in range(len(plant_ids))
        ]) <= constraints.total_budget, "Budget_Constraint"
        
        # 제약 2: 면적 제약
        print(f"✓ 면적 제약 추가: ≤ {constraints.total_area:,.1f}m²")
        self.prob += pulp.lpSum([
            area_coeffs[i] * self.variables[plant_ids[i]]
            for i in range(len(plant_ids))
        ]) <= constraints.total_area, "Area_Constraint"
        
        # 제약 3: 생태 다양성 제약 (각 수종별)
        max_area_per_species = constraints.total_area * constraints.diversity_ratio
        print(f"✓ 생태 다양성 제약 추가: 각 수종 ≤ {max_area_per_species:,.1f}m² ({constraints.diversity_ratio*100:.0f}%)")
        
        for i in range(len(plant_ids)):
            self.prob += (
                area_coeffs[i] * self.variables[plant_ids[i]] <= max_area_per_species,
                f"Diversity_{plant_ids[i]}"
            )
        
        # 제약 4: 최소 탄소 목표 (옵션)
        if constraints.target_co2 > 0:
            print(f"✓ 최소 탄소 목표 제약 추가: ≥ {constraints.target_co2:,.1f}kg CO₂/년")
            self.prob += pulp.lpSum([
                carbon_coeffs[i] * self.variables[plant_ids[i]]
                for i in range(len(plant_ids))
            ]) >= constraints.target_co2, "Minimum_Carbon_Target"
        
        print(f"\n{'='*70}\n")
        
    def solve(self, solver_name: str = 'PULP_CBC_CMD') -> OptimizationResult:
        """
        최적화 문제 해결
        
        Args:
            solver_name: 사용할 솔버 ('PULP_CBC_CMD', 'GLPK_CMD' 등)
            
        Returns:
            OptimizationResult 객체
        """
        if self.prob is None:
            raise ValueError("build_problem()을 먼저 실행해야 합니다.")
        
        print(f"{'='*70}")
        print(f"🚀 최적화 엔진 실행 중...")
        print(f"{'='*70}\n")
        
        # 솔버 실행
        if solver_name == 'PULP_CBC_CMD':
            solver = pulp.PULP_CBC_CMD(msg=1)
        else:
            solver = pulp.getSolver(solver_name)
        
        self.prob.solve(solver)
        
        # 결과 상태 확인
        status = pulp.LpStatus[self.prob.status]
        print(f"\n{'='*70}")
        print(f"📊 최적화 결과: {status}")
        print(f"{'='*70}\n")
        
        if status != 'Optimal':
            return OptimizationResult(
                status=status,
                total_carbon=0,
                total_cost=0,
                total_area=0,
                plant_quantities={},
                plant_details=pd.DataFrame()
            )
        
        # 최적해 추출
        plant_quantities = {}
        for plant_id, var in self.variables.items():
            qty = int(var.varValue) if var.varValue is not None else 0
            if qty > 0:
                plant_quantities[plant_id] = qty
        
        # 상세 결과 계산
        result_details = []
        total_carbon = 0
        total_cost = 0
        total_area = 0
        
        df = self.model.feasible_plants
        
        for plant_id, qty in plant_quantities.items():
            plant_row = df[df['plant_id'] == plant_id].iloc[0]
            
            carbon = plant_row['carbon_factor'] * qty
            cost = plant_row['cost_per_unit'] * qty
            area = plant_row['space_req_m2'] * qty
            
            total_carbon += carbon
            total_cost += cost
            total_area += area
            
            result_details.append({
                'plant_id': plant_id,
                'plant_name': plant_row['plant_name'],
                'category': plant_row['category'],
                'quantity': qty,
                'carbon_per_unit': plant_row['carbon_factor'],
                'total_carbon': carbon,
                'cost_per_unit': plant_row['cost_per_unit'],
                'total_cost': cost,
                'area_per_unit': plant_row['space_req_m2'],
                'total_area': area
            })
        
        result_df = pd.DataFrame(result_details)
        result_df = result_df.sort_values('total_carbon', ascending=False)
        
        self.result = OptimizationResult(
            status=status,
            total_carbon=total_carbon,
            total_cost=total_cost,
            total_area=total_area,
            plant_quantities=plant_quantities,
            plant_details=result_df
        )
        
        return self.result
    
    def print_solution(self):
        """최적화 결과를 보기 좋게 출력"""
        if self.result is None or self.result.status != 'Optimal':
            print("❌ 최적해를 찾지 못했습니다.")
            return
        
        result = self.result
        
        print(f"\n{'='*80}")
        print(f"🎯 최적화 솔루션 (Optimal Solution)")
        print(f"{'='*80}\n")
        
        print(f"📈 목적 함수 값 (연간 총 탄소 흡수량)")
        print(f"   └─ {result.total_carbon:,.2f} kg CO₂/년\n")
        
        print(f"💰 총 소요 예산")
        print(f"   └─ {result.total_cost:,.0f} 원")
        print(f"   └─ 예산 사용률: {result.total_cost/self.model.constraints.total_budget*100:.1f}%\n")
        
        print(f"📏 총 점유 면적")
        print(f"   └─ {result.total_area:,.2f} m²")
        print(f"   └─ 면적 사용률: {result.total_area/self.model.constraints.total_area*100:.1f}%\n")
        
        print(f"🌳 식재 계획 상세 (총 {len(result.plant_quantities)}종)")
        print(f"{'─'*80}")
        
        for idx, row in result.plant_details.iterrows():
            print(f"\n   [{idx+1}] {row['plant_name']} ({row['category']})")
            print(f"       ├─ 식재 본수: {row['quantity']:,}그루")
            print(f"       ├─ 탄소 흡수: {row['total_carbon']:,.2f} kg/년 "
                  f"(= {row['carbon_per_unit']:.1f} × {row['quantity']})")
            print(f"       ├─ 소요 비용: {row['total_cost']:,.0f}원 "
                  f"(= {row['cost_per_unit']:,} × {row['quantity']})")
            print(f"       └─ 점유 면적: {row['total_area']:,.2f}m² "
                  f"(= {row['area_per_unit']:.2f} × {row['quantity']}) "
                  f"[{row['total_area']/self.model.constraints.total_area*100:.1f}%]")
        
        print(f"\n{'='*80}\n")
        
        # 환경 기여도 비유
        print(f"🌍 환경 기여도 해석")
        print(f"{'─'*80}")
        # 소나무 1그루 연간 탄소 흡수량 약 5.4kg 기준
        pine_equivalent = result.total_carbon / 5.4
        print(f"   • 소나무 {pine_equivalent:,.0f}그루를 심은 것과 동일한 효과")
        
        # 승용차 1대 연간 CO2 배출량 약 2,000kg 기준
        car_equivalent = result.total_carbon / 2000
        print(f"   • 승용차 {car_equivalent:,.2f}대의 연간 배출량 상쇄 효과\n")


if __name__ == "__main__":
    print("🌳 조경 최적화 엔진 - PuLP 솔버 테스트\n")
    
    # 시나리오 1: 중규모 옥상 정원
    print("\n" + "="*80)
    print("시나리오 1: 중규모 옥상 정원 (예산 5천만원, 면적 200m²)")
    print("="*80)
    
    model = LandscapingOptimizationModel('/mnt/project/master_table.xlsx')
    
    scenario = ScenarioConstraints(
        location_type='Rooftop',
        total_budget=50_000_000,
        total_area=200,
        max_soil_depth=30,
        min_hardiness_zone=4,
        diversity_ratio=0.3
    )
    
    # Hard constraint 필터링
    model.apply_hard_constraints(scenario)
    
    # 솔버 실행
    solver = LandscapingOptimizationSolver(model)
    solver.build_problem()
    result = solver.solve()
    solver.print_solution()
    
    # 시나리오 2: 공원 조경 (대규모)
    print("\n\n" + "="*80)
    print("시나리오 2: 공원 조경 (예산 2억원, 면적 1000m²)")
    print("="*80)
    
    model2 = LandscapingOptimizationModel('/mnt/project/master_table.xlsx')
    
    scenario2 = ScenarioConstraints(
        location_type='Park',
        total_budget=200_000_000,
        total_area=1000,
        max_soil_depth=100,  # 공원은 토심 제약 완화
        min_hardiness_zone=4,
        diversity_ratio=0.3,
        target_co2=3000  # 최소 3톤 목표
    )
    
    model2.apply_hard_constraints(scenario2)
    
    solver2 = LandscapingOptimizationSolver(model2)
    solver2.build_problem()
    result2 = solver2.solve()
    solver2.print_solution()
