"""
조경 최적화 시스템 - 종합 데모
Phase 2 완료: 수리 모델링 및 최적화 엔진 검증
"""

from optimization_model import LandscapingOptimizationModel, ScenarioConstraints
from optimization_solver import LandscapingOptimizationSolver
import pandas as pd


def run_scenario(scenario_name: str, model_path: str, constraints: ScenarioConstraints):
    """시나리오 실행 함수"""
    
    print("\n" + "="*90)
    print(f"📋 {scenario_name}")
    print("="*90)
    
    # 모델 초기화
    model = LandscapingOptimizationModel(model_path)
    
    # Hard constraint 필터링
    feasible = model.apply_hard_constraints(constraints)
    
    if len(feasible) == 0:
        print("\n❌ 제약 조건을 만족하는 식물이 없습니다.")
        return None
    
    # 수학적 모델 정의 출력
    model.print_model_formulation()
    
    # 최적화 솔버 실행
    solver = LandscapingOptimizationSolver(model)
    solver.build_problem(problem_name=scenario_name.replace(" ", "_"))
    result = solver.solve()
    
    # 결과 출력
    solver.print_solution()
    
    return result


def main():
    """Phase 2 종합 데모 실행"""
    
    print("\n" + "🌳"*40)
    print("도심 외부 조경 최적화 AI 시스템 - Phase 2 완료 데모")
    print("수리 모델링 및 최적화 엔진 검증")
    print("🌳"*40 + "\n")
    
    master_path = '/mnt/project/master_table.xlsx'
    results = {}
    
    # ==================== 시나리오 1: 소규모 옥상 정원 ====================
    scenario1 = ScenarioConstraints(
        location_type='Rooftop',
        total_budget=30_000_000,  # 3천만원
        total_area=100,           # 100m²
        max_soil_depth=30,
        min_hardiness_zone=4,
        diversity_ratio=0.3
    )
    
    results['소규모 옥상'] = run_scenario(
        "시나리오 1: 소규모 옥상 정원 (예산 3천만원, 100m²)",
        master_path,
        scenario1
    )
    
    # ==================== 시나리오 2: 중규모 옥상 정원 ====================
    scenario2 = ScenarioConstraints(
        location_type='Rooftop',
        total_budget=50_000_000,  # 5천만원
        total_area=200,           # 200m²
        max_soil_depth=30,
        min_hardiness_zone=4,
        diversity_ratio=0.3
    )
    
    results['중규모 옥상'] = run_scenario(
        "시나리오 2: 중규모 옥상 정원 (예산 5천만원, 200m²)",
        master_path,
        scenario2
    )
    
    # ==================== 시나리오 3: 공원 조경 (교목 포함) ====================
    scenario3 = ScenarioConstraints(
        location_type='Park',
        total_budget=100_000_000,  # 1억원
        total_area=500,            # 500m²
        max_soil_depth=100,        # 공원은 토심 여유
        min_hardiness_zone=4,
        diversity_ratio=0.3
    )
    
    results['공원 조경'] = run_scenario(
        "시나리오 3: 공원 조경 (예산 1억원, 500m²)",
        master_path,
        scenario3
    )
    
    # ==================== 시나리오 4: 건물 외벽 녹화 ====================
    scenario4 = ScenarioConstraints(
        location_type='Wall',
        total_budget=20_000_000,  # 2천만원
        total_area=80,            # 80m² (벽면)
        max_soil_depth=25,
        min_hardiness_zone=5,
        diversity_ratio=0.4
    )
    
    results['건물 외벽'] = run_scenario(
        "시나리오 4: 건물 외벽 녹화 (예산 2천만원, 80m²)",
        master_path,
        scenario4
    )
    
    # ==================== 결과 비교 ====================
    print("\n\n" + "="*90)
    print("📊 전체 시나리오 비교 요약")
    print("="*90 + "\n")
    
    comparison_data = []
    for name, result in results.items():
        if result and result.status == 'Optimal':
            comparison_data.append({
                '시나리오': name,
                '총 탄소흡수량(kg/년)': f"{result.total_carbon:,.1f}",
                '소요예산(만원)': f"{result.total_cost/10000:,.0f}",
                '면적사용률(%)': f"{result.total_area/(result.total_area)*100:.1f}",
                '수종개수': len(result.plant_quantities),
                '총 식재본수': sum(result.plant_quantities.values())
            })
    
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.to_string(index=False))
    
    print("\n\n" + "="*90)
    print("✅ Phase 2: 최적화 및 수리 모델링 완료")
    print("="*90)
    print("\n주요 산출물:")
    print("  1. 수학적 수식 정의 모듈 (optimization_model.py)")
    print("  2. PuLP 기반 최적화 솔버 (optimization_solver.py)")
    print("  3. 다양한 시나리오 검증 완료")
    print("\n다음 단계: Phase 3 - AI 서비스 및 인터페이스 구현")
    print("="*90 + "\n")


if __name__ == "__main__":
    main()
