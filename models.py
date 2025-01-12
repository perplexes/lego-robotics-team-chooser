from dataclasses import dataclass, field
from typing import List, Dict
import pandas as pd

ROLE_COLUMNS = [
    'team_captain',
    'innovation_project_leader',
    'mission_strategist',
    'public_relations_lead',
    'lego_lead_builder',
    'lead_coder'
]

@dataclass
class OptimizationConfig:
    min_team_size: int = 4
    max_team_size: int = 8
    special_team_size: int = 4
    min_roles_per_student: int = 1
    max_roles_per_student: int = 2
    sixth_grade_multi_role_penalty: int = -5

@dataclass
class TeamData:
    data: pd.DataFrame
    num_students: int
    female_indices: List[int]
    eighth_grade_indices: List[int]
    config: OptimizationConfig
    role_columns: List[str] = field(default_factory=lambda: ROLE_COLUMNS)

@dataclass
class OptimizationResult:
    student_assignments: pd.DataFrame
    objective_value: float
    status: str
