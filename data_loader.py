import pandas as pd
import math
from models import TeamData, OptimizationConfig, ROLE_COLUMNS

def load_team_data(data_path: str) -> TeamData:
    """Load and process the student data from CSV."""
    data = pd.read_csv(data_path)
    num_students = len(data)
    
    # Get special group indices
    female_indices = data[data['Gender'] == 'F'].index.tolist()
    eighth_grade_indices = data[data['Grade'] == '8th'].index.tolist()
    
    # Calculate number of teams needed
    remaining_students = num_students - len(female_indices) - len(eighth_grade_indices)
    config = OptimizationConfig()
    additional_teams = math.ceil(remaining_students / config.max_team_size)
    num_teams = 2 + additional_teams  # 2 special teams + additional teams
    
    print_data_statistics(data, num_teams, female_indices, eighth_grade_indices, remaining_students)
    
    return TeamData(
        data=data,
        num_students=num_students,
        num_teams=num_teams,
        female_indices=female_indices,
        eighth_grade_indices=eighth_grade_indices,
        config=config
    )

def print_data_statistics(data: pd.DataFrame, num_teams: int, female_indices: list, 
                         eighth_grade_indices: list, remaining_students: int) -> None:
    """Print statistics about the loaded data."""
    print(f"Number of students: {len(data)}")
    print("\nGender distribution:")
    print(data['Gender'].value_counts())
    print("\nGrade distribution:")
    print(data['Grade'].value_counts())
    print(f"\nNumber of teams: {num_teams}")
    print(f"- Team 0: Female students ({len(female_indices)})")
    print(f"- Team 1: 8th grade students ({len(eighth_grade_indices)})")
    print(f"- Teams 2-{num_teams-1}: Remaining students ({remaining_students})")
    print(f"Roles per team: {len(ROLE_COLUMNS)}")
    print(f"Target team size range: 4-6 students")
