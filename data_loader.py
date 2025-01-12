import pandas as pd
import math
from models import TeamData, OptimizationConfig, ROLE_COLUMNS

def load_team_data(data_path: str, eighth_grade_mode: str = 'separate') -> TeamData:
    """Load and process the student data from CSV."""
    data = pd.read_csv(data_path)
    num_students = len(data)
    
    # Get special group indices
    female_indices = data[data['Gender'] == 'F'].index.tolist()
    eighth_grade_indices = data[data['Grade'] == '8th'].index.tolist()
    
    config = OptimizationConfig()
    
    if eighth_grade_mode == 'separate':
        # Calculate remaining students after females and 8th graders
        remaining_students = num_students - len(female_indices) - len(eighth_grade_indices)
        # Need 2 special teams + teams for remaining students
        suggested_teams = 2 + math.ceil(remaining_students / config.max_team_size)
    else:  # distributed mode
        # Calculate remaining students after females only
        remaining_students = num_students - len(female_indices)
        # Need 1 special team + teams for remaining students (including 8th graders)
        suggested_teams = 1 + math.ceil(remaining_students / config.max_team_size)
    
    print_data_statistics(data, suggested_teams, female_indices, eighth_grade_indices, 
                         remaining_students, eighth_grade_mode)
    
    return TeamData(
        data=data,
        num_students=num_students,
        female_indices=female_indices,
        eighth_grade_indices=eighth_grade_indices,
        config=config
    )

def print_data_statistics(data: pd.DataFrame, suggested_teams: int, female_indices: list, 
                         eighth_grade_indices: list, remaining_students: int,
                         eighth_grade_mode: str = 'separate') -> None:
    """Print statistics about the loaded data."""
    print(f"Number of students: {len(data)}")
    print("\nGender distribution:")
    print(data['Gender'].value_counts())
    print("\nGrade distribution:")
    print(data['Grade'].value_counts())
    print(f"\nSuggested minimum teams: {suggested_teams}")
    print(f"- Team 0: Female students ({len(female_indices)})")
    if eighth_grade_mode == 'separate':
        print(f"- Team 1: 8th grade students ({len(eighth_grade_indices)})")
        print(f"- Teams 2+: Remaining students ({remaining_students})")
    else:
        print(f"- Teams 1+: Remaining students ({remaining_students}, including {len(eighth_grade_indices)} 8th graders)")
    print(f"Roles per team: {len(ROLE_COLUMNS)}")
