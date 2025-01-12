import pytest
import pandas as pd
from team_optimizer import TeamAssignmentOptimizer

@pytest.fixture
def sample_data():
    """Create a small sample dataset for testing"""
    data = {
        'ID': [1, 2, 3, 4, 5, 6],
        'Name (first, last)': ['Student 1', 'Student 2', 'Student 3', 
                              'Student 4', 'Student 5', 'Student 6'],
        'Gender': ['M', 'F', 'M', 'F', 'M', 'M'],
        'Grade': ['6th', '7th', '7th', '7th', '8th', '8th'],
        'team_captain': [3, 2, 1, 2, 3, 1],
        'innovation_project_leader': [2, 3, 2, 1, 2, 3],
        'mission_strategist': [1, 2, 3, 2, 1, 2],
        'public_relations_lead': [2, 1, 2, 3, 2, 1],
        'lego_lead_builder': [3, 2, 1, 2, 3, 2],
        'lead_coder': [1, 2, 3, 2, 1, 3]
    }
    return pd.DataFrame(data)

def test_optimizer_initialization(tmp_path, sample_data):
    """Test that the optimizer initializes correctly"""
    # Save sample data to temporary CSV
    csv_path = tmp_path / "test_data.csv"
    sample_data.to_csv(csv_path, index=False)
    
    optimizer = TeamAssignmentOptimizer(csv_path, num_teams=2)
    assert optimizer.num_students == 6
    assert optimizer.num_teams == 2
    assert len(optimizer.role_columns) == 6

def test_grade_affinity_calculation(tmp_path, sample_data):
    """Test grade affinity calculation"""
    csv_path = tmp_path / "test_data.csv"
    sample_data.to_csv(csv_path, index=False)
    
    optimizer = TeamAssignmentOptimizer(csv_path, num_teams=2)
    
    # Same grade should have maximum affinity
    assert optimizer._get_grade_affinity('7th', '7th') == 1.0
    
    # Adjacent grades should have medium affinity
    assert optimizer._get_grade_affinity('7th', '8th') == 0.5
    
    # Two grades apart should have minimum affinity
    assert optimizer._get_grade_affinity('6th', '8th') == 0.0

def test_optimization_results(tmp_path, sample_data):
    """Test that optimization produces valid results"""
    csv_path = tmp_path / "test_data.csv"
    sample_data.to_csv(csv_path, index=False)
    
    optimizer = TeamAssignmentOptimizer(csv_path, num_teams=2)
    results = optimizer.optimize()
    
    assert results is not None
    assert len(results) == 6  # All students assigned
    
    # Check that each student is assigned to exactly one team
    team_counts = results['team'].value_counts()
    assert len(team_counts) == 2  # Two teams
    
    # Check that females are in the same team
    female_teams = results[sample_data['Gender'] == 'F']['team']
    assert len(female_teams.unique()) == 1
    
    # Check that each student has 1-2 roles
    role_counts = results['roles'].apply(len)
    assert all(role_counts >= 1) and all(role_counts <= 2)

def test_invalid_input(tmp_path):
    """Test handling of invalid input data"""
    # Create empty DataFrame
    empty_df = pd.DataFrame()
    csv_path = tmp_path / "empty.csv"
    empty_df.to_csv(csv_path, index=False)
    
    with pytest.raises(Exception):
        TeamAssignmentOptimizer(csv_path, num_teams=2)

if __name__ == '__main__':
    pytest.main([__file__])