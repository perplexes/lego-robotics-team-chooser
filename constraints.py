from ortools.sat.python import cp_model
from models import TeamData, ROLE_COLUMNS

def setup_model_variables(model: cp_model.CpModel, team_data: TeamData, total_teams: int):
    """Create and return the decision variables for the optimization model."""
    # student_team[i][t] = 1 if student i is in team t
    student_team = {}
    for i in range(team_data.num_students):
        for t in range(total_teams):
            student_team[i, t] = model.NewBoolVar(f'student_{i}_team_{t}')
    
    # student_role[i][r] = 1 if student i has role r
    student_role = {}
    for i in range(team_data.num_students):
        for r in ROLE_COLUMNS:
            student_role[i, r] = model.NewBoolVar(f'student_{i}_role_{r}')
            
    return student_team, student_role

def add_basic_constraints(model: cp_model.CpModel, team_data: TeamData, 
                         student_team: dict, student_role: dict, total_teams: int,
                         eighth_grade_mode: str = 'separate'):
    """Add basic assignment constraints to the model."""
    # Each student must be in exactly one team
    for i in range(team_data.num_students):
        model.Add(sum(student_team[i, t] for t in range(total_teams)) == 1)
    
    # Handle special teams
    # Put all females in team 0
    for i in team_data.female_indices:
        model.Add(student_team[i, 0] == 1)
    
    if eighth_grade_mode == 'separate':
        # Put all 8th graders in team 1
        for i in team_data.eighth_grade_indices:
            model.Add(student_team[i, 1] == 1)
        
        # Ensure non-special students cannot be in teams 0 and 1
        for i in range(team_data.num_students):
            if i not in team_data.female_indices and i not in team_data.eighth_grade_indices:
                model.Add(student_team[i, 0] == 0)
                model.Add(student_team[i, 1] == 0)
    else:  # distributed mode
        # Only ensure non-females cannot be in team 0
        for i in range(team_data.num_students):
            if i not in team_data.female_indices:
                model.Add(student_team[i, 0] == 0)
            
def add_team_size_constraints(model: cp_model.CpModel, team_data: TeamData, 
                            student_team: dict, total_teams: int,
                            eighth_grade_mode: str = 'separate'):
    """Add constraints for team sizes."""
    # For female team (team 0), enforce exact size
    team_size = sum(student_team[i, 0] for i in range(team_data.num_students))
    model.Add(team_size == team_data.config.special_team_size)
    
    if eighth_grade_mode == 'separate':
        # For 8th grade team (team 1), enforce exact size
        team_size = sum(student_team[i, 1] for i in range(team_data.num_students))
        model.Add(team_size == team_data.config.special_team_size)
        start_team = 2
    else:
        start_team = 1
    
    # For regular teams, enforce min/max size
    for t in range(start_team, total_teams):
        team_size = sum(student_team[i, t] for i in range(team_data.num_students))
        model.Add(team_size >= team_data.config.min_team_size)
        model.Add(team_size <= team_data.config.max_team_size)

def add_role_constraints(model: cp_model.CpModel, team_data: TeamData, 
                        student_team: dict, student_role: dict, total_teams: int):
    """Add constraints for role assignments."""
    # Each student can have 0-2 roles
    for i in range(team_data.num_students):
        num_roles = sum(student_role[i, r] for r in ROLE_COLUMNS)
        model.Add(num_roles <= 2)
    
    # Each role must be assigned exactly once per team
    for t in range(total_teams):
        for r in ROLE_COLUMNS:
            role_in_team = []
            for i in range(team_data.num_students):
                # student_role_team[i, r, t] = 1 if student i has role r in team t
                student_role_team = model.NewBoolVar(f'student_{i}_role_{r}_team_{t}')
                model.AddBoolAnd([student_team[i, t], student_role[i, r]]).OnlyEnforceIf(student_role_team)
                model.AddBoolOr([student_team[i, t].Not(), student_role[i, r].Not()]).OnlyEnforceIf(student_role_team.Not())
                role_in_team.append(student_role_team)
            
            # Each role must be assigned exactly once per team
            model.Add(sum(role_in_team) == 1)

def add_grade_constraints(model: cp_model.CpModel, team_data: TeamData, 
                         student_team: dict, total_teams: int,
                         eighth_grade_mode: str = 'separate'):
    """Add constraints for grade grouping."""
    # Grade grouping constraint: if a grade appears in a team, there must be at least 2 students of that grade
    # (except for the special teams 0 and 1 which have their own constraints)
    start_team = 2 if eighth_grade_mode == 'separate' else 1
    
    # In distributed mode, ensure at most one 8th grader per team (except team 0)
    if eighth_grade_mode == 'distributed':
        for t in range(1, total_teams):
            eighth_graders_in_team = []
            for i in team_data.eighth_grade_indices:
                eighth_graders_in_team.append(student_team[i, t])
            if eighth_graders_in_team:
                model.Add(sum(eighth_graders_in_team) <= 1)
    
    # Grade grouping for 6th and 7th grades
    for t in range(start_team, total_teams):  # Only for non-special teams
        for grade in ['6th', '7th']:  # Only handle 6th and 7th grades here
            # Count students of this grade in this team
            grade_members = []
            for i in range(team_data.num_students):
                if team_data.data.iloc[i]['Grade'] == grade:
                    grade_members.append(student_team[i, t])
            
            if grade_members:  # If there are any students of this grade
                # Either no students of this grade, or at least 2
                grade_present = model.NewBoolVar(f'grade_{grade}_team_{t}')
                
                # If grade is present, must have at least 2 students
                model.Add(sum(grade_members) >= 2).OnlyEnforceIf(grade_present)
                # If grade not present, must have 0 students
                model.Add(sum(grade_members) == 0).OnlyEnforceIf(grade_present.Not())
