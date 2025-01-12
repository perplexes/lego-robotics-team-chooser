from ortools.sat.python import cp_model
from models import TeamData, ROLE_COLUMNS

def setup_model_variables(model: cp_model.CpModel, team_data: TeamData):
    """Create and return the decision variables for the optimization model."""
    # student_team[i][t] = 1 if student i is in team t
    student_team = {}
    for i in range(team_data.num_students):
        for t in range(team_data.num_teams):
            student_team[i, t] = model.NewBoolVar(f'student_{i}_team_{t}')
    
    # student_role[i][r] = 1 if student i has role r
    student_role = {}
    for i in range(team_data.num_students):
        for r in ROLE_COLUMNS:
            student_role[i, r] = model.NewBoolVar(f'student_{i}_role_{r}')
            
    return student_team, student_role

def add_basic_constraints(model: cp_model.CpModel, team_data: TeamData, 
                         student_team: dict, student_role: dict,
                         total_teams_constraint: int = None):
    """Add basic assignment constraints to the model."""
    # Each student must be in exactly one team
    for i in range(team_data.num_students):
        model.Add(sum(student_team[i, t] for t in range(team_data.num_teams)) == 1)
    
    # Handle special teams
    # Put all females in team 0
    for i in team_data.female_indices:
        model.Add(student_team[i, 0] == 1)
    
    # Put all 8th graders in team 1
    for i in team_data.eighth_grade_indices:
        model.Add(student_team[i, 1] == 1)
    
    # Ensure non-special students cannot be in teams 0 and 1
    for i in range(team_data.num_students):
        if i not in team_data.female_indices and i not in team_data.eighth_grade_indices:
            model.Add(student_team[i, 0] == 0)
            model.Add(student_team[i, 1] == 0)
            
    # If total_teams_constraint is specified, ensure no students are assigned beyond that limit
    if total_teams_constraint is not None:
        for i in range(team_data.num_students):
            model.Add(sum(student_team[i, t] for t in range(total_teams_constraint, team_data.num_teams)) == 0)

def add_team_size_constraints(model: cp_model.CpModel, team_data: TeamData, 
                            student_team: dict):
    """Add constraints for team sizes."""
    for t in range(team_data.num_teams):
        team_size = sum(student_team[i, t] for i in range(team_data.num_students))
        if t <= 1:  # Special teams (females and 8th graders)
            model.Add(team_size == team_data.config.special_team_size)
        else:  # Other teams
            model.Add(team_size >= team_data.config.min_team_size)
            model.Add(team_size <= team_data.config.max_team_size)

def add_role_constraints(model: cp_model.CpModel, team_data: TeamData, 
                        student_team: dict, student_role: dict):
    """Add constraints for role assignments."""
    # Each student must have 1 or 2 roles
    for i in range(team_data.num_students):
        model.Add(sum(student_role[i, r] for r in ROLE_COLUMNS) >= 
                 team_data.config.min_roles_per_student)
        model.Add(sum(student_role[i, r] for r in ROLE_COLUMNS) <= 
                 team_data.config.max_roles_per_student)
    
    # Each role must be assigned exactly once per team
    for t in range(team_data.num_teams):
        for r in ROLE_COLUMNS:
            role_in_team = []
            for i in range(team_data.num_students):
                # student_role_team[i, r, t] = 1 if student i has role r in team t
                student_role_team = model.NewBoolVar(f'student_{i}_role_{r}_team_{t}')
                model.AddBoolAnd([student_team[i, t], student_role[i, r]]).OnlyEnforceIf(student_role_team)
                model.AddBoolOr([student_team[i, t].Not(), student_role[i, r].Not()]).OnlyEnforceIf(student_role_team.Not())
                role_in_team.append(student_role_team)
            model.Add(sum(role_in_team) == 1)

def add_grade_constraints(model: cp_model.CpModel, team_data: TeamData, 
                         student_team: dict):
    """Add constraints for grade grouping."""
    # Grade grouping constraint: if a grade appears in a team, there must be at least 2 students of that grade
    # (except for the special teams 0 and 1 which have their own constraints)
    for t in range(2, team_data.num_teams):  # Only for non-special teams
        for grade in ['6th', '7th', '8th']:
            # Count students of this grade in this team
            grade_members = []
            for i in range(team_data.num_students):
                if team_data.data.iloc[i]['Grade'] == grade:
                    grade_members.append(student_team[i, t])
            
            if grade_members:  # If there are any students of this grade
                # Either no students of this grade, or at least 2
                grade_present = model.NewBoolVar(f'grade_{grade}_team_{t}')
                
                # grade_present is true if any student of this grade is in the team
                model.Add(sum(grade_members) >= 1).OnlyEnforceIf(grade_present)
                model.Add(sum(grade_members) == 0).OnlyEnforceIf(grade_present.Not())
                
                # If grade is present, must have at least 2 students
                model.Add(sum(grade_members) >= 2).OnlyEnforceIf(grade_present)
