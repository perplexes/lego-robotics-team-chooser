from ortools.sat.python import cp_model
from models import TeamData
from validator import get_role_score

def setup_objective_function(model: cp_model.CpModel, team_data: TeamData,
                           student_team: dict, student_role: dict, total_teams: int) -> None:
    """
    Set up the optimization objective function.
    Combines:
    1. Role score matches
    2. Penalties for 6th graders with multiple roles
    3. Penalties for uneven team sizes
    """
    # Role score component
    role_scores = []
    for i in range(team_data.num_students):
        student_role_score = []
        for r in team_data.role_columns:
            score = get_role_score(team_data.data, i, r)
            role_var = student_role[i, r]
            student_role_score.append(score * role_var)
        role_scores.extend(student_role_score)
    
    # Role count scoring - prefer one role, penalize zero or two roles
    role_count_scores = []
    for i in range(team_data.num_students):
        num_roles = sum(student_role[i, r] for r in team_data.role_columns)
        is_sixth_grade = team_data.data.iloc[i]['Grade'] == '6th'
        
        # Create boolean variables for role counts
        has_zero_roles = model.NewBoolVar(f'student_{i}_has_zero_roles')
        has_one_role = model.NewBoolVar(f'student_{i}_has_one_role')
        has_two_roles = model.NewBoolVar(f'student_{i}_has_two_roles')
        
        # Set conditions
        model.Add(num_roles == 0).OnlyEnforceIf(has_zero_roles)
        model.Add(num_roles > 0).OnlyEnforceIf(has_zero_roles.Not())
        
        model.Add(num_roles == 1).OnlyEnforceIf(has_one_role)
        model.Add(num_roles != 1).OnlyEnforceIf(has_one_role.Not())
        
        model.Add(num_roles == 2).OnlyEnforceIf(has_two_roles)
        model.Add(num_roles != 2).OnlyEnforceIf(has_two_roles.Not())
        
        # Apply scores based on grade
        if is_sixth_grade:
            # Higher penalties for 6th graders
            role_count_scores.extend([
                50 * has_one_role,    # High reward for one role
                -100 * has_zero_roles,  # Very high penalty for no roles
                -80 * has_two_roles     # High penalty for two roles
            ])
        else:
            # Regular penalties for other grades
            role_count_scores.extend([
                30 * has_one_role,    # Reward for one role
                -50 * has_zero_roles,  # Penalty for no roles
                -20 * has_two_roles    # Small penalty for two roles
            ])
    
    # Penalty for uneven team sizes (excluding special teams)
    team_size_penalties = []
    if total_teams > 2:  # Only if we have regular teams
        # Calculate team sizes
        team_sizes = []
        for t in range(2, total_teams):  # Skip special teams
            team_size = sum(student_team[i, t] for i in range(team_data.num_students))
            team_sizes.append(team_size)
        
        # Calculate target size (average of min and max allowed)
        target_size = (team_data.config.min_team_size + team_data.config.max_team_size) // 2
        
        # Add penalty for each student above/below target size
        # Using a higher penalty weight to make this a priority
        size_penalty_weight = -10  # Negative because we want to minimize deviation
        for size in team_sizes:
            deviation = model.NewIntVar(0, team_data.config.max_team_size, 'deviation')
            # Set deviation to |size - target_size|
            model.AddAbsEquality(deviation, size - target_size)
            team_size_penalties.append(size_penalty_weight * deviation)
    
    # Role preference conflict penalties - simpler version since all teams must meet min size
    role_conflict_penalties = []
    conflict_penalty_weight = -5  # Small penalty for role preference conflicts
    
    # For each team and role, penalize having multiple students who strongly wanted that role
    for t in range(total_teams):
        for r in team_data.role_columns:
            # Count students in this team who had strong preference (score 3) for this role
            strong_preferences = []
            for i in range(team_data.num_students):
                if get_role_score(team_data.data, i, r) == 3:
                    strong_preferences.append(student_team[i, t])
            
            if strong_preferences:
                # Penalize having multiple students who wanted this role
                role_conflict_penalties.append(
                    conflict_penalty_weight * (sum(strong_preferences) - 1)
                )
    
    # Combine all objective components
    objective_terms = (role_scores + role_count_scores + 
                      team_size_penalties + role_conflict_penalties)
    model.Maximize(sum(objective_terms))

class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Callback to print intermediate solutions during optimization."""
    
    def __init__(self):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._solution_count = 0
        self._start_time = None
        self._student_team = None
        self._student_role = None
        self._team_data = None
    
    def set_parameters(self, student_team: dict, student_role: dict, team_data: TeamData, total_teams: int):
        """Set the parameters needed for solution printing."""
        self._student_team = student_team
        self._student_role = student_role
        self._team_data = team_data
        self._total_teams = total_teams
    
    def on_solution_callback(self):
        """Called for each intermediate solution found."""
        import time
        if self._start_time is None:
            self._start_time = time.time()
        
        current_time = time.time()
        print(f'\nFound solution {self._solution_count + 1} '
              f'(obj={self.ObjectiveValue():.0f}) '
              f'time={current_time - self._start_time:.1f}s')
        
        # Extract and print current solution
        results = []
        for i in range(self._team_data.num_students):
            student_data = {
                'student_id': self._team_data.data.iloc[i]['ID'],
                'name': f'Student {self._team_data.data.iloc[i]["ID"]}',
                'gender': self._team_data.data.iloc[i]['Gender'],
                'grade': self._team_data.data.iloc[i]['Grade'],
                'team': next(t for t in range(self._total_teams) 
                           if self.Value(self._student_team[i, t]) == 1),
                'roles': [r for r in self._team_data.role_columns 
                         if self.Value(self._student_role[i, r]) == 1]
            }
            results.append(student_data)
        
        # Print team compositions
        import pandas as pd
        df = pd.DataFrame(results)
        for t in range(self._total_teams):
            team_df = df[df['team'] == t]
            print(f"\nTeam {t} ({len(team_df)} students):")
            for _, student in team_df.iterrows():
                print(f"  Student {student['student_id']} "
                      f"({student['gender']}, {student['grade']}): "
                      f"{', '.join(student['roles'])}")
        
        print("\n" + "="*50)  # Separator between solutions
        self._solution_count += 1
