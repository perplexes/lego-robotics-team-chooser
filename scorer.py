from ortools.sat.python import cp_model
from models import TeamData
from validator import get_role_score

def setup_objective_function(model: cp_model.CpModel, team_data: TeamData,
                           student_team: dict, student_role: dict) -> None:
    """
    Set up the optimization objective function.
    Combines role score matches and penalties for 6th graders with multiple roles.
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
    
    # Penalty for 6th graders having multiple roles
    sixth_grade_penalties = []
    for i in range(team_data.num_students):
        if team_data.data.iloc[i]['Grade'] == '6th':
            # Count number of roles for this 6th grader
            num_roles = sum(student_role[i, r] for r in team_data.role_columns)
            # Add penalty for each role beyond the first
            sixth_grade_penalties.append(
                team_data.config.sixth_grade_multi_role_penalty * (num_roles - 1)
            )
    
    # Combine all objective components
    objective_terms = role_scores + sixth_grade_penalties
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
    
    def set_parameters(self, student_team: dict, student_role: dict, team_data: TeamData):
        """Set the parameters needed for solution printing."""
        self._student_team = student_team
        self._student_role = student_role
        self._team_data = team_data
    
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
                'team': next(t for t in range(self._team_data.num_teams) 
                           if self.Value(self._student_team[i, t]) == 1),
                'roles': [r for r in self._team_data.role_columns 
                         if self.Value(self._student_role[i, r]) == 1]
            }
            results.append(student_data)
        
        # Print team compositions
        import pandas as pd
        df = pd.DataFrame(results)
        for t in range(self._team_data.num_teams):
            team_df = df[df['team'] == t]
            print(f"\nTeam {t} ({len(team_df)} students):")
            for _, student in team_df.iterrows():
                print(f"  Student {student['student_id']} "
                      f"({student['gender']}, {student['grade']}): "
                      f"{', '.join(student['roles'])}")
        
        print("\n" + "="*50)  # Separator between solutions
        self._solution_count += 1
