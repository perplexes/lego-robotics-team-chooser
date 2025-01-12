import pandas as pd
from ortools.sat.python import cp_model
from models import TeamData, OptimizationResult

def extract_solution(solver: cp_model.CpSolver, team_data: TeamData,
                    student_team: dict, student_role: dict, total_teams: int) -> OptimizationResult:
    """Extract the solution from the solver and format it for output."""
    results = []
    csv_results = []
    
    for i in range(team_data.num_students):
        team_id = next(t for t in range(total_teams) 
                      if solver.Value(student_team[i, t]) == 1)
        roles = [r for r in team_data.role_columns 
                if solver.Value(student_role[i, r]) == 1]
        
        # Original format for display
        student_data = {
            'student_id': team_data.data.iloc[i]['ID'],
            'name': f'Student {team_data.data.iloc[i]["ID"]}',
            'team': team_id,
            'roles': roles
        }
        results.append(student_data)
        
        # CSV format
        while len(roles) < 2:  # Pad with empty strings if student has only 1 role
            roles.append('')
        csv_data = {
            'student_id': team_data.data.iloc[i]['ID'],
            'team_id': team_id,
            'role_1': roles[0],
            'role_2': roles[1]
        }
        csv_results.append(csv_data)
    
    # Create DataFrames
    results_df = pd.DataFrame(results)
    csv_df = pd.DataFrame(csv_results)
    
    # Save CSV output
    csv_df.to_csv('team_assignments.csv', index=False)
    
    # Return OptimizationResult
    return OptimizationResult(
        student_assignments=results_df,
        objective_value=solver.ObjectiveValue(),
        status=get_solver_status(solver.StatusName())
    )

def get_solver_status(status_name: str) -> str:
    """Convert solver status to human-readable format."""
    status_map = {
        'OPTIMAL': 'Optimal solution found',
        'FEASIBLE': 'Feasible solution found',
        'INFEASIBLE': 'Problem is infeasible',
        'MODEL_INVALID': 'Model is invalid',
        'UNKNOWN': 'Unknown status'
    }
    return status_map.get(status_name, status_name)

def print_solution(result: OptimizationResult | None) -> None:
    """Print the optimization results in a readable format."""
    if result is None:
        print("\nNo solution found. The constraints may be too restrictive.")
        print("Try increasing the number of allowed teams or adjusting other constraints.")
        return
        
    if result.student_assignments is not None:
        print("\nOptimal team assignments found:")
        print(result.student_assignments)
        print("\nCSV output saved to team_assignments.csv")
        
        # Print team sizes
        print("\nTeam sizes:")
        team_sizes = result.student_assignments['team'].value_counts().sort_index()
        for team, size in team_sizes.items():
            print(f"Team {team}: {size} students")
    else:
        print("\nNo feasible solution found.")
        print("The solver could not find a valid assignment that satisfies all constraints.")
