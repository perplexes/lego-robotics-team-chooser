from ortools.sat.python import cp_model
import time
import argparse

from models import TeamData
from data_loader import load_team_data
from validator import validate_data
from constraints import (
    setup_model_variables,
    add_basic_constraints,
    add_team_size_constraints,
    add_role_constraints,
    add_grade_constraints
)
from scorer import setup_objective_function, SolutionPrinter
from output import extract_solution, print_solution

class TeamAssignmentOptimizer:
    def __init__(self, data_path: str, total_teams: int = None):
        """Initialize the optimizer with student data."""
        # Load and validate data
        self.team_data = load_team_data(data_path)
        validate_data(self.team_data.data)
        
        # Store total_teams constraint if specified
        self.total_teams_constraint = None
        if total_teams is not None:
            # Ensure we have at least 2 teams for special groups
            self.total_teams_constraint = max(total_teams, 2)
        
    def optimize(self):
        """Run the optimization process to find optimal team assignments."""
        # Create the model
        model = cp_model.CpModel()
        
        # Set up variables
        student_team, student_role = setup_model_variables(model, self.team_data)
        
        # Add constraints
        add_basic_constraints(model, self.team_data, student_team, student_role, self.total_teams_constraint)
        add_team_size_constraints(model, self.team_data, student_team)
        add_role_constraints(model, self.team_data, student_team, student_role)
        add_grade_constraints(model, self.team_data, student_team)
        
        # Set up objective function
        setup_objective_function(model, self.team_data, student_team, student_role)
        
        # Create solver and solution printer
        solver = cp_model.CpSolver()
        solution_printer = SolutionPrinter()
        solution_printer.set_parameters(student_team, student_role, self.team_data)
        
        # Solve with time limit
        print("\nSolving...")
        solver.parameters.max_time_in_seconds = 60.0
        status = solver.Solve(model, solution_printer)
        print("\n")  # New line after progress output
        
        # Extract and return results if solution found
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            return extract_solution(solver, self.team_data, student_team, student_role)
        
        # Print detailed error for infeasible cases
        if status == cp_model.INFEASIBLE:
            print("\nError: No valid solution exists with the current constraints.")
            if self.total_teams_constraint is not None:
                print(f"The specified total teams limit ({self.total_teams_constraint}) may be too restrictive.")
                print(f"Try increasing --total-teams or removing the constraint.")
            else:
                print("Try adjusting the team size or other constraints.")
        elif status == cp_model.MODEL_INVALID:
            print("\nError: The optimization model is invalid.")
        else:
            print(f"\nError: Solver failed with status: {solver.StatusName()}")
        
        return None

def main():
    """Main entry point for the team assignment optimizer."""
    parser = argparse.ArgumentParser(description='Team Assignment Optimizer')
    parser.add_argument('--total-teams', type=int, help='Total number of teams to create (must be at least 2 for special teams)')
    parser.add_argument('--data-path', default='anonymized_student_data.csv', help='Path to student data CSV file')
    args = parser.parse_args()

    optimizer = TeamAssignmentOptimizer(args.data_path, args.total_teams)
    result = optimizer.optimize()
    print_solution(result)

if __name__ == "__main__":
    main()
