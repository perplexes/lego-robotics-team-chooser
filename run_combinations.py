#!/usr/bin/env python3
from team_optimizer import TeamAssignmentOptimizer
from output import print_solution

def main():
    """Run optimizer with different combinations of teams and 8th grade modes."""
    team_counts = [4, 5]
    modes = ['separate', 'distributed']
    data_path = 'anonymized_student_data.csv'
    
    print("Running optimizer with all combinations:")
    print("----------------------------------------")
    
    for team_count in team_counts:
        for mode in modes:
            # Create descriptive output filename
            output_file = f'teams_{team_count}_{mode}.csv'
            
            print(f"\nRunning with {team_count} teams in {mode} mode")
            print(f"Output will be saved to: {output_file}")
            
            # Initialize and run optimizer
            optimizer = TeamAssignmentOptimizer(
                data_path=data_path,
                total_teams=team_count,
                eighth_grade_mode=mode,
                output_file=output_file
            )
            
            # Run optimization
            result = optimizer.optimize()
            
            # Print results
            print_solution(result, output_file)
            print("\n" + "="*50)

if __name__ == "__main__":
    main()
