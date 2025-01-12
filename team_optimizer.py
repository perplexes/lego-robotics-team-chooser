from ortools.sat.python import cp_model
import pandas as pd
import numpy as np
import time
import math

class TeamAssignmentOptimizer:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.num_students = len(self.data)
        self.role_columns = ['team_captain', 'innovation_project_leader', 
                           'mission_strategist', 'public_relations_lead',
                           'lego_lead_builder', 'lead_coder']
        
        # Get special group indices
        self.female_indices = self.data[self.data['Gender'] == 'F'].index.tolist()
        self.eighth_grade_indices = self.data[self.data['Grade'] == '8th'].index.tolist()
        
        # Calculate number of teams needed
        remaining_students = self.num_students - len(self.female_indices) - len(self.eighth_grade_indices)
        additional_teams = math.ceil(remaining_students / 6)  # At most 6 students per team
        self.num_teams = 2 + additional_teams  # 2 special teams + additional teams
        
        # Print data statistics
        print(f"Number of students: {self.num_students}")
        print("\nGender distribution:")
        print(self.data['Gender'].value_counts())
        print("\nGrade distribution:")
        print(self.data['Grade'].value_counts())
        print(f"\nNumber of teams: {self.num_teams}")
        print(f"- Team 0: Female students ({len(self.female_indices)})")
        print(f"- Team 1: 8th grade students ({len(self.eighth_grade_indices)})")
        print(f"- Teams 2-{self.num_teams-1}: Remaining students ({remaining_students})")
        print(f"Roles per team: {len(self.role_columns)}")
        print(f"Target team size range: 4-6 students")
        
        # Run preflight checks
        self._validate_data()
        
    def _validate_data(self):
        """
        Performs preflight checks on the input data to ensure it meets all requirements.
        Raises ValueError with detailed message if any check fails.
        """
        errors = []
        
        # Check required columns exist
        required_columns = {
            'ID': 'integer',
            'Gender': 'category',
            'Grade': 'grade',
            **{role: 'score' for role in self.role_columns}
        }
        
        for col, dtype in required_columns.items():
            if col not in self.data.columns:
                errors.append(f"Missing required column: {col}")
                continue
                
            # Check for null values
            null_count = self.data[col].isnull().sum()
            if null_count > 0:
                errors.append(f"Found {null_count} null values in column: {col}")
            
            # Type-specific validation
            if dtype == 'integer':
                if not all(isinstance(x, (int, float)) and pd.notnull(x) for x in self.data[col]):
                    errors.append(f"Column {col} must contain valid integers")
                    
            elif dtype == 'string':
                if not all(isinstance(x, str) and len(x.strip()) > 0 for x in self.data[col] if pd.notnull(x)):
                    errors.append(f"Column {col} must contain non-empty strings")
                    
            elif dtype == 'category':
                valid_genders = {'M', 'F'}
                invalid_genders = set(self.data[col].unique()) - valid_genders
                if invalid_genders:
                    errors.append(f"Invalid gender values found: {invalid_genders}. Must be one of: {valid_genders}")
                    
            elif dtype == 'grade':
                valid_grades = {6, 7, 8, '6th', '7th', '8th', 6.0, 7.0, 8.0}
                try:
                    for grade in self.data[col]:
                        if isinstance(grade, str):
                            grade_num = int(grade.replace('th', ''))
                            if grade_num not in {6, 7, 8}:
                                errors.append(f"Invalid grade value: {grade}")
                        elif isinstance(grade, (int, float)):
                            if int(grade) not in {6, 7, 8}:
                                errors.append(f"Invalid grade value: {grade}")
                        else:
                            errors.append(f"Invalid grade format: {grade}")
                except (ValueError, AttributeError) as e:
                    errors.append(f"Error processing grade values: {str(e)}")
                    
            elif dtype == 'score':
                try:
                    scores = pd.to_numeric(self.data[col])
                    if not all(1 <= score <= 3 for score in scores if pd.notnull(score)):
                        errors.append(f"Role scores in {col} must be between 1 and 3")
                except ValueError:
                    errors.append(f"Invalid score values in column: {col}")
        
        # If any errors were found, raise ValueError with all error messages
        if errors:
            raise ValueError("Data validation failed:\n" + "\n".join(f"- {error}" for error in errors))
            
    def _get_role_score(self, student_idx, role):
        return self.data.iloc[student_idx][role]
    
    def _get_grade_affinity(self, grade1, grade2):
        # Helper function to convert grade to number
        def grade_to_num(grade):
            if isinstance(grade, (int, float)):
                return int(grade)
            return int(str(grade).replace('th', ''))
        
        # Convert grades to numbers
        grade1_num = grade_to_num(grade1)
        grade2_num = grade_to_num(grade2)
        
        # Return affinity score:
        # 3 for same grade (strongly encourage same-grade grouping)
        # 0 for 7th graders with different grades (discourage isolating 7th graders)
        # 1 for other different grades
        grade_diff = abs(grade1_num - grade2_num)
        if grade_diff == 0:
            return 3
        elif grade1_num == 7 or grade2_num == 7:
            return 0  # Strongly discourage mixing lone 7th graders
        else:
            return 1
    
    def optimize(self):
        model = cp_model.CpModel()
        
        # Decision Variables
        # student_team[i][t] = 1 if student i is in team t
        student_team = {}
        for i in range(self.num_students):
            for t in range(self.num_teams):
                student_team[i, t] = model.NewBoolVar(f'student_{i}_team_{t}')
        
        # student_role[i][r] = 1 if student i has role r
        student_role = {}
        for i in range(self.num_students):
            for r in self.role_columns:
                student_role[i, r] = model.NewBoolVar(f'student_{i}_role_{r}')
        
        # Constraints
        # Each student must be in exactly one team
        for i in range(self.num_students):
            model.Add(sum(student_team[i, t] for t in range(self.num_teams)) == 1)
        
        # Handle special teams
        # Put all females in team 0
        for i in self.female_indices:
            model.Add(student_team[i, 0] == 1)
        
        # Put all 8th graders in team 1
        for i in self.eighth_grade_indices:
            model.Add(student_team[i, 1] == 1)
        
        # Ensure non-special students cannot be in teams 0 and 1
        for i in range(self.num_students):
            if i not in self.female_indices and i not in self.eighth_grade_indices:
                model.Add(student_team[i, 0] == 0)
                model.Add(student_team[i, 1] == 0)
        
        # Each team must have appropriate size
        for t in range(self.num_teams):
            team_size = sum(student_team[i, t] for i in range(self.num_students))
            if t <= 1:  # Special teams (females and 8th graders)
                model.Add(team_size == 4)
            else:  # Other teams
                model.Add(team_size >= 4)
                model.Add(team_size <= 6)
        
        # Each student must have 1 or 2 roles
        for i in range(self.num_students):
            model.Add(sum(student_role[i, r] for r in self.role_columns) >= 1)
            model.Add(sum(student_role[i, r] for r in self.role_columns) <= 2)
        
        # Each role must be assigned exactly once per team
        for t in range(self.num_teams):
            for r in self.role_columns:
                role_in_team = []
                for i in range(self.num_students):
                    # student_role_team[i, r, t] = 1 if student i has role r in team t
                    student_role_team = model.NewBoolVar(f'student_{i}_role_{r}_team_{t}')
                    model.AddBoolAnd([student_team[i, t], student_role[i, r]]).OnlyEnforceIf(student_role_team)
                    model.AddBoolOr([student_team[i, t].Not(), student_role[i, r].Not()]).OnlyEnforceIf(student_role_team.Not())
                    role_in_team.append(student_role_team)
                model.Add(sum(role_in_team) == 1)
        
        # Objective function components
        role_scores = []
        grade_scores = []
        
        # Role score component
        for i in range(self.num_students):
            student_role_score = []
            for r in self.role_columns:
                score = self._get_role_score(i, r)
                role_var = student_role[i, r]
                student_role_score.append(score * role_var)
            role_scores.extend(student_role_score)
        
        # Grade grouping constraint: if a grade appears in a team, there must be at least 2 students of that grade
        # (except for the special teams 0 and 1 which have their own constraints)
        for t in range(2, self.num_teams):  # Only for non-special teams
            for grade in ['6th', '7th', '8th']:
                # Count students of this grade in this team
                grade_members = []
                for i in range(self.num_students):
                    if self.data.iloc[i]['Grade'] == grade:
                        grade_members.append(student_team[i, t])
                
                if grade_members:  # If there are any students of this grade
                    # Either no students of this grade, or at least 2
                    grade_present = model.NewBoolVar(f'grade_{grade}_team_{t}')
                    
                    # grade_present is true if any student of this grade is in the team
                    model.Add(sum(grade_members) >= 1).OnlyEnforceIf(grade_present)
                    model.Add(sum(grade_members) == 0).OnlyEnforceIf(grade_present.Not())
                    
                    # If grade is present, must have at least 2 students
                    model.Add(sum(grade_members) >= 2).OnlyEnforceIf(grade_present)
        
        # Objective function components
        # 1. Role score matches
        objective_terms = role_scores
        
        # 2. Penalty for 6th graders having multiple roles
        for i in range(self.num_students):
            if self.data.iloc[i]['Grade'] == '6th':
                # Count number of roles for this 6th grader
                num_roles = sum(student_role[i, r] for r in self.role_columns)
                # Add penalty (-5) for each role beyond the first
                objective_terms.append(-5 * (num_roles - 1))
        
        model.Maximize(sum(objective_terms))
        
        # Solve with progress logging
        solver = cp_model.CpSolver()
        
        class SolutionPrinter(cp_model.CpSolverSolutionCallback):
            def __init__(self):
                cp_model.CpSolverSolutionCallback.__init__(self)
                self._solution_count = 0
                self._start_time = time.time()
                self._student_team = None
                self._student_role = None
                self._num_students = None
                self._num_teams = None
                self._data = None
                self._role_columns = None
            
            def set_parameters(self, student_team, student_role, num_students, num_teams, data, role_columns):
                self._student_team = student_team
                self._student_role = student_role
                self._num_students = num_students
                self._num_teams = num_teams
                self._data = data
                self._role_columns = role_columns
            
            def on_solution_callback(self):
                current_time = time.time()
                print(f'\nFound solution {self._solution_count + 1} '
                      f'(obj={self.ObjectiveValue():.0f}) '
                      f'time={current_time - self._start_time:.1f}s')
                
                # Extract and print current solution
                results = []
                for i in range(self._num_students):
                    student_data = {
                        'student_id': self._data.iloc[i]['ID'],
                        'name': f'Student {self._data.iloc[i]["ID"]}',
                        'gender': self._data.iloc[i]['Gender'],
                        'grade': self._data.iloc[i]['Grade'],
                        'team': next(t for t in range(self._num_teams) 
                                   if self.Value(self._student_team[i, t]) == 1),
                        'roles': [r for r in self._role_columns 
                                if self.Value(self._student_role[i, r]) == 1]
                    }
                    results.append(student_data)
                
                df = pd.DataFrame(results)
                
                # Print team compositions
                for t in range(self._num_teams):
                    team_df = df[df['team'] == t]
                    print(f"\nTeam {t} ({len(team_df)} students):")
                    for _, student in team_df.iterrows():
                        print(f"  Student {student['student_id']} "
                              f"({student['gender']}, {student['grade']}): "
                              f"{', '.join(student['roles'])}")
                
                print("\n" + "="*50)  # Separator between solutions
                self._solution_count += 1
        
        solution_printer = SolutionPrinter()
        solution_printer.set_parameters(student_team, student_role, 
                                     self.num_students, self.num_teams,
                                     self.data, self.role_columns)
        print("\nSolving...")
        solver.parameters.max_time_in_seconds = 60.0  # Set 60 second time limit
        status = solver.Solve(model, solution_printer)
        print("\n")  # New line after progress output
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            # Extract results
            results = []
            csv_results = []
            for i in range(self.num_students):
                team_id = next(t for t in range(self.num_teams) 
                           if solver.Value(student_team[i, t]) == 1)
                roles = [r for r in self.role_columns 
                        if solver.Value(student_role[i, r]) == 1]
                
                # Original format for display
                student_data = {
                    'student_id': self.data.iloc[i]['ID'],
                    'name': f'Student {self.data.iloc[i]["ID"]}',
                    'team': team_id,
                    'roles': roles
                }
                results.append(student_data)
                
                # CSV format
                while len(roles) < 2:
                    roles.append('')
                csv_data = {
                    'student_id': self.data.iloc[i]['ID'],
                    'team_id': team_id,
                    'role_1': roles[0],
                    'role_2': roles[1]
                }
                csv_results.append(csv_data)
            
            # Save CSV format
            csv_df = pd.DataFrame(csv_results)
            csv_df.to_csv('team_assignments.csv', index=False)
            
            # Return original format for display
            return pd.DataFrame(results)
        else:
            return None

# Example usage:
if __name__ == "__main__":
    optimizer = TeamAssignmentOptimizer('anonymized_student_data.csv')
    results = optimizer.optimize()
    if results is not None:
        print("\nOptimal team assignments found:")
        print(results)
        print("\nCSV output saved to team_assignments.csv")
        
        # Print team sizes
        print("\nTeam sizes:")
        team_sizes = results['team'].value_counts().sort_index()
        for team, size in team_sizes.items():
            print(f"Team {team}: {size} students")
    else:
        print("No feasible solution found.")
