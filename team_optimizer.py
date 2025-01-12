from ortools.sat.python import cp_model
import pandas as pd
import numpy as np

class TeamAssignmentOptimizer:
    def __init__(self, data_path, num_teams):
        self.data = pd.read_csv(data_path)
        self.num_teams = num_teams
        self.num_students = len(self.data)
        self.role_columns = ['team_captain', 'innovation_project_leader', 
                           'mission_strategist', 'public_relations_lead',
                           'lego_lead_builder', 'lead_coder']
        
    def _get_role_score(self, student_idx, role):
        return self.data.iloc[student_idx][role]
    
    def _get_grade_affinity(self, grade1, grade2):
        # Convert grades to numbers (6th=6, 7th=7, 8th=8)
        grade1_num = int(grade1.replace('th', ''))
        grade2_num = int(grade2.replace('th', ''))
        return 1.0 - abs(grade1_num - grade2_num) / 2  # Normalize to [0,1]
    
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
        
        # Gender constraint: females must be in the same team
        female_indices = self.data[self.data['Gender'] == 'F'].index.tolist()
        if len(female_indices) > 1:
            for i in range(len(female_indices)):
                for j in range(i + 1, len(female_indices)):
                    for t in range(self.num_teams):
                        model.AddImplication(student_team[female_indices[i], t], student_team[female_indices[j], t])
        
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
        
        # Grade affinity component
        for t in range(self.num_teams):
            for i in range(self.num_students):
                for j in range(i + 1, self.num_students):
                    grade_i = self.data.iloc[i]['Grade']
                    grade_j = self.data.iloc[j]['Grade']
                    affinity = self._get_grade_affinity(grade_i, grade_j)
                    
                    # students_together[i, j, t] = 1 if both students i and j are in team t
                    students_together = model.NewBoolVar(f'students_{i}_{j}_team_{t}')
                    model.AddBoolAnd([student_team[i, t], student_team[j, t]]).OnlyEnforceIf(students_together)
                    model.AddBoolOr([student_team[i, t].Not(), student_team[j, t].Not()]).OnlyEnforceIf(students_together.Not())
                    
                    grade_scores.append(affinity * students_together)
        
        # Combine objective components with weights
        ROLE_WEIGHT = 1
        GRADE_WEIGHT = 2
        
        objective_terms = (
            [ROLE_WEIGHT * score for score in role_scores] +
            [GRADE_WEIGHT * score for score in grade_scores]
        )
        
        model.Maximize(sum(objective_terms))
        
        # Solve
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            # Extract results
            results = []
            for i in range(self.num_students):
                student_data = {
                    'student_id': self.data.iloc[i]['ID'],
                    'name': self.data.iloc[i]['Name (first, last)'],
                    'team': next(t for t in range(self.num_teams) 
                               if solver.Value(student_team[i, t]) == 1),
                    'roles': [r for r in self.role_columns 
                            if solver.Value(student_role[i, r]) == 1]
                }
                results.append(student_data)
            
            return pd.DataFrame(results)
        else:
            return None

# Example usage:
if __name__ == "__main__":
    optimizer = TeamAssignmentOptimizer('student_data.csv', num_teams=3)
    results = optimizer.optimize()
    if results is not None:
        print("Optimal team assignments found:")
        print(results)
    else:
        print("No feasible solution found.")
