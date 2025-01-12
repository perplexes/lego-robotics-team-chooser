from typing import List, Dict
import pandas as pd
from models import ROLE_COLUMNS

def validate_data(data: pd.DataFrame) -> None:
    """
    Validates the input data meets all requirements.
    Raises ValueError with detailed message if any check fails.
    """
    errors = []
    
    required_columns = {
        'ID': 'integer',
        'Gender': 'category',
        'Grade': 'grade',
        **{role: 'score' for role in ROLE_COLUMNS}
    }
    
    for col, dtype in required_columns.items():
        if col not in data.columns:
            errors.append(f"Missing required column: {col}")
            continue
            
        # Check for null values
        null_count = data[col].isnull().sum()
        if null_count > 0:
            errors.append(f"Found {null_count} null values in column: {col}")
        
        # Type-specific validation
        if dtype == 'integer':
            if not all(isinstance(x, (int, float)) and pd.notnull(x) for x in data[col]):
                errors.append(f"Column {col} must contain valid integers")
                
        elif dtype == 'string':
            if not all(isinstance(x, str) and len(x.strip()) > 0 for x in data[col] if pd.notnull(x)):
                errors.append(f"Column {col} must contain non-empty strings")
                
        elif dtype == 'category':
            valid_genders = {'M', 'F'}
            invalid_genders = set(data[col].unique()) - valid_genders
            if invalid_genders:
                errors.append(f"Invalid gender values found: {invalid_genders}. Must be one of: {valid_genders}")
                
        elif dtype == 'grade':
            valid_grades = {6, 7, 8, '6th', '7th', '8th', 6.0, 7.0, 8.0}
            try:
                for grade in data[col]:
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
                scores = pd.to_numeric(data[col])
                if not all(1 <= score <= 3 for score in scores if pd.notnull(score)):
                    errors.append(f"Role scores in {col} must be between 1 and 3")
            except ValueError:
                errors.append(f"Invalid score values in column: {col}")
    
    # If any errors were found, raise ValueError with all error messages
    if errors:
        raise ValueError("Data validation failed:\n" + "\n".join(f"- {error}" for error in errors))

def get_role_score(data: pd.DataFrame, student_idx: int, role: str) -> float:
    """Get the role score for a specific student and role."""
    return data.iloc[student_idx][role]

def get_grade_affinity(grade1: str, grade2: str) -> int:
    """
    Calculate grade affinity score between two students.
    Returns:
    - 3 for same grade (strongly encourage same-grade grouping)
    - 0 for 7th graders with different grades (discourage isolating 7th graders)
    - 1 for other different grades
    """
    def grade_to_num(grade):
        if isinstance(grade, (int, float)):
            return int(grade)
        return int(str(grade).replace('th', ''))
    
    grade1_num = grade_to_num(grade1)
    grade2_num = grade_to_num(grade2)
    
    grade_diff = abs(grade1_num - grade2_num)
    if grade_diff == 0:
        return 3
    elif grade1_num == 7 or grade2_num == 7:
        return 0  # Strongly discourage mixing lone 7th graders
    else:
        return 1
