import pandas as pd
import numpy as np

def normalize_preferences(row, role_columns):
    """Normalize preference scores for a student."""
    # Get preference scores for this student
    scores = row[role_columns].values.astype(float)
    unique_scores = np.unique(scores)
    
    # If they only used two values
    if len(unique_scores) == 2:
        min_score, max_score = unique_scores.min(), unique_scores.max()
        
        # Case: Used 1s and 2s -> map 2s to 3s
        if min_score == 1 and max_score == 2:
            scores = np.where(scores == 2, 3, scores)
        # Case: Used 2s and 3s -> map 2s to 1s
        elif min_score == 2 and max_score == 3:
            scores = np.where(scores == 2, 1, scores)
    
    # Create a new row with normalized scores
    result = row.copy()
    result[role_columns] = scores
    return result

def anonymize_data(input_file, output_file):
    # Read the original CSV
    df = pd.read_csv(input_file)
    
    # Select only the columns we need
    columns_to_keep = [
        'ID', 'Gender', 'Grade',
        'team_captain', 'innovation_project_leader', 'mission_strategist',
        'public_relations_lead', 'lego_lead_builder', 'lead_coder'
    ]
    
    # Role columns for normalization
    role_columns = [
        'team_captain', 'innovation_project_leader', 'mission_strategist',
        'public_relations_lead', 'lego_lead_builder', 'lead_coder'
    ]
    
    # Create anonymized dataframe
    anonymized_df = df[columns_to_keep].copy()
    
    # Normalize preferences for each student
    anonymized_df = anonymized_df.apply(
        lambda row: normalize_preferences(row, role_columns), 
        axis=1
    )
    
    # Save to new CSV
    anonymized_df.to_csv(output_file, index=False)
    print(f"Anonymized data saved to {output_file}")

if __name__ == "__main__":
    anonymize_data('student_data.csv', 'anonymized_student_data.csv')
