import pandas as pd

def anonymize_data(input_file, output_file):
    # Read the original CSV
    df = pd.read_csv(input_file)
    
    # Select only the columns we need
    columns_to_keep = [
        'ID', 'Gender', 'Grade',
        'team_captain', 'innovation_project_leader', 'mission_strategist',
        'public_relations_lead', 'lego_lead_builder', 'lead_coder'
    ]
    
    # Create anonymized dataframe
    anonymized_df = df[columns_to_keep].copy()
    
    # Save to new CSV
    anonymized_df.to_csv(output_file, index=False)
    print(f"Anonymized data saved to {output_file}")

if __name__ == "__main__":
    anonymize_data('student_data.csv', 'anonymized_student_data.csv')
