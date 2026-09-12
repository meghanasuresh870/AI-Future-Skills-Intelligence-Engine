import pandas as pd


def load_data(file_path):
    """Load the skills intelligence dataset."""
    return pd.read_csv(file_path)


def get_summary(df):
    """Return basic dataset summary."""
    return {
        "records": len(df),
        "roles": df["Role"].nunique(),
        "skills": df["Current_Skill"].nunique(),
        "future_skills": df["Future_Skill"].nunique(),
        "processes": df["Process"].nunique(),
    }


def filter_data(df, industry=None, role=None, impact=None):
    """Filter intelligence data."""
    result = df.copy()

    if industry and industry != "All":
        result = result[result["Industry"] == industry]

    if role and role != "All":
        result = result[result["Role"] == role]

    if impact and impact != "All":
        result = result[result["AI_Impact"] == impact]

    return result