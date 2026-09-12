import pandas as pd


def top_future_skills(df, top_n=10):
    """Rank future skills based on impact score."""
    result = (
        df.groupby("Future_Skill")
        .agg(
            Impact_Score=("Impact_Score", "mean"),
            Activities=("Activity", "count"),
            Roles=("Role", "nunique"),
        )
        .reset_index()
    )

    result["Priority_Score"] = (
        result["Impact_Score"] * 0.6
        + result["Activities"] * 0.2
        + result["Roles"] * 0.2
    )

    return result.sort_values(
        "Priority_Score", ascending=False
    ).head(top_n)


def declining_skills(df):
    """Find skills most affected by automation."""
    return (
        df[df["AI_Impact"] == "Declining"]
        .groupby("Current_Skill")
        .agg(
            Impact_Score=("Impact_Score", "mean"),
            Activities=("Activity", "count"),
            Roles=("Role", "nunique"),
        )
        .reset_index()
        .sort_values("Impact_Score", ascending=False)
    )


def reskilling_priorities(df, top_n=10):
    """Identify roles requiring the greatest reskilling."""
    result = (
        df.groupby("Role")
        .agg(
            Average_Impact=("Impact_Score", "mean"),
            Activities=("Activity", "count"),
            Current_Skills=("Current_Skill", "nunique"),
            Future_Skills=("Future_Skill", "nunique"),
        )
        .reset_index()
    )

    result["Reskilling_Score"] = (
        result["Average_Impact"] * 0.6
        + result["Future_Skills"] * 0.3
        + result["Activities"] * 0.1
    )

    return result.sort_values(
        "Reskilling_Score", ascending=False
    ).head(top_n)


def impact_distribution(df):
    """Count skills by AI impact category."""
    return (
        df["AI_Impact"]
        .value_counts()
        .reset_index()
        .rename(columns={"count": "Count"})
    )


def role_skill_gap(df, role):
    """Show current and future skills for a selected role."""
    result = df[df["Role"] == role][
        ["Role", "Activity", "Current_Skill", "AI_Impact", "Future_Skill"]
    ]

    return result