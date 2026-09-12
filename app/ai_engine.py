def explain_skill(df, future_skill):
    """Explain why a future skill is recommended."""

    data = df[df["Future_Skill"] == future_skill]

    if data.empty:
        return "No supporting data found for this skill."

    avg_impact = data["Impact_Score"].mean()
    roles = ", ".join(data["Role"].unique())
    activities = ", ".join(data["Activity"].unique())

    explanation = (
        f"{future_skill} is recommended because it is linked to "
        f"{len(data)} activity/activities across the role(s): {roles}. "
        f"The related activities include: {activities}. "
        f"The average AI impact score is {avg_impact:.1f}/5. "
        f"This indicates that AI is significantly changing the related work "
        f"and increasing the need for this future capability."
    )

    return explanation


def explain_role(df, role):
    """Explain why a role needs reskilling."""

    data = df[df["Role"] == role]

    if data.empty:
        return "No supporting data found for this role."

    avg_impact = data["Impact_Score"].mean()
    future_skills = ", ".join(data["Future_Skill"].unique())

    explanation = (
        f"{role} has an average AI impact score of {avg_impact:.1f}/5. "
        f"The role is connected to future skills such as {future_skills}. "
        f"Reskilling is recommended because AI is changing some of the "
        f"activities performed in this role."
    )

    return explanation


def answer_executive_question(df, question):
    """Simple executive Q&A based on the intelligence dataset."""

    question_lower = question.lower()

    if "future skill" in question_lower:
        skills = (
            df.groupby("Future_Skill")["Impact_Score"]
            .mean()
            .sort_values(ascending=False)
            .head(5)
        )

        return (
            "Top future skills based on the current intelligence data:\n\n"
            + "\n".join(
                [f"• {skill} — impact score {score:.1f}/5"
                 for skill, score in skills.items()]
            )
        )

    if "declining" in question_lower or "automation" in question_lower:
        data = df[df["AI_Impact"] == "Declining"]

        if data.empty:
            return "No declining skills were identified."

        skills = data["Current_Skill"].unique()

        return (
            "Skills most affected by automation:\n\n"
            + "\n".join([f"• {skill}" for skill in skills])
        )

    if "reskill" in question_lower or "role" in question_lower:
        roles = (
            df.groupby("Role")["Impact_Score"]
            .mean()
            .sort_values(ascending=False)
            .head(5)
        )

        return (
            "Roles requiring greater reskilling attention:\n\n"
            + "\n".join(
                [f"• {role} — impact score {score:.1f}/5"
                 for role, score in roles.items()]
            )
        )

    return (
        "I can answer questions about future skills, declining skills, "
        "automation impact, and reskilling priorities."
    )