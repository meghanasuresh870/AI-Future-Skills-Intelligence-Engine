import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set.")

client = genai.Client(api_key=api_key)


def generate_answer(
    question,
    context,
    intelligence_context="",
    intent="general"
):

    prompt = f"""
You are an AI Future Skills Intelligence assistant.

You have two sources:

1. STRUCTURED INTELLIGENCE DATA
This contains:
Process → Activity → Role → Current Skill → AI Impact → Future Skill.

2. KNOWLEDGE BASE
Retrieved documents that provide supporting workforce context.

STRUCTURED INTELLIGENCE DATA:
{intelligence_context}

KNOWLEDGE BASE:
{context}

USER QUESTION:
{question}

QUESTION INTENT:
{intent}

Instructions:
- Answer using the structured intelligence data and knowledge base.
- Give priority to structured intelligence data for project-specific findings.
- Do not invent facts.
- Keep the answer concise and executive-friendly.
- Maximum 150 words.
- Do not use Markdown headings.
- Do not generate links.
- Do not generate [svg] or HTML.
- Do not repeat the same information.

Use the detected question intent.

If intent is "future_skills":
Use the CALCULATED TOP FUTURE SKILLS section.

If intent is "declining_skills":
Use the CALCULATED DECLINING SKILLS section.

If intent is "reskilling":
Use the CALCULATED RESKILLING PRIORITIES section.

If intent is "general":
Use the detailed intelligence mappings and knowledge base.

For questions about a specific role, skill, process, or activity,
focus on that specific entity and do not unnecessarily show unrelated top-5 rankings.

Question handling rules:

- If the user asks for a ranking or "top" list, use the corresponding calculated intelligence section and show the requested top findings.
- If the user asks about a specific role, focus only on that role. Do not show the general top 5 roles.
- If the user asks about a specific skill, focus only on that skill. Do not show unrelated rankings.
- If the user asks about a specific process or activity, focus on the relevant process/activity and explain its AI impact.
- For role-specific answers, mention the Current Skill, AI Impact, Future Skill, Impact Score, and Evidence when available.
- For skill-specific answers, mention the related role, activity, AI Impact, Impact Score, and Evidence when available.
- For process/activity questions, use the detailed intelligence mappings.
- Never invent information that is not present in the provided intelligence data or knowledge base.

Use this simple format when appropriate:

Summary:
...

Top Findings:
1. ...
2. ...
3. ...
4. ...
5. ...

Why:
...

Keep the response short.
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text