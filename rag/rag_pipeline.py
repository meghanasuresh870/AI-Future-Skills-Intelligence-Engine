from rag.retriever import RAGRetriever, build_context
from rag.gemini_client import generate_answer


def detect_intent(question):
    """
    Detect the type of question asked by the user.
    """

    q = question.lower()

    if any(word in q for word in [
        "future skill",
        "future skills",
        "top skill",
        "top skills"
    ]):
        return "future_skills"

    if any(word in q for word in [
        "declining",
        "decline",
        "automation",
        "automated"
    ]):
        return "declining_skills"

    if any(word in q for word in [
        "reskill",
        "reskilling"
    ]):
        return "reskilling"

    return "general"


def ask_ai(question, intelligence_context=""):
    """
    Retrieve relevant knowledge and generate an AI answer.
    """

    retriever = RAGRetriever()

    results = retriever.retrieve(
        question,
        top_k=2
    )

    context = build_context(results)

    intent = detect_intent(question)

    answer = generate_answer(
        question,
        context,
        intelligence_context,
        intent=intent
    )

    return {
        "answer": answer,
        "sources": results
    }