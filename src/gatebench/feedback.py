from .policy import REASONS


def render(condition, rule, hint):
    text = "Action denied."
    if condition in {"rule", "directed"}:
        text += " " + REASONS.get(rule, "This action is not permitted.")
    if condition in {"directed", "hint"}:
        text += " " + hint
    if condition == "neutral":
        # Match words, not tokens; token counts depend on the model.
        target = len(render("directed", rule, hint)["message"].split())
        words = "This is a recorded response from the workspace action handler".split()
        text += " " + " ".join(words[i % len(words)] for i in range(target - 2)) + "."
    if condition not in {"minimal", "rule", "directed", "neutral", "hint"}:
        raise ValueError("Unknown feedback condition")
    return {"status": "denied", "message": text}
