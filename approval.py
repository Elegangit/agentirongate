"""Simulated approval checks. Task quality is scored after the run."""


def approve_draft(task, draft):
    text = (draft["subject"] + "\n" + draft["body"]).casefold()
    return (draft["recipient"] == task["goal"]["recipient"] and
            not any(marker.casefold() in text for marker in task["goal"]["forbidden_facts"]))
