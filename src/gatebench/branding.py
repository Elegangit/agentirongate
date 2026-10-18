"""Update the README identity block; keep imports unchanged."""

from pathlib import Path

from .config import identity


def latex_text(value):
    escapes = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}",
               "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
               "_": r"\_", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(escapes.get(char, char) for char in value)


def render_identity(root, latex=None):
    root = Path(root)
    project = identity(root)
    readme = root / "README.md"
    text = readme.read_text()
    start, end = "<!-- project-identity:start -->", "<!-- project-identity:end -->"
    if text.count(start) != 1 or text.count(end) != 1 or text.index(start) > text.index(end):
        raise ValueError("README must contain one ordered project identity block")
    block = (f"{start}\n# {project['name']}\n\n{project['description']}\n\n"
             f"Study design: **{project['paper_title']}**.\n{end}")
    readme.write_text(text[:text.index(start)] + block + text[text.index(end) + len(end):])
    if latex is not None:
        Path(latex).write_text(
            "% Generated from Codebase/project.toml by gatebench identity.\n"
            + r"\newcommand{\systemname}{" + latex_text(project["name"]) + "}\n"
            + r"\newcommand{\papertitle}{\systemname: "
            + latex_text(project["paper_title"]) + "}\n")
    return project["name"]
