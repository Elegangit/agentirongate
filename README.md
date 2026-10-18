<!-- project-identity:start -->
# AgentIronGate

A controlled research sandbox for what agents do after a denied action.

Study design: **A Controlled Study Design for Denial Feedback in Tool-Using AI Agents**.
<!-- project-identity:end -->

When a permission check blocks an AI agent, does the explanation help it get
back to the task? This code compares a short refusal, a rule explanation, and
a rule explanation with a task hint. Each run starts from the same saved state,
with the same permissions and action limits.

The default demo is scripted and runs offline. It checks the software, not
whether one message works better with a real model.

## Setup

Requires Python 3.11 or later.

Download or clone this repository, then open a terminal in its root folder.

```bash
conda env create -f environment.yml
conda activate agentgate
```

Already have the `agentgate` environment? Activate it and run
`python -m pip install -e '.[dev,live,plots]'` instead of creating it again.

## Try it

```bash
python -m gatebench run --output outputs/smoke
python -m gatebench analyze outputs/smoke
python -m gatebench view outputs/smoke
```

Open http://127.0.0.1:8765 and use **Next recorded step** to compare the branches.
No API key is needed. The viewer reads saved runs; it does not call a model.
Use a fresh output directory for each run. Stop the viewer with Ctrl+C, or add
`--port 8766` if the default port is busy.

For plots, run `python -m gatebench plot outputs/smoke`. To export a static viewer:

```bash
python -m gatebench view outputs/smoke --export dist/research-viewer
```

Serve the exported folder through a static host or local HTTP server. Review
its `data.json` before sharing—it includes prompts, tool results, and mock emails.

## Gemini runs

Set a model ID and request cap in `configs/gemini.toml`. Supply `GEMINI_API_KEY`
or `GOOGLE_API_KEY` through your shell; `.env` files are not loaded automatically.
Live requests may cost money.

```bash
python -m gatebench validate --config configs/gemini.toml
python -m gatebench run --config configs/gemini.toml \
  --allow-network --output outputs/gemini-pilot
python -m gatebench analyze outputs/gemini-pilot
```

Read the [experiment protocol](docs/protocol.md) before collecting results.
For a planned study, use `preflight` to check and freeze settings, then pass
the saved plan to `run --plan`. Plans bind research code, settings, tasks, and
runtime versions. These checks make no model calls.
The tasks are synthetic, approval is simulated, and completion uses literal
fact matching. Human checks are covered in the [scoring guide](docs/scoring-audit.md).
No live comparative results or production security guarantees are claimed.

## Development

```bash
python -m pytest -q
python -m ruff check src tests validation
python validation/check.py
```

Code lives in `src/gatebench/`, tasks in `data/`, and run settings in `configs/`.
Validation runs offline and keeps its reports local. Generated outputs and
credentials are excluded from Git; never publish private traces or API keys.

To rename the project, edit `project.toml` and run `python -m gatebench identity`.
This updates the identity block above without replacing the rest of the README.

## License

[MIT](LICENSE), including the synthetic task fixtures.

Maintained by [Aniqa Afzal](https://github.com/Elegangit).
