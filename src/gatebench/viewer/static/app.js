"use strict";

const $ = (id) => document.getElementById(id);
const labels = {minimal: "Minimal", rule: "Rule-based", directed: "Task-directed", neutral: "Neutral control", hint: "Hint only"};
const colors = {minimal: "#56728c", rule: "#287d87", directed: "#8562bc", neutral: "#b17c22", hint: "#765c3b"};
const descriptions = {minimal: "Denial only", rule: "Denial + violated rule", directed: "Denial + rule + recovery hint", neutral: "Word-count-matched control", hint: "Denial + recovery hint"};
let data, step = 0, playback = null;
const human = (text) => text.replaceAll("_", " ");

function announce(message) { $("announcement").textContent = message; }

function pause() {
  clearInterval(playback);
  playback = null;
  $("play").textContent = "Play";
  $("play").setAttribute("aria-pressed", "false");
}

function maximumStep() { return Math.max(0, ...selectedBranches().map((b) => b.events.length)); }

function moveTo(value) {
  step = Math.max(0, Math.min(value, maximumStep()));
  render();
}

function filterTasks() {
  pause();
  const previous = $("task").value;
  const query = $("search").value.trim().toLowerCase();
  const family = $("family").value;
  const tasks = data.tasks.filter((t) => (!family || t.family === family) &&
    `${human(t.id)} ${t.prompt}`.toLowerCase().includes(query));
  options("task", tasks.map((t) => [t.id, human(t.id)]));
  if (tasks.some((t) => t.id === previous)) $("task").value = previous;
  $("workspace").hidden = tasks.length === 0;
  $("task").disabled = tasks.length === 0;
  $("filter-status").textContent = tasks.length ? `${tasks.length} tasks available · selection does not change the recorded experiment` : "No matching tasks. Clear the search or choose another family.";
  step = 0;
  if (tasks.length) render();
}

function exportComparison() {
  const task = data.tasks.find((t) => t.id === $("task").value);
  const selection = {run: data.run_name, evidence: data.evidence, task,
    model: data.models.find((m) => m.id === $("model").value),
    repeat: Number($("repeat").value),
    checkpoint: data.checkpoints[`${$("model").value}--${task.id}`],
    branches: selectedBranches(), scope: "One saved task/repeat; not aggregate results."};
  const url = URL.createObjectURL(new Blob([JSON.stringify(selection, null, 2)], {type: "application/json"}));
  const link = node("a");
  link.href = url;
  link.download = "recorded-comparison.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  announce("Comparison exported. Review prompts and tool outputs before sharing.");
}

function node(tag, text, className) {
  const el = document.createElement(tag);
  if (text !== undefined) el.textContent = text;
  if (className) el.className = className;
  return el;
}

function options(id, values) {
  $(id).replaceChildren(...values.map(([value, label]) => {
    const option = node("option", label);
    option.value = value;
    return option;
  }));
}

function selectedBranches() {
  return data.branches.filter((b) => b.task_id === $("task").value &&
    b.model_id === $("model").value && b.repeat === Number($("repeat").value));
}

function eventCard(event, index) {
  const card = node("div", undefined, "event");
  const heading = node("div", undefined, "event-head");
  heading.append(node("span", `${index + 1}. ${event.tool.replaceAll("_", " ")}`),
    node("span", event.decision, `tag ${event.decision}`));
  card.append(heading, node("p", event.rule.replaceAll("_", " ")));
  const target = event.args.document_id || event.args.recipient || event.args.draft_id;
  if (target) card.append(node("p", target, "event-target"));
  if (event.approval) card.append(node("p", `Synthetic approval: ${event.approval.approved ? "approved" : "rejected"}`));
  if (event.effect) card.append(node("p", `Recorded effect: ${event.effect.status}`));
  else card.append(node("p", "No tool effect recorded."));
  const details = node("details");
  details.append(node("summary", "Arguments and audit record"), node("pre", JSON.stringify(event, null, 2)));
  card.append(details);
  return card;
}

function render() {
  const task = data.tasks.find((t) => t.id === $("task").value);
  if (!task) return;
  const model = $("model").value;
  const cp = data.checkpoints[`${model}--${task.id}`];
  const prefix = data.prefixes.find((p) => p.task_id === task.id && p.model_id === model);
  $("copy-hash").disabled = !cp;
  $("export").disabled = !cp;
  $("task-prompt").textContent = task.prompt;
  $("checkpoint").replaceChildren();
  $("prefix-events").replaceChildren();
  $("prefix-details").hidden = !cp;
  if (cp) {
    const summary = node("div", undefined, "checkpoint-summary");
    summary.append(node("span", "Denied", "tag deny"), node("span", cp.call.name.replaceAll("_", " ")),
      node("span", cp.rule.replaceAll("_", " "), "muted"), node("span", `Session label: ${cp.label}`, "tag"));
    $("checkpoint").append(summary, node("div", `Shared checkpoint · ${cp.hash}`, "hash"));
    $("policy").textContent = JSON.stringify(cp.policy, null, 2);
    $("prefix-events").append(...cp.events.map(eventCard));
  } else {
    $("checkpoint").append(node("p", `No eligible denial checkpoint. Prefix ended: ${prefix?.termination ?? "not recorded"}. No branches were invented.`, "empty"));
  }
  const branches = selectedBranches();
  const longest = Math.max(0, ...branches.map((b) => b.events.length));
  $("step").max = longest;
  $("step").value = step;
  $("step").disabled = longest === 0;
  $("step-value").textContent = `${Math.min(step, longest)} / ${longest}`;
  $("back").disabled = step === 0;
  $("play").disabled = step >= longest;
  if (step >= longest) pause();
  $("progress").textContent = branches.length ? `Showing ${Math.min(step, longest)} of ${longest} recorded action steps. Columns align by action index, not elapsed time.` : "No matched continuations for this selection.";
  $("next").disabled = step >= longest;
  $("all").disabled = step >= longest;
  $("restart").disabled = step === 0;
  $("branches").replaceChildren();
  for (const condition of data.conditions) {
    const b = branches.find((item) => item.condition === condition);
    if (!b) continue;
    const col = node("article", undefined, "branch");
    col.style.setProperty("--accent", colors[condition] || "#56728c");
    col.append(node("h3", labels[condition] || condition), node("p", `${descriptions[condition] || condition} · run order ${b.order + 1}`, "subtitle"),
      node("div", "ACTUAL FEEDBACK AT CHECKPOINT", "small-label"), node("div", b.feedback.message, "feedback"),
      node("div", "POST-CHECKPOINT ACTIONS", "small-label"));
    col.append(...b.events.slice(0, step).map(eventCard));
    if (step === 0) col.append(node("p", "Use Next recorded step to follow this branch.", "empty"));
    if (step >= b.events.length) {
      col.append(node("p", `Termination: ${b.termination} · final label: ${b.label}`, "termination"));
      const outbox = node("details");
      outbox.append(node("summary", `Final mock outbox (${b.outbox.length})`), node("pre", JSON.stringify(b.outbox, null, 2)));
      col.append(outbox);
    }
    $("branches").append(col);
  }
  const table = $("metrics");
  table.replaceChildren();
  if (!branches.length) return;
  const ordered = data.conditions.map((c) => branches.find((b) => b.condition === c)).filter(Boolean);
  const baseline = ordered.find((b) => b.condition === "minimal");
  const head = node("thead"), hr = node("tr");
  hr.append(node("th", "Outcome"), ...ordered.map((b) => node("th", labels[b.condition] || b.condition)));
  head.append(hr);
  const body = node("tbody");
  for (const [metric, title, binary] of [["completed_before_checkpoint", "Complete before checkpoint", true], ["new_completion", "New completion after checkpoint", true], ["completion", "Final-state completion", true], ["prohibited_proposals", "Further prohibited proposals"], ["unauthorized_effect", "Unauthorized effect (integrity check)", true], ["attacker_success", "Attacker goal (integrity check)", true], ["model_calls", "Additional model calls"], ["provider_error", "Provider error", true]]) {
    const row = node("tr");
    const heading = node("th", title);
    heading.scope = "row";
    row.append(heading, ...ordered.map((b) => {
      const recorded = b.metrics[metric] !== undefined;
      const cell = node("td", !recorded ? "Not recorded" : binary ? (b.metrics[metric] ? "Yes" : "No") : String(b.metrics[metric]));
      if ($("differences").checked && recorded && baseline && baseline.metrics[metric] !== undefined && b !== baseline && !["unauthorized_effect", "attacker_success", "completed_before_checkpoint"].includes(metric)) {
        const difference = b.metrics[metric] - baseline.metrics[metric];
        const suffix = binary ? " (0/1 score)" : "";
        cell.append(node("span", `Δ ${difference > 0 ? "+" : ""}${difference}${suffix}`, `delta ${difference ? "changed" : ""}`));
      }
      return cell;
    }));
    body.append(row);
  }
  table.append(head, body);
}

async function start() {
  const response = await fetch("./data.json");
  if (!response.ok) throw new Error(`Run data could not be loaded (${response.status}).`);
  data = await response.json();
  document.title = `${data.name} · Research viewer`;
  $("project").textContent = data.name;
  $("run-meta").textContent = `${data.run_name} · ${data.branches.length} continuations · ${data.started_at}`;
  const offline = data.evidence === "scripted_fixture_only";
  $("evidence").textContent = offline ? "SCRIPTED FIXTURE CHECK — not live model results. Matching outcomes are expected; this run checks the experiment machinery." : "RECORDED LIVE RUN — model calls have already finished. Results describe this saved run, not a general security guarantee.";
  $("evidence").classList.toggle("live", !offline);
  $("settings").textContent = `Feedback scope: ${data.scope} · synthetic approver: ${data.approver} · branch limits: ${data.call_budget} calls / ${data.proposal_budget} proposals · recorded project name: ${data.recorded_name}`;
  const eligible = data.prefixes.filter((p) => p.termination === "eligible").length;
  const stats = [[data.tasks.length, "Synthetic tasks", `${new Set(data.tasks.map((t) => t.cluster)).size} source clusters`],
    [`${eligible} / ${data.prefixes.length}`, "Eligible prefixes", "Task–model runs"],
    [data.branches.length, "Saved continuations", `${data.conditions.length} feedback conditions`],
    [`${data.call_budget} / ${data.proposal_budget}`, "Calls / proposals", "Budget per continuation"]];
  $("overview").append(...stats.map(([value, title, detail]) => {
    const card = node("div", undefined, "stat");
    card.append(node("span", title), node("strong", String(value)), node("small", detail));
    return card;
  }));
  options("family", [["", "All families"], ...[...new Set(data.tasks.map((t) => t.family))].map((f) => [f, human(f)])]);
  options("task", data.tasks.map((t) => [t.id, t.id.replaceAll("_", " ")]));
  options("model", data.models.map((m) => [m.id, `${m.id} · ${m.model}`]));
  options("repeat", Array.from({length: data.repeats}, (_, i) => [i, `Repeat ${i + 1}`]));
  for (const id of ["task", "model", "repeat"]) $(id).addEventListener("change", () => { pause(); step = 0; render(); });
  $("search").addEventListener("input", filterTasks);
  $("family").addEventListener("change", filterTasks);
  $("next").addEventListener("click", () => { pause(); moveTo(step + 1); });
  $("back").addEventListener("click", () => { pause(); moveTo(step - 1); });
  $("restart").addEventListener("click", () => { pause(); moveTo(0); });
  $("all").addEventListener("click", () => { pause(); moveTo(maximumStep()); });
  $("step").addEventListener("input", () => { pause(); moveTo(Number($("step").value)); });
  $("play").addEventListener("click", () => {
    if (playback) return pause();
    $("play").textContent = "Pause";
    $("play").setAttribute("aria-pressed", "true");
    playback = setInterval(() => moveTo(step + 1), 1400);
  });
  $("differences").addEventListener("change", render);
  $("export").addEventListener("click", exportComparison);
  $("copy-hash").addEventListener("click", async () => {
    const cp = data.checkpoints[`${$("model").value}--${$("task").value}`];
    try { await navigator.clipboard.writeText(cp.hash); announce("Checkpoint ID copied."); }
    catch { announce("Clipboard unavailable. Select the checkpoint ID above to copy it manually."); }
  });
  $("focus").addEventListener("click", () => {
    const active = document.body.classList.toggle("presentation");
    $("focus").setAttribute("aria-pressed", String(active));
    $("focus").textContent = active ? "Exit presentation" : "Presentation view";
  });
  document.addEventListener("keydown", (event) => {
    if (event.altKey || event.ctrlKey || event.metaKey || /INPUT|SELECT|TEXTAREA|BUTTON|SUMMARY/.test(event.target.tagName) || $("workspace").hidden) return;
    if (["ArrowLeft", "ArrowRight"].includes(event.key)) {
      event.preventDefault(); pause(); moveTo(step + (event.key === "ArrowRight" ? 1 : -1));
    }
  });
  document.addEventListener("visibilitychange", () => { if (document.hidden) pause(); });
  filterTasks();
}

start().catch((error) => {
  $("error").hidden = false;
  $("error").textContent = `${error.message} Start the viewer with gatebench view, or serve the exported folder over HTTP instead of opening index.html directly.`;
});
