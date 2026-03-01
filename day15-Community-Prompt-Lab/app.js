import { addVote, createEntry, leaderboard, remixDraft } from "./core.js";

const STORAGE_KEY = "day15_prompt_lab_entries";

const defaultEntries = [
  {
    id: crypto.randomUUID(),
    title: "Teach quantum tunneling with a kitchen analogy",
    prompt: "Explain quantum tunneling like I am 12 years old using only kitchen objects.",
    model: "GPT-4 class",
    type: "text",
    theme: "education",
    result: "Used eggs and walls analogy. Great clarity, but needed fewer metaphors.",
    votes: { creative: 1, useful: 2, repro: 1 },
    createdAt: Date.now(),
  },
];

const form = document.querySelector("#prompt-form");
const cardsWrap = document.querySelector("#cards");
const clearBtn = document.querySelector("#clear-all");
const themeFilter = document.querySelector("#theme-filter");
const boardWrap = document.querySelector("#leaderboard");
const tpl = document.querySelector("#card-template");

function loadEntries() {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return defaultEntries;
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : defaultEntries;
  } catch {
    return defaultEntries;
  }
}

function saveEntries(entries) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
}

let entries = loadEntries();

function typeLabel(value) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function formatDate(ts) {
  return new Date(ts).toLocaleString();
}

function filteredEntries() {
  const selectedTheme = themeFilter.value;
  if (selectedTheme === "all") return entries;
  return entries.filter((entry) => entry.theme === selectedTheme);
}

function publish(entry) {
  entries = [entry, ...entries];
  saveEntries(entries);
  render();
}

function vote(id, bucket) {
  entries = addVote(entries, id, bucket);
  saveEntries(entries);
  render();
}

function removeEntry(id) {
  entries = entries.filter((entry) => entry.id !== id);
  saveEntries(entries);
  render();
}

function remix(entry) {
  const draft = remixDraft(entry);
  form.title.value = draft.title;
  form.prompt.value = draft.prompt;
  form.model.value = draft.model;
  form.type.value = draft.type;
  form.theme.value = draft.theme;
  form.result.value = draft.result;
  form.scrollIntoView({ behavior: "smooth", block: "start" });
  form.title.focus();
}

function renderLeaderboard() {
  const top = leaderboard(entries, 5);
  boardWrap.innerHTML = "";
  if (top.length === 0) {
    boardWrap.textContent = "No entries yet.";
    return;
  }

  const list = document.createElement("ol");
  top.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = `${item.title} — ${item.points} pts`;
    list.appendChild(li);
  });
  boardWrap.appendChild(list);
}

function render() {
  cardsWrap.innerHTML = "";

  for (const entry of filteredEntries()) {
    const node = tpl.content.firstElementChild.cloneNode(true);
    node.querySelector(".card-title").textContent = entry.title;
    node.querySelector(".card-type").textContent = typeLabel(entry.type);
    node.querySelector(".meta").textContent = `Model: ${entry.model} • Theme: ${entry.theme} • ${formatDate(entry.createdAt)}`;
    node.querySelector(".card-prompt").textContent = entry.prompt;
    node.querySelector(".card-result").textContent = entry.result;

    node.querySelectorAll("[data-vote]").forEach((btn) => {
      const key = btn.dataset.vote;
      btn.querySelector(".count").textContent = `(${entry.votes?.[key] || 0})`;
      btn.addEventListener("click", () => vote(entry.id, key));
    });

    node.querySelector('[data-action="remix"]').addEventListener("click", () => remix(entry));
    node.querySelector('[data-action="delete"]').addEventListener("click", () => removeEntry(entry.id));

    cardsWrap.appendChild(node);
  }

  renderLeaderboard();
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const data = new FormData(form);

  const result = createEntry(
    {
      title: data.get("title"),
      prompt: data.get("prompt"),
      model: data.get("model"),
      type: data.get("type"),
      theme: data.get("theme"),
      result: data.get("result"),
    },
    () => crypto.randomUUID(),
  );

  if (!result.ok) return;

  publish(result.entry);
  form.reset();
  form.theme.value = "open";
});

themeFilter.addEventListener("change", render);

clearBtn.addEventListener("click", () => {
  localStorage.removeItem(STORAGE_KEY);
  entries = defaultEntries;
  render();
});

render();
