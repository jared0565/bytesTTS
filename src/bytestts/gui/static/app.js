// SPDX-License-Identifier: MPL-2.0
// The control panel page (spec §5.1, §6.2, §6.7 design B).
// Everything goes over the loopback API with the secret in the Authorization header. Untrusted text (file names,
// errors, catalogue attributions) goes in with textContent, never as HTML.
"use strict";

const POLL_MS = 330;
const SECRET_KEY = "bytestts.secret";
const MAX_TEXT_FILE_BYTES = 1000000; // a text file larger than this is refused rather than frozen into the box
const GPU_REASON = "GPU needs the [gpu] extra with CUDA 13 and cuDNN 9";
const $ = (id) => document.getElementById(id);

const state = {
  status: null, // GET /api/status
  voices: [], // GET /api/voices
  job: null, // the running job's id
  waiting: false, // a Speak queued behind another CPU task
  downloading: null, // the voice being downloaded
  recording: null, // {blob, url, name}
  unsaved: false,
  overLimitShown: false,
  locked: false,
  closed: false,
};

// ---- The secret (spec §6.2 rule 3, open item 1): the address carries a one-time launch code, never the secret.
// The page swaps the code once at POST /api/session for the secret, keeps it here and in sessionStorage (so a
// reload still works), and removes the code from view. Using the same address a second time stops the panel.

const ADDRESS_CHANGED = "This panel's address has changed: open the address printed in the terminal.";
const SERVER_GONE = "The panel has stopped: start it again with bytestts gui.";
let heldSecret = "";

async function takeSecret() {
  const code = new URLSearchParams(location.search).get("code");
  if (location.search) {
    history.replaceState(null, "", "/");
  }
  if (!code) {
    return; // a reload, or Back and Forward: the secret is already in sessionStorage
  }
  if (document.prerendering) {
    // a browser preparing the page ahead of a visit must not use up the code
    await new Promise((resolve) => document.addEventListener("prerenderingchange", resolve, { once: true }));
  }
  let response;
  try {
    response = await fetch("/api/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code }),
      cache: "no-store",
    });
  } catch (e) {
    lockOut(SERVER_GONE);
    return;
  }
  let data = null;
  try {
    data = await response.json();
  } catch (e) {
    // not JSON: the default message
  }
  if (response.ok && data && typeof data.secret === "string") {
    heldSecret = data.secret;
    try {
      sessionStorage.setItem(SECRET_KEY, data.secret);
    } catch (e) {
      // no sessionStorage: this page works, a reload will show the address message
    }
    return;
  }
  lockOut(data && typeof data.error === "string" ? data.error : ADDRESS_CHANGED);
}

function secret() {
  if (heldSecret) {
    return heldSecret;
  }
  try {
    return sessionStorage.getItem(SECRET_KEY) || "";
  } catch (e) {
    return "";
  }
}

class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

async function api(path, options = {}) {
  const headers = Object.assign({ Authorization: "Bearer " + secret() }, options.headers || {});
  let body = options.body;
  if (options.json !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(options.json);
  }
  let response;
  try {
    response = await fetch("/api" + path, { method: options.method || "GET", headers, body, cache: "no-store" });
  } catch (e) {
    throw new ApiError(0, SERVER_GONE); // no answer at all: Quit, Ctrl+C or a reused address stopped the server
  }
  if (response.status === 401) {
    lockOut();
    throw new ApiError(401, ADDRESS_CHANGED);
  }
  if (!response.ok) {
    let message = "The panel answered " + response.status;
    try {
      const data = await response.json();
      if (data && typeof data.error === "string") {
        message = data.error;
      }
    } catch (e) {
      // not JSON: keep the status line
    }
    throw new ApiError(response.status, message);
  }
  return options.raw ? response : response.json();
}

function lockOut(message = ADDRESS_CHANGED) {
  state.locked = true;
  setStatus(message, "error");
  for (const control of document.querySelectorAll("button, input, select, textarea")) {
    control.disabled = true;
  }
}

function setStatus(text, kind) {
  const line = $("status");
  line.textContent = text;
  line.dataset.state = kind || "ready";
}

function count(n, word) {
  return n + " " + word + (n === 1 ? "" : "s");
}

function megabytes(bytes) {
  return (bytes / 1e6).toFixed(1) + " MB";
}

function option(value, text) {
  const item = document.createElement("option");
  item.value = value;
  item.textContent = text;
  return item;
}

// ---- Voices, downloads and repair (spec §6.5).

function currentVoice() {
  return state.voices.find((v) => v.id === $("voice").value) || null;
}

async function refreshVoices() {
  const chosen = $("voice").value;
  state.voices = await api("/voices");
  const marks = { downloaded: " (downloaded)", "needs-repair": " (needs repair)", missing: "" };
  $("voice").replaceChildren(...state.voices.map((v) => option(v.id, v.id + (marks[v.state] || ""))));
  if (state.voices.some((v) => v.id === chosen)) {
    $("voice").value = chosen;
  } else {
    const ready = state.voices.find((v) => v.state === "downloaded");
    if (ready) {
      $("voice").value = ready.id;
    }
  }
  showVoice();
}

function showVoice() {
  const voice = currentVoice();
  const busy = state.downloading !== null;
  $("download").hidden = !voice || voice.state !== "missing" || busy;
  $("repair").hidden = !voice || voice.state !== "needs-repair" || busy;
  if (voice) {
    $("download").textContent = "Download " + megabytes(voice.size_bytes);
  }
  const speakers = voice ? voice.speakers : [];
  $("speaker").replaceChildren(...speakers.map((name) => option(name, name)));
  $("speaker").disabled = speakers.length <= 1;
  $("info").disabled = !voice;
  updateControls();
}

async function startDownload() {
  const voice = currentVoice();
  if (!voice || state.downloading !== null) {
    return;
  }
  try {
    await api("/voices/" + encodeURIComponent(voice.id) + "/download", { method: "POST" });
  } catch (e) {
    setStatus(e.message, "error");
    return;
  }
  state.downloading = voice.id;
  $("download-bar").value = 0;
  $("download-text").textContent = "";
  for (const id of ["download-bar", "download-text", "cancel"]) {
    $(id).hidden = false;
  }
  showVoice();
  setStatus((voice.state === "needs-repair" ? "Repairing " : "Downloading ") + voice.id, "busy");
  pollDownload(voice.id);
}

async function pollDownload(id) {
  let s;
  try {
    s = await api("/voices/" + encodeURIComponent(id) + "/download");
  } catch (e) {
    endDownload();
    setStatus(e.message, "error");
    return;
  }
  $("download-bar").max = Math.max(s.total, 1);
  $("download-bar").value = s.done;
  $("download-text").textContent = megabytes(s.done) + " of " + megabytes(s.total);
  if (s.state === "running") {
    setTimeout(() => pollDownload(id), POLL_MS);
    return;
  }
  endDownload();
  if (s.state === "done") {
    setStatus(id + " is ready", "ready");
  } else if (s.state === "cancelled") {
    setStatus("Download cancelled", "ready");
  } else {
    setStatus(s.error || "The download failed", "error");
  }
  await refreshVoices().catch((e) => setStatus(e.message, "error"));
}

function endDownload() {
  state.downloading = null;
  for (const id of ["download-bar", "download-text", "cancel"]) {
    $(id).hidden = true;
  }
  showVoice();
}

async function cancelDownload() {
  if (state.downloading === null) {
    return;
  }
  try {
    await api("/voices/" + encodeURIComponent(state.downloading) + "/download/cancel", { method: "POST" });
  } catch (e) {
    setStatus(e.message, "error");
  }
}

function showInfo() {
  const voice = currentVoice();
  if (!voice) {
    return;
  }
  $("info-title").textContent = voice.id;
  $("info-language").textContent = voice.language;
  $("info-licence").textContent = voice.licence || "not stated";
  $("info-data-licence").textContent = voice.data_licence || "not stated";
  $("info-size").textContent = megabytes(voice.size_bytes);
  $("info-rate").textContent = voice.sample_rate + " Hz";
  $("info-attribution").textContent = voice.attribution || "none";
  $("copy-attribution").textContent = "Copy attribution";
  $("info-dialog").showModal();
}

async function copyAttribution() {
  const button = $("copy-attribution");
  try {
    await navigator.clipboard.writeText($("info-attribution").textContent);
    button.textContent = "Copied";
  } catch (e) {
    button.textContent = "Copy failed";
  }
}

// ---- Text (spec §5.1): counted in code points, as the server counts.

function textLength() {
  return Array.from($("text").value).length;
}

function textLimit() {
  return state.status ? state.status.text_limit : 5000;
}

function updateCounter() {
  const n = textLength();
  $("counter").textContent = n + " / " + textLimit();
  $("counter").classList.toggle("over", n > textLimit());
  updateControls();
}

async function openTextFile() {
  const file = $("file").files[0];
  $("file").value = "";
  if (!file) {
    return;
  }
  if (file.size > MAX_TEXT_FILE_BYTES) {
    setStatus(file.name + " is larger than 1 MB; open a shorter text file", "error");
    return;
  }
  try {
    const text = new TextDecoder("utf-8", { fatal: true }).decode(await file.arrayBuffer()); // drops a BOM
    $("text").value = text;
    setStatus("Opened " + file.name, "ready");
  } catch (e) {
    setStatus("That file is not UTF-8 text", "error");
  }
  updateCounter();
}

// ---- Controls: what can be pressed right now.

function updateControls() {
  if (state.locked || state.closed) {
    return;
  }
  const voice = currentVoice();
  const n = textLength();
  const over = n > textLimit();
  const running = state.job !== null;
  $("speak").disabled = running || state.waiting || !voice || voice.state !== "downloaded" || n === 0 || over;
  $("stop").disabled = !running && !state.waiting && $("player").paused;
  $("save").disabled = !state.recording || running;
  if (over) {
    setStatus("Remove " + count(n - textLimit(), "character") + " to speak", "error");
    state.overLimitShown = true;
  } else if (state.overLimitShown) {
    state.overLimitShown = false;
    setStatus("Ready", "ready");
  }
}

function applyStatus(s) {
  $("version").textContent = "version " + s.version;
  $("threads").max = String(s.max_threads);
  $("threads-max").textContent = "(max " + s.max_threads + ")";
  $("gpu").checked = false;
  $("gpu").disabled = !s.cuda_available;
  $("gpu-reason").textContent = s.cuda_available ? "" : GPU_REASON;
  $("gpu-reason").hidden = s.cuda_available;
}

function threadsValue() {
  const max = state.status ? state.status.max_threads : 1;
  const n = Math.round(Number($("threads").value));
  const clamped = Number.isFinite(n) ? Math.min(Math.max(n, 1), max) : 1;
  $("threads").value = String(clamped);
  return clamped;
}

// ---- Speak, Stop and the player (spec §6.4).

function recordingName(voiceId) {
  const now = new Date();
  const two = (n) => String(n).padStart(2, "0");
  return voiceId + " " + two(now.getHours()) + "-" + two(now.getMinutes()) + "-" + two(now.getSeconds());
}

async function speak() {
  const voice = currentVoice();
  if (!voice) {
    return;
  }
  const request = {
    text: $("text").value,
    voice: voice.id,
    speaker: $("speaker").disabled ? null : $("speaker").value || null,
    speed: Number($("speed").value),
    device: $("gpu").checked ? "cuda" : "cpu",
    threads: threadsValue(),
  };
  try {
    const started = await api("/jobs", { method: "POST", json: request });
    state.waiting = false;
    state.job = started.id;
    setStatus("Generating: sentence 1", "busy");
    $("progress").textContent = "";
    updateControls();
    pollJob(started.id, voice.id);
  } catch (e) {
    if (e.status === 409) {
      waitThenSpeak(); // spec §6.2 rule 9: the page queues instead of sending a second CPU task
      return;
    }
    state.waiting = false;
    setStatus(e.message, "error");
    updateControls();
  }
}

async function waitThenSpeak() {
  if (!state.waiting) {
    state.waiting = true;
    setStatus("Waiting for the running task to finish", "busy");
    updateControls();
  }
  let s = null;
  try {
    s = await api("/status");
  } catch (e) {
    state.waiting = false;
    setStatus(e.message, "error");
    updateControls();
    return;
  }
  if (!state.waiting) {
    return; // Stop pressed while waiting
  }
  if (s.busy) {
    setTimeout(waitThenSpeak, POLL_MS);
  } else {
    speak();
  }
}

async function pollJob(id, voiceId) {
  let s;
  try {
    s = await api("/jobs/" + id);
  } catch (e) {
    state.job = null;
    setStatus(e.message, "error");
    updateControls();
    return;
  }
  if (s.state === "running") {
    setStatus("Generating: sentence " + (s.sentences_done + 1), "busy");
    setTimeout(() => pollJob(id, voiceId), POLL_MS);
    return;
  }
  state.job = null;
  if (s.state === "failed") {
    setStatus(s.error || "Generation failed", "error");
    await refreshVoices().catch(() => {}); // a damaged voice now shows Repair
    updateControls();
    return;
  }
  if (s.sentences_done === 0) {
    setStatus("Stopped before the first sentence was finished", "ready");
    updateControls();
    return;
  }
  try {
    const response = await api("/jobs/" + id + "/audio", { raw: true });
    loadRecording(await response.blob(), recordingName(voiceId));
    const done = s.state === "stopped" ? "Stopped: kept " + count(s.sentences_done, "sentence") : "Done";
    setStatus([done, ...s.warnings].join(". "), "ready");
    $("player").play().catch(() => {});
  } catch (e) {
    setStatus(e.message, "error");
  }
  updateControls();
}

function loadRecording(blob, name) {
  if (state.recording) {
    URL.revokeObjectURL(state.recording.url);
  }
  state.recording = { blob, url: URL.createObjectURL(blob), name };
  $("player").src = state.recording.url;
  setUnsaved(true);
}

async function stop() {
  $("player").pause();
  if (state.waiting) {
    state.waiting = false;
    setStatus("Ready", "ready");
  }
  if (state.job !== null) {
    try {
      await api("/jobs/" + state.job + "/stop", { method: "POST" });
      setStatus("Stopping after the current sentence", "busy");
    } catch (e) {
      setStatus(e.message, "error");
    }
  }
  updateControls();
}

// ---- Save: the window's Save dialog through the server (design B), or a download in the browser.

async function save() {
  const recording = state.recording;
  if (!recording) {
    return;
  }
  if (!state.status.save_dialog) {
    const link = document.createElement("a");
    link.href = recording.url;
    link.download = recording.name + ".wav";
    link.click();
    setStatus("Downloaded " + recording.name + ".wav", "ready");
    setUnsaved(false);
    return;
  }
  let choice;
  try {
    choice = await api("/save-dialog", { method: "POST", json: { name: recording.name, extension: "wav" } });
  } catch (e) {
    setStatus(e.message, "error");
    return;
  }
  if (!choice.ticket) {
    return; // Cancel in the dialog
  }
  setStatus("Saving " + choice.name, "busy");
  try {
    const path = "/export?format=wav&quality=standard&ticket=" + encodeURIComponent(choice.ticket);
    const result = await api(path, { method: "POST", body: recording.blob, headers: { "Content-Type": "audio/wav" } });
    setStatus("Saved " + result.saved, "ready");
    setUnsaved(false);
  } catch (e) {
    setStatus(e.message, "error");
  }
}

// ---- Unsaved work, closing and Quit (spec §6.7).

function setUnsaved(flag) {
  state.unsaved = flag;
  api("/unsaved", { method: "POST", json: { unsaved: flag } }).catch(() => {});
}

function quit() {
  if (state.unsaved) {
    $("quit-dialog").showModal();
  } else {
    doQuit();
  }
}

async function doQuit() {
  state.closed = true; // the page is going away on purpose: no "leave page?" question
  try {
    await api("/quit", { method: "POST" });
  } catch (e) {
    // the server may already be gone
  }
  $("panel").hidden = true;
  $("closed").hidden = false;
}

// ---- Wiring.

function wire() {
  $("voice").addEventListener("change", showVoice);
  $("download").addEventListener("click", startDownload);
  $("repair").addEventListener("click", startDownload);
  $("cancel").addEventListener("click", cancelDownload);
  $("info").addEventListener("click", showInfo);
  $("copy-attribution").addEventListener("click", copyAttribution);
  $("speed").addEventListener("input", () => {
    $("speed-value").textContent = Number($("speed").value).toFixed(2);
  });
  $("speed-reset").addEventListener("click", () => {
    $("speed").value = "1";
    $("speed-value").textContent = "1.00";
  });
  $("threads").addEventListener("change", threadsValue);
  $("text").addEventListener("input", updateCounter);
  $("file").addEventListener("change", openTextFile);
  $("clear").addEventListener("click", () => {
    $("text").value = "";
    updateCounter();
    $("text").focus();
  });
  $("speak").addEventListener("click", speak);
  $("stop").addEventListener("click", stop);
  $("save").addEventListener("click", save);
  $("quit").addEventListener("click", quit);
  $("quit-dialog").addEventListener("close", () => {
    if ($("quit-dialog").returnValue === "quit") {
      doQuit();
    }
  });
  for (const event of ["play", "pause", "ended"]) {
    $("player").addEventListener(event, updateControls);
  }
  window.addEventListener("beforeunload", (e) => {
    if (state.unsaved && !state.closed) {
      e.preventDefault();
      e.returnValue = "";
    }
  });
  // A file dropped on the page would replace it with that file: refuse file drops, keep text drops.
  const refuseFiles = (e) => {
    if (e.dataTransfer && Array.from(e.dataTransfer.types).includes("Files")) {
      e.preventDefault();
      e.dataTransfer.dropEffect = "none";
    }
  };
  document.addEventListener("dragover", refuseFiles);
  document.addEventListener("drop", refuseFiles);
}

async function start() {
  await takeSecret();
  if (state.locked) {
    return; // the exchange was refused: its message is showing and every control is disabled
  }
  wire();
  try {
    state.status = await api("/status");
    applyStatus(state.status);
    await refreshVoices();
  } catch (e) {
    if (e.status !== 401) {
      setStatus(e.message, "error");
    }
    return;
  }
  updateCounter();
  if (!state.status.espeak.found) {
    setStatus(state.status.espeak.hint, "error");
  } else if (!state.overLimitShown) {
    setStatus("Ready", "ready");
  }
}

start();
