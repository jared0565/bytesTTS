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
  fetchingAudio: false, // a finished job's audio is on its way: until then the job still counts as running
  waiting: false, // a Speak queued behind another CPU task
  downloading: null, // the voice being downloaded
  recording: null, // {blob, url, name}
  unsaved: false,
  overLimitShown: false,
  chosenSpeaker: null, // the speaker the user chose, or the voice's default; kept while the filters hide it
  noMatchShown: false,
  choice: "", // the voice and speaker in view, so a change of either can stop a sample
  sampleToken: 0, // bumped by every sample press, voice or speaker change, and Stop
  sampleAbort: null, // the pending clip fetch's AbortController
  sampleUrl: null, // the blob URL the sampler plays
  samplePlaying: null, // {voice, speaker}: what the sampler holds
  sampleButton: null, // the button whose clip plays: it reads ■ Stop
  jobKind: "speak", // "speak" or "preview": where a finished job's audio goes
  jobNote: null, // what a running job's status says instead of its sentence count
  preview: null, // {token, voice, speaker}: the running live Preview
  locked: false,
  closed: false,
};

// ---- The secret (spec §6.2 rule 3, open item 1): the address carries a one-time launch code, never the secret.
// The page swaps the code once at POST /api/session for the secret, keeps it here and in sessionStorage (so a
// reload still works), and removes the code from view. Using the same address a second time stops the panel.

const NOT_CONNECTED = "This page is not connected to the running panel: switch to the panel's own tab or window, or start it again with bytestts gui.";
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
  lockOut(data && typeof data.error === "string" ? data.error : NOT_CONNECTED);
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
    response = await fetch("/api" + path, {
      method: options.method || "GET",
      headers,
      body,
      cache: "no-store",
      signal: options.signal,
    });
  } catch (e) {
    if (e && e.name === "AbortError") {
      throw e; // the caller's own time limit, not a stopped server
    }
    throw new ApiError(0, SERVER_GONE); // no answer at all: Quit, Ctrl+C or a reused address stopped the server
  }
  if (response.status === 401) {
    lockOut();
    throw new ApiError(401, NOT_CONNECTED);
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

function lockOut(message = NOT_CONNECTED) {
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
  $("voice").replaceChildren(...state.voices.map((v) => option(v.id, v.label + (marks[v.state] || ""))));
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
  $("sample").hidden = !voice || !voice.has_sample;
  if (voice) {
    $("download").textContent = "Download " + megabytes(voice.size_bytes);
  }
  showFilters(voice);
  fillSpeakers(voice, false);
  noteChoice();
  $("info").disabled = !voice;
  updateControls();
}

// The Voice list's own change: another voice, so the filters go back to Any and its default speaker is chosen.
// refreshVoices() calls showVoice() instead, which keeps both (after a finished download or a failed job).
function onVoiceChange() {
  if ($("status").textContent.startsWith("Speaker is now ")) {
    setStatus("Ready", "ready"); // that note was about the voice being left
  }
  $("gender").value = "";
  $("accent").value = "";
  state.chosenSpeaker = null;
  showVoice();
}

// ---- Speakers (voice labels design §5): their labels, the Gender and Accent filters, and the chosen speaker.

const NO_MATCH = "No speaker matches these filters: set one back to Any";
const NOT_STATED = "none"; // a stated fact's value is "=" + the fact, so no gender or accent can be taken for it

function speakerInfo(voice, name) {
  return (voice.speaker_info && voice.speaker_info[name]) || { label: name, gender: null, accent: null };
}

function factValue(fact) {
  return fact === null || fact === undefined ? NOT_STATED : "=" + fact;
}

// The filters are for a voice with more than one speaker whose gender or accent is stated.
function hasFilters(voice) {
  if (!voice || voice.speakers.length <= 1) {
    return false;
  }
  const stated = voice.speakers.filter((name) => {
    const info = speakerInfo(voice, name);
    return info.gender !== null || info.accent !== null;
  });
  return stated.length > 1;
}

// One filter's options: Any, each stated value with its count (a value no speaker has never appears), then Not
// stated. The choice is kept when the new options still hold it.
function fillFilter(select, voice, key, order) {
  const counts = new Map();
  for (const name of voice ? voice.speakers : []) {
    const fact = speakerInfo(voice, name)[key];
    const k = fact === undefined ? null : fact;
    counts.set(k, (counts.get(k) || 0) + 1);
  }
  const stated = order(
    Array.from(counts.keys()).filter((fact) => fact !== null),
    counts,
  );
  const items = [option("", "Any"), ...stated.map((fact) => option("=" + fact, fact + " (" + counts.get(fact) + ")"))];
  if (counts.has(null)) {
    items.push(option(NOT_STATED, "Not stated (" + counts.get(null) + ")"));
  }
  const kept = select.value;
  select.replaceChildren(...items);
  select.value = items.some((item) => item.value === kept) ? kept : "";
}

function showFilters(voice) {
  const shown = hasFilters(voice);
  $("filter-row").hidden = !shown;
  const genders = (facts) => ["Female", "Male"].filter((g) => facts.includes(g));
  const byCount = (facts, counts) => facts.sort((a, b) => counts.get(b) - counts.get(a) || (a < b ? -1 : a > b ? 1 : 0));
  fillFilter($("gender"), shown ? voice : null, "gender", genders);
  fillFilter($("accent"), shown ? voice : null, "accent", byCount);
}

function matchesFilters(voice, name) {
  const info = speakerInfo(voice, name);
  const gender = $("gender").value;
  const accent = $("accent").value;
  return (gender === "" || factValue(info.gender) === gender) && (accent === "" || factValue(info.accent) === accent);
}

// The Speaker list: the speakers both filters match, by their labels, featured first. It is disabled for a voice
// with one speaker or none, never because the filters leave one match. The chosen speaker stays while it matches;
// otherwise the first match is chosen, and after a filter change the status line says so, since Speak would
// otherwise use a speaker nobody picked. With no match the chosen speaker is kept for when a filter goes back.
function fillSpeakers(voice, filterChanged) {
  const list = $("speaker");
  const names = voice ? voice.speakers : [];
  list.disabled = names.length <= 1;
  if (names.length === 0) {
    state.chosenSpeaker = null;
    list.replaceChildren(...(voice && voice.reader_label ? [option("", voice.reader_label)] : []));
    return;
  }
  const shown = hasFilters(voice) ? names.filter((name) => matchesFilters(voice, name)) : names;
  if (shown.length === 0) {
    const none = option("", "No speaker matches");
    none.disabled = true;
    list.replaceChildren(none);
    none.selected = true; // a list whose one entry is disabled would otherwise show no entry at all
    return;
  }
  const item = (name) => option(name, speakerInfo(voice, name).label);
  if (voice.featured.length === 0) {
    list.replaceChildren(...shown.map(item));
  } else {
    const groups = [
      ["Featured", shown.filter((name) => voice.featured.includes(name))],
      ["More speakers", shown.filter((name) => !voice.featured.includes(name))],
    ];
    list.replaceChildren(
      ...groups
        .filter(([, members]) => members.length > 0)
        .map(([title, members]) => {
          const group = document.createElement("optgroup");
          group.label = title;
          group.append(...members.map(item));
          return group;
        }),
    );
  }
  const old = state.chosenSpeaker;
  const pick = old !== null && shown.includes(old) ? old : shown[0];
  list.value = pick;
  state.chosenSpeaker = pick;
  if (filterChanged && old !== null && pick !== old) {
    setStatus("Speaker is now " + pick + ": the filters hide " + old, "ready");
  }
}

function onFilterChange() {
  fillSpeakers(currentVoice(), true);
  noteChoice();
  updateControls();
}

function onSpeakerChange() {
  state.chosenSpeaker = $("speaker").value;
  noteChoice();
  updateControls();
}

// The speaker Speak and Preview send: the chosen name whenever the voice has speakers, else null. Null for a voice
// with speakers means the filters match none, and then nothing may be sent: the server would use its default.
function selectedSpeaker(voice) {
  if (!voice || voice.speakers.length === 0) {
    return null;
  }
  const name = $("speaker").value;
  return voice.speakers.includes(name) ? name : null;
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
  setStatus((voice.state === "needs-repair" ? "Repairing " : "Downloading ") + voice.name, "busy");
  pollDownload(voice.id);
}

// A voice's short name (`English · VCTK`) for status lines, by its id; the id itself for a voice no longer listed.
function voiceName(id) {
  const voice = state.voices.find((v) => v.id === id);
  return voice ? voice.name : id;
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
    setStatus(voiceName(id) + " is ready", "ready");
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
  $("info-title").textContent = voice.label;
  $("info-id").textContent = voice.id;
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
  const running = state.job !== null || state.fetchingAudio;
  const noMatch = voice !== null && voice.speakers.length > 0 && selectedSpeaker(voice) === null;
  $("speak").disabled =
    running || state.waiting || !voice || voice.state !== "downloaded" || n === 0 || over || noMatch;
  const plan = previewPlan(voice);
  $("preview").hidden = plan === null;
  // Playing a clip needs no CPU slot: only a live Preview waits for a running task or a waiting Speak.
  $("preview").disabled =
    state.sampleButton !== $("preview") && (noMatch || (plan === "live" && (running || state.waiting)));
  $("stop").disabled =
    !running && !state.waiting && $("player").paused && sampler.paused && state.sampleAbort === null;
  $("save").disabled = !state.recording || running;
  if (over) {
    setStatus("Remove " + count(n - textLimit(), "character") + " to speak", "error");
    state.overLimitShown = true;
  } else if (noMatch) {
    if (state.overLimitShown || !state.noMatchShown) {
      setStatus(NO_MATCH, "error");
    }
    state.overLimitShown = false;
  } else if (state.overLimitShown || (state.noMatchShown && $("status").textContent === NO_MATCH)) {
    state.overLimitShown = false;
    setStatus("Ready", "ready");
  }
  state.noMatchShown = noMatch;
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

// What a recording, and the file Save writes, is named after: the voice's name with a space for its separator
// (" · " is odd in a file name), and the speaker when one was sent: "English VCTK p253".
function recordingBase(voice, speaker) {
  const name = voice.name.split(" · ").join(" ");
  return speaker === null ? name : name + " " + speaker;
}

function recordingName(base) {
  const now = new Date();
  const two = (n) => String(n).padStart(2, "0");
  return base + " " + two(now.getHours()) + "-" + two(now.getMinutes()) + "-" + two(now.getSeconds());
}

// The Speak button. It asks first when the last recording is unsaved; speak() itself never asks, because
// waitThenSpeak() calls it again after a busy wait, and a question there would come twice or after the choice.
function onSpeakClick() {
  if (state.unsaved) {
    $("replace-dialog").returnValue = ""; // Esc closes without a choice and would otherwise keep the last one
    $("replace-dialog").showModal();
  } else {
    speak();
  }
}

async function speak() {
  const voice = currentVoice();
  if (!voice) {
    return;
  }
  if (voice.speakers.length > 0 && selectedSpeaker(voice) === null) {
    state.waiting = false; // a filter changed while the Speak waited, and now matches no speaker
    setStatus(NO_MATCH, "error");
    updateControls();
    return;
  }
  const request = {
    text: $("text").value,
    voice: voice.id,
    speaker: selectedSpeaker(voice),
    speed: Number($("speed").value),
    device: $("gpu").checked ? "cuda" : "cpu",
    threads: threadsValue(),
  };
  try {
    const started = await api("/jobs", { method: "POST", json: request });
    state.waiting = false;
    state.job = started.id;
    state.jobKind = "speak";
    state.jobNote = null;
    setStatus("Generating: sentence 1", "busy");
    $("progress").textContent = "";
    updateControls();
    pollJob(started.id, recordingBase(voice, request.speaker));
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

async function pollJob(id, recordingBase) {
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
    setStatus(state.jobNote || "Generating: sentence " + (s.sentences_done + 1), "busy");
    setTimeout(() => pollJob(id, recordingBase), POLL_MS);
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
  // The server keeps only the newest job's audio, so a job started before this fetch ends would lose it.
  state.fetchingAudio = true;
  try {
    const response = await api("/jobs/" + id + "/audio", { raw: true });
    const blob = await response.blob();
    if (state.jobKind === "preview") {
      await playPreview(blob); // to the sampler: a Preview never becomes the recording
    } else {
      loadRecording(blob, recordingName(recordingBase));
      const done = s.state === "stopped" ? "Stopped: kept " + count(s.sentences_done, "sentence") : "Done";
      setStatus([done, ...s.warnings].join(". "), "ready");
      $("player").play().catch(() => {});
    }
  } catch (e) {
    setStatus(e.message, "error");
  } finally {
    state.fetchingAudio = false;
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
  stopSample();
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

// ---- Samples (voice labels design §5): recorded clips, played by a sampler that is not #player. A sample never
// touches the recording: it comes through api() with the secret as a blob, plays from a blob URL, and is let go after.

const sampler = new Audio();
const SAMPLE_TIMEOUT_MS = 30000; // the server can't bound a slow fetch, so the page gives up on it here
const FETCHING_SAMPLE = "Fetching the sample";
const SAMPLE_TIMED_OUT = "could not fetch the sample: timed out";

// Plays a voice's recorded clip: the default speaker's when `speaker` is null. Only the newest press plays: each
// press takes a token, and a reply that comes after a later press, a voice or speaker change, or Stop is dropped.
// Resolves false only when this press's clip could not be fetched or played (so ▶ Preview can speak it live), and
// true when it plays or something later replaced it.
async function playClip(voice, speaker, button) {
  stopSample();
  const token = state.sampleToken;
  const controller = new AbortController();
  state.sampleAbort = controller;
  const timer = setTimeout(() => controller.abort(), SAMPLE_TIMEOUT_MS);
  setStatus(FETCHING_SAMPLE, "busy");
  updateControls();
  let blob;
  try {
    let path = "/voices/" + encodeURIComponent(voice.id) + "/sample";
    if (speaker !== null) {
      path += "?speaker=" + encodeURIComponent(speaker);
    }
    const response = await api(path, { raw: true, signal: controller.signal });
    blob = await response.blob(); // the time limit can land here too, after api() has answered
  } catch (e) {
    if (token !== state.sampleToken) {
      return true;
    }
    state.sampleAbort = null;
    setStatus(e.name === "AbortError" ? SAMPLE_TIMED_OUT : e.message, "error");
    updateControls();
    return false;
  } finally {
    clearTimeout(timer);
  }
  if (token !== state.sampleToken) {
    return true;
  }
  state.sampleAbort = null;
  return playBlob(blob, { voice: voice.id, speaker }, button, token);
}

// Plays a clip, or a live Preview's audio, on the sampler while `token` is still the newest press. Resolves false
// when it won't play.
async function playBlob(blob, playing, button, token) {
  if (token !== state.sampleToken) {
    return true;
  }
  sampler.pause();
  releaseSample();
  state.sampleUrl = URL.createObjectURL(blob);
  sampler.src = state.sampleUrl;
  state.samplePlaying = playing;
  state.sampleButton = button;
  showSampleButtons();
  $("player").pause();
  try {
    await sampler.play();
  } catch (e) {
    if (token !== state.sampleToken) {
      return true; // stopped before it started
    }
    releaseSample();
    setStatus("could not play the sample: " + e.message, "error");
    updateControls();
    return false;
  }
  if (token === state.sampleToken) {
    setStatus("Ready", "ready");
  }
  updateControls();
  return true;
}

// Stops a playing sample and drops a pending one: Stop, a voice or speaker change, #player starting, a new press.
function stopSample() {
  state.sampleToken += 1;
  if (state.sampleAbort !== null) {
    state.sampleAbort.abort();
    state.sampleAbort = null;
  }
  // Not only while a fetch is pending: also between a clip's arrival and its start.
  if ($("status").textContent === FETCHING_SAMPLE) {
    setStatus("Ready", "ready");
  }
  sampler.pause();
  releaseSample();
}

// Lets go of the sampler's clip (its blob URL is revoked), and the button that started it reads ▶ again.
function releaseSample() {
  if (state.sampleUrl !== null) {
    sampler.removeAttribute("src");
    sampler.load();
    URL.revokeObjectURL(state.sampleUrl);
    state.sampleUrl = null;
  }
  state.samplePlaying = null;
  state.sampleButton = null;
  showSampleButtons();
}

// While its clip plays, the button that started it reads ■ Stop, and its name says what pressing it now does.
function showSampleButtons() {
  const name = (button, playing, text, label, stopLabel) => {
    button.textContent = playing ? "■ Stop" : text;
    button.setAttribute("aria-label", playing ? stopLabel : label);
  };
  name($("sample"), state.sampleButton === $("sample"), "▶ Sample", "Play a sample of this voice", "Stop the sample");
  name($("preview"), state.sampleButton === $("preview"), "▶ Preview", "Preview this speaker", "Stop the preview");
}

// A different voice or speaker stops a playing sample and drops a pending one. A refresh that keeps both (a finished
// download, a failed job) leaves a sample alone, and so does a filter change that keeps the chosen speaker.
function noteChoice() {
  const voice = currentVoice();
  const choice = voice ? voice.id + "\n" + (selectedSpeaker(voice) ?? "") : "";
  if (choice !== state.choice) {
    state.choice = choice;
    stopSample();
  }
}

// ▶ Sample: the default speaker's recorded clip, before or after download; never spoken live.
function onSampleClick() {
  const voice = currentVoice();
  if (state.sampleButton === $("sample")) {
    stopSample();
  } else if (voice && voice.has_sample) {
    playClip(voice, null, $("sample"));
  }
}


// ---- ▶ Preview (voice labels design §5). Before download it plays the chosen speaker's clip. On a downloaded voice it
// always works: the clip at Speed 1.00, otherwise (or when the clip can't be fetched) the language's sentence spoken
// live as a job, with the current Speed, GPU and Threads. A live Preview never queues, and its audio goes to the
// sampler, never to the recording.

const PREVIEW_FELL_BACK = "Couldn't fetch the sample, so speaking it live";
const PREVIEW_BUSY = "Wait for the running task to finish, then press Preview again";

function speedIsOne() {
  return Math.abs(Number($("speed").value) - 1) < 1e-9;
}

// What ▶ Preview does for the voice and speaker in view: "clip", "live", or null to hide it.
function previewPlan(voice) {
  if (!voice) {
    return null;
  }
  const speaker = selectedSpeaker(voice);
  const clip =
    voice.speakers.length === 0 ? voice.has_sample : speaker !== null && voice.speakers_with_samples.includes(speaker);
  if (voice.state !== "downloaded" || !voice.sample_text) {
    return clip ? "clip" : null;
  }
  return clip && speedIsOne() ? "clip" : "live";
}

async function onPreviewClick() {
  const voice = currentVoice();
  if (state.sampleButton === $("preview")) {
    stopSample();
    return;
  }
  const plan = previewPlan(voice);
  const speaker = selectedSpeaker(voice);
  if (plan === null || (voice.speakers.length > 0 && speaker === null)) {
    return;
  }
  if (plan === "live") {
    previewLive(voice, speaker, false);
    return;
  }
  const played = await playClip(voice, speaker, $("preview"));
  if (!played && voice.state === "downloaded" && voice.sample_text) {
    previewLive(voice, speaker, true);
  }
}

// A live Preview: a job like Speak's, so it takes the CPU slot and Stop stops it; on a busy CPU it says so and stops
// there. `fellBack` when the clip couldn't be fetched: the status says why it is speaking live.
async function previewLive(voice, speaker, fellBack) {
  if (!fellBack) {
    stopSample();
  }
  const token = state.sampleToken;
  const request = {
    text: voice.sample_text,
    voice: voice.id,
    speaker,
    speed: Number($("speed").value),
    device: $("gpu").checked ? "cuda" : "cpu",
    threads: threadsValue(),
  };
  try {
    const started = await api("/jobs", { method: "POST", json: request });
    state.job = started.id;
    state.jobKind = "preview";
    state.jobNote = fellBack ? PREVIEW_FELL_BACK : null;
    state.preview = { token, voice: voice.id, speaker };
    setStatus(state.jobNote || "Generating: sentence 1", "busy");
    updateControls();
    pollJob(started.id, recordingBase(voice, speaker)); // a Preview's audio is never a recording
  } catch (e) {
    setStatus(e.status === 409 ? PREVIEW_BUSY : e.message, "error");
    updateControls();
  }
}

// A finished live Preview's audio: played on the sampler unless something was pressed or changed since.
async function playPreview(blob) {
  const preview = state.preview;
  state.preview = null;
  if (preview === null || preview.token !== state.sampleToken) {
    setStatus("Ready", "ready");
    return;
  }
  await playBlob(blob, { voice: preview.voice, speaker: preview.speaker }, $("preview"), preview.token);
}

// ---- End of samples.

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
  $("voice").addEventListener("change", onVoiceChange);
  $("gender").addEventListener("change", onFilterChange);
  $("accent").addEventListener("change", onFilterChange);
  $("speaker").addEventListener("change", onSpeakerChange);
  $("download").addEventListener("click", startDownload);
  $("repair").addEventListener("click", startDownload);
  $("cancel").addEventListener("click", cancelDownload);
  $("info").addEventListener("click", showInfo);
  $("copy-attribution").addEventListener("click", copyAttribution);
  $("speed").addEventListener("input", () => {
    $("speed-value").textContent = Number($("speed").value).toFixed(2);
    updateControls(); // Preview plays the clip only at 1.00
  });
  $("speed-reset").addEventListener("click", () => {
    $("speed").value = "1";
    $("speed-value").textContent = "1.00";
    updateControls();
  });
  $("threads").addEventListener("change", threadsValue);
  $("text").addEventListener("input", updateCounter);
  $("file").addEventListener("change", openTextFile);
  $("clear").addEventListener("click", () => {
    $("text").value = "";
    updateCounter();
    $("text").focus();
  });
  $("speak").addEventListener("click", onSpeakClick);
  $("replace-dialog").addEventListener("close", () => {
    if ($("replace-dialog").returnValue === "replace") {
      speak();
    }
  });
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
    sampler.addEventListener(event, updateControls);
  }
  $("player").addEventListener("play", stopSample); // a recording starting, or a finished Speak
  sampler.addEventListener("ended", releaseSample);
  $("sample").addEventListener("click", onSampleClick);
  $("preview").addEventListener("click", onPreviewClick);
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
