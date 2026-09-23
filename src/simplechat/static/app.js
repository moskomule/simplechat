// Small UI behaviours that htmx does not cover.

const scroller = document.getElementById("scroller");
const composer = document.getElementById("composer");
const input = composer.querySelector("textarea");
const isTouch = window.matchMedia("(pointer: coarse)").matches;

// --- keep the thread scrolled to the bottom unless the user scrolled up ---

let stickToBottom = true;

function scrollToBottom() {
  scroller.scrollTop = scroller.scrollHeight;
}

scroller.addEventListener("scroll", () => {
  const distance = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight;
  stickToBottom = distance < 80;
});

document.body.addEventListener("htmx:sseMessage", () => {
  if (stickToBottom) scrollToBottom();
});

document.body.addEventListener("htmx:afterSwap", (event) => {
  if (event.detail.target === document.getElementById("thread") && stickToBottom) {
    scrollToBottom();
  }
});

// The thread's viewport shrinks when the composer grows (longer text, image
// previews) or the top bar opens a panel; stay at the bottom if we were there.
new ResizeObserver(() => {
  if (stickToBottom) scrollToBottom();
}).observe(scroller);

scrollToBottom();

// --- composer: auto-grow, and Enter to send on devices with a keyboard ---

function resizeInput() {
  input.style.height = "auto";
  input.style.height = `${input.scrollHeight + 2}px`;
}

input.addEventListener("input", resizeInput);

input.addEventListener("keydown", (event) => {
  // isComposing: Enter confirms IME input (e.g. Japanese) and must not send.
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing && !isTouch) {
    event.preventDefault();
    if (!isBusy()) composer.requestSubmit();
  }
});

composer.addEventListener("htmx:beforeRequest", () => {
  stickToBottom = true;
});

composer.addEventListener("reset", () => {
  // The value is cleared after this event, so resize on the next frame.
  requestAnimationFrame(resizeInput);
});

// --- composer: attach images from the picker or by pasting (vision models only) ---

// The selected files live in `attachments`. After every change they are written
// back to the file input with a DataTransfer, so the normal htmx submit sends them.
// The limits match the server's, which checks them again.

const IMAGE_TYPES = ["image/png", "image/jpeg", "image/webp", "image/gif"];
const MAX_IMAGES = 10;
const MAX_IMAGE_BYTES = 20 * 1024 * 1024;

const fileInput = document.getElementById("attach-input");
const previews = document.getElementById("attachments");
let attachments = [];

// The attach button is replaced out of band when the model changes, so look it up each time.
function canAttach() {
  const button = document.getElementById("attach-button");
  return button !== null && !button.hidden;
}

function addAttachments(files) {
  const problems = [];
  for (const file of files) {
    if (!IMAGE_TYPES.includes(file.type)) {
      problems.push(`${file.name}: only PNG, JPEG, WebP and GIF images can be sent.`);
    } else if (file.size > MAX_IMAGE_BYTES) {
      problems.push(`${file.name}: images must be 20 MB or smaller.`);
    } else if (attachments.length >= MAX_IMAGES) {
      problems.push(`${file.name}: at most ${MAX_IMAGES} images can be sent at once.`);
    } else {
      attachments.push(file);
    }
  }
  if (problems.length > 0) alert(problems.join("\n"));
  renderAttachments();
}

function renderAttachments() {
  const transfer = new DataTransfer();
  attachments.forEach((file) => transfer.items.add(file));
  fileInput.files = transfer.files;
  // Images alone make a valid message.
  input.required = attachments.length === 0;

  previews.querySelectorAll("img").forEach((image) => URL.revokeObjectURL(image.src));
  previews.replaceChildren(...attachments.map(attachmentPreview));
}

function attachmentPreview(file, index) {
  const image = document.createElement("img");
  image.src = URL.createObjectURL(file);
  image.alt = file.name;

  const remove = document.createElement("button");
  remove.className = "remove-image";
  remove.type = "button";
  remove.setAttribute("aria-label", `Remove ${file.name}`);
  remove.textContent = "×";
  remove.addEventListener("click", () => {
    attachments.splice(index, 1);
    renderAttachments();
  });

  const item = document.createElement("div");
  item.className = "attachment";
  item.append(image, remove);
  return item;
}

document.body.addEventListener("click", (event) => {
  if (event.target.closest("#attach-button")) fileInput.click();
});

fileInput.addEventListener("change", () => addAttachments([...fileInput.files]));

input.addEventListener("paste", (event) => {
  const images = [...event.clipboardData.files].filter((file) => file.type.startsWith("image/"));
  if (images.length === 0 || !canAttach()) return;
  event.preventDefault();
  addAttachments(images);
});

// Cleared after a successful send (the form is reset on htmx:afterRequest).
composer.addEventListener("reset", () => {
  attachments = [];
  renderAttachments();
});

// Switching to a model without vision hides the button and drops the selection.
document.body.addEventListener("htmx:afterSettle", () => {
  if (attachments.length > 0 && !canAttach()) {
    attachments = [];
    renderAttachments();
  }
});

// --- edit form: × drops an image; the text may be empty while images remain ---

document.body.addEventListener("click", (event) => {
  const remove = event.target.closest(".edit-form .remove-image");
  if (!remove) return;
  const form = remove.closest(".edit-form");
  remove.closest(".attachment").remove();
  form.querySelector("textarea").required = form.querySelector(".attachment") === null;
});

// --- no new requests while a reply is being generated ---

// The server refuses them anyway (409); this keeps the controls from looking or
// acting usable. `inert` blocks mouse, touch and keyboard, unlike pointer-events.

function isBusy() {
  return document.querySelector(".msg.pending, .msg.streaming") !== null;
}

function syncBusyState() {
  const busy = isBusy();
  document.querySelectorAll(".msg .actions, .composer .send").forEach((element) => {
    element.inert = busy;
  });
}

// Fires after every swap, including new turns, thread redraws and the final
// `done` swap that replaces a streamed message.
document.body.addEventListener("htmx:afterSettle", syncBusyState);
syncBusyState();

// --- system prompt: saved only with the Save button or Ctrl/Cmd+Enter ---

// The textarea's defaultValue (its server-rendered text) is the saved prompt, so
// "unsaved" means the value differs from it. The textarea is never swapped out,
// which keeps the panel open and the cursor in place.

const promptForm = document.getElementById("system-prompt-form");
const promptInput = promptForm.querySelector("textarea");
const promptSave = promptForm.querySelector("button[type=submit]");
const promptStatus = promptForm.querySelector(".save-status");
const promptUnsavedMark = document.querySelector(".system-prompt .unsaved-mark");
// One save at a time: while one is in flight, Save stays disabled even if the user
// keeps typing, so `sentPrompt` always belongs to the request that is answering.
let saving = false;
let sentPrompt = "";
let promptStatusTimer;

function showPromptStatus(text, clearAfterMs = 0) {
  clearTimeout(promptStatusTimer);
  promptStatus.textContent = text;
  if (clearAfterMs) promptStatusTimer = setTimeout(() => showPromptStatus(""), clearAfterMs);
}

function syncPromptState() {
  const unsaved = promptInput.value !== promptInput.defaultValue;
  promptSave.disabled = saving || !unsaved;
  promptUnsavedMark.hidden = !unsaved;
}

promptInput.addEventListener("input", () => {
  showPromptStatus("");
  syncPromptState();
});

promptInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey) && !event.isComposing) {
    event.preventDefault();
    if (!promptSave.disabled) promptForm.requestSubmit();
  }
});

promptForm.addEventListener("htmx:beforeRequest", () => {
  sentPrompt = promptInput.value;
  // Disabling a focused button drops focus to <body>, so go back to the text first.
  if (document.activeElement === promptSave) promptInput.focus();
  saving = true;
  syncPromptState();
});

promptForm.addEventListener("htmx:afterRequest", (event) => {
  saving = false;
  if (event.detail.successful) {
    // The user may have kept typing, so the sent text, not the current one, is saved.
    promptInput.defaultValue = sentPrompt;
    if (promptInput.value === sentPrompt) showPromptStatus("Saved", 2000);
  } else {
    showPromptStatus("Not saved");
  }
  syncPromptState();
});

// The browser may restore unsaved text on reload.
syncPromptState();

// --- sidebar drawer on small screens ---

document.querySelectorAll("[data-sidebar-toggle]").forEach((element) => {
  element.addEventListener("click", () => document.body.classList.toggle("sidebar-open"));
});
