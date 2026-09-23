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
    composer.requestSubmit();
  }
});

composer.addEventListener("htmx:beforeRequest", () => {
  stickToBottom = true;
});

composer.addEventListener("reset", () => {
  // The value is cleared after this event, so resize on the next frame.
  requestAnimationFrame(resizeInput);
});

// --- sidebar drawer on small screens ---

document.querySelectorAll("[data-sidebar-toggle]").forEach((element) => {
  element.addEventListener("click", () => document.body.classList.toggle("sidebar-open"));
});
