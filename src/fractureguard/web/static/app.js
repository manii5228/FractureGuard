"use strict";

document.querySelectorAll("[data-preview]").forEach((input) => {
  const image = document.getElementById(input.dataset.preview);
  let previous;
  input.addEventListener("change", () => {
    if (previous) URL.revokeObjectURL(previous);
    const file = input.files[0];
    previous = file && file.size <= 10 * 1024 * 1024 ? URL.createObjectURL(file) : null;
    image.hidden = !previous;
    if (previous) image.src = previous;
    else image.removeAttribute("src");
  });
});

const upload = document.querySelector("[data-upload]");
if (upload) upload.addEventListener("submit", () => {
  document.getElementById("upload-progress").textContent = "Submitting both views…";
});

const statusLabel = document.querySelector("[data-status-url]");
if (statusLabel) {
  async function updateStatus() {
    try {
      const response = await fetch(statusLabel.dataset.statusUrl, {cache: "no-store"});
      if (!response.ok) return;
      const current = await response.json();
      statusLabel.textContent = current.label;
      statusLabel.className = `state state-${current.status}`;
      if (["queued", "processing"].includes(current.status)) setTimeout(updateStatus, 2000);
      else {
        const refresh = document.createElement("a");
        refresh.href = location.pathname;
        refresh.textContent = "Refresh case details";
        statusLabel.after(refresh);
      }
    } catch {
      setTimeout(updateStatus, 5000);
    }
  }
  setTimeout(updateStatus, 1300);
}

const review = document.querySelector("[data-review]");
if (review) {
  const boxes = JSON.parse(document.getElementById("review-boxes").textContent);
  const choice = document.getElementById("box-choice");
  const viewInput = document.getElementById("box-view");
  const hidden = document.getElementById("boxes");
  const dimensions = new Map();
  const cards = [...review.querySelectorAll("[data-view]")];
  const switches = [...review.querySelectorAll("[data-switch]")];
  let selected = "";

  function showView(name) {
    cards.forEach((card) => card.toggleAttribute("data-current", card.dataset.view === name));
    switches.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.switch === name)));
  }

  function draw() {
    if (hidden) hidden.value = JSON.stringify(boxes);
    cards.forEach((card) => {
      const group = card.querySelector("[data-overlays]");
      group.replaceChildren();
      boxes.filter((box) => box.view === card.dataset.view && box.status !== "rejected").forEach((box) => {
        const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        const [x1, y1, x2, y2] = box.coords;
        rect.setAttribute("x", x1);
        rect.setAttribute("y", y1);
        rect.setAttribute("width", x2 - x1);
        rect.setAttribute("height", y2 - y1);
        rect.setAttribute("class", box.status + (box.id === selected ? " selected" : ""));
        group.append(rect);
      });
    });
  }

  switches.forEach((button) => button.addEventListener("click", () => showView(button.dataset.switch)));
  cards.forEach((card) => {
    const {width, height} = card.querySelector("svg").viewBox.baseVal;
    dimensions.set(card.dataset.view, {width, height});
    const group = card.querySelector("[data-image]");
    const overlays = card.querySelector("[data-overlays]");
    let scale = 1;
    let x = 0;
    let y = 0;
    card.querySelectorAll("[data-tool]").forEach((button) => button.addEventListener("click", () => {
      switch (button.dataset.tool) {
        case "in": scale = Math.min(5, scale * 1.25); break;
        case "out": scale = Math.max(1, scale / 1.25); break;
        case "left": x -= width * .08; break;
        case "right": x += width * .08; break;
        case "up": y -= height * .08; break;
        case "down": y += height * .08; break;
        case "reset": scale = 1; x = 0; y = 0; break;
        case "overlays": {
          const visible = button.getAttribute("aria-pressed") === "true";
          overlays.setAttribute("visibility", visible ? "hidden" : "visible");
          button.setAttribute("aria-pressed", String(!visible));
          break;
        }
      }
      x = Math.max(-width * scale * .48, Math.min(width * scale * .48, x));
      y = Math.max(-height * scale * .48, Math.min(height * scale * .48, y));
      group.setAttribute("transform", `translate(${x + width * (1 - scale) / 2} ${y + height * (1 - scale) / 2}) scale(${scale})`);
    }));
  });

  if (choice) {
    const message = document.getElementById("box-message");
    const fields = ["x1", "y1", "x2", "y2"].map((coordinate) => document.getElementById(`box-${coordinate}`));
    function refreshOptions() {
      choice.replaceChildren(new Option("New box", ""));
      boxes.forEach((box, index) => {
        const label = box.view === "frontal" ? "AP/PA" : "Lateral";
        choice.add(new Option(`${index + 1}. ${label} · ${box.status}`, box.id));
      });
      choice.value = selected;
      draw();
    }
    function updateHelp() {
      const {width, height} = dimensions.get(viewInput.value);
      document.getElementById("box-help").textContent = `Original image: ${width} × ${height} pixels. Approved boxes appear on the report.`;
    }
    viewInput.addEventListener("change", () => { showView(viewInput.value); updateHelp(); });
    choice.addEventListener("change", () => {
      selected = choice.value;
      const box = boxes.find((entry) => entry.id === selected);
      fields.forEach((field, index) => { field.value = box ? box.coords[index] : ""; });
      viewInput.disabled = Boolean(box);
      if (box) viewInput.value = box.view;
      showView(viewInput.value);
      updateHelp();
      draw();
    });
    review.querySelectorAll("[data-box-action]").forEach((button) => button.addEventListener("click", () => {
      const action = button.dataset.boxAction;
      const box = boxes.find((entry) => entry.id === selected);
      if (action === "rejected" && box) box.status = "rejected";
      else if (action === "rejected" || (action === "update" && !box)) {
        message.textContent = "Select an existing box first.";
        return;
      } else {
        const coords = fields.map((field) => field.valueAsNumber);
        const [x1, y1, x2, y2] = coords;
        const {width, height} = dimensions.get(viewInput.value);
        if (!coords.every(Number.isFinite) || x1 < 0 || y1 < 0 || x2 > width || y2 > height || x1 >= x2 || y1 >= y2) {
          message.textContent = "Enter coordinates within the image. Right and bottom must exceed left and top.";
          return;
        }
        if (box) {
          box.coords = coords;
          if (action === "approved") box.status = "approved";
        } else {
          if (boxes.length >= 120) {
            message.textContent = "This review already has 120 annotations.";
            return;
          }
          selected = `manual-${crypto.randomUUID()}`;
          boxes.push({id: selected, view: viewInput.value, status: "approved", coords});
          viewInput.disabled = true;
        }
      }
      message.textContent = "Annotation updated. Save findings to keep this change.";
      refreshOptions();
    }));
    updateHelp();
    refreshOptions();
  } else draw();
}
