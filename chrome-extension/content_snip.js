(() => {
  const EXISTING = document.getElementById("__lao_ocr_snip_overlay");
  if (EXISTING) EXISTING.remove();

  const overlay = document.createElement("div");
  overlay.id = "__lao_ocr_snip_overlay";
  Object.assign(overlay.style, {
    position: "fixed",
    inset: "0",
    zIndex: "2147483647",
    background: "rgba(15, 23, 42, 0.28)",
    cursor: "crosshair",
    userSelect: "none"
  });

  const hint = document.createElement("div");
  hint.textContent = "ลากครอบข้อความที่ต้องการอ่าน (กด Esc เพื่อยกเลิก)";
  Object.assign(hint.style, {
    position: "fixed",
    top: "16px",
    left: "50%",
    transform: "translateX(-50%)",
    background: "#1e3a8a",
    color: "#ffffff",
    padding: "8px 16px",
    borderRadius: "999px",
    fontFamily: "system-ui, sans-serif",
    fontSize: "13px",
    fontWeight: "600",
    boxShadow: "0 6px 20px rgba(0,0,0,0.3)",
    pointerEvents: "none"
  });
  overlay.appendChild(hint);

  const box = document.createElement("div");
  Object.assign(box.style, {
    position: "fixed",
    border: "2px solid #38bdf8",
    background: "rgba(56, 189, 248, 0.14)",
    borderRadius: "4px",
    display: "none",
    pointerEvents: "none"
  });
  overlay.appendChild(box);
  document.documentElement.appendChild(overlay);

  let sx = 0, sy = 0, dragging = false;

  const cleanup = () => {
    window.removeEventListener("keydown", onKey, true);
    overlay.remove();
  };

  const onKey = (e) => {
    if (e.key === "Escape") cleanup();
  };
  window.addEventListener("keydown", onKey, true);

  overlay.addEventListener("mousedown", (e) => {
    dragging = true;
    sx = e.clientX;
    sy = e.clientY;
    Object.assign(box.style, { left: sx + "px", top: sy + "px", width: "0px", height: "0px", display: "block" });
  });

  overlay.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const x = Math.min(sx, e.clientX);
    const y = Math.min(sy, e.clientY);
    const w = Math.abs(e.clientX - sx);
    const h = Math.abs(e.clientY - sy);
    Object.assign(box.style, { left: x + "px", top: y + "px", width: w + "px", height: h + "px" });
  });

  overlay.addEventListener("mouseup", (e) => {
    if (!dragging) return;
    dragging = false;
    const x = Math.min(sx, e.clientX);
    const y = Math.min(sy, e.clientY);
    const w = Math.abs(e.clientX - sx);
    const h = Math.abs(e.clientY - sy);
    cleanup();
    if (w < 8 || h < 8) return;
    // Wait 1 animation frame so overlay disappears before captureVisibleTab
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        chrome.runtime.sendMessage({
          type: "SNIP_RECT",
          rect: { x, y, w, h },
          dpr: window.devicePixelRatio || 1
        });
      });
    });
  });
})();
