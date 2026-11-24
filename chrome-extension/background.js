// Open Side Panel automatically when clicking the toolbar icon
chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true }).catch(() => {});

// Context menu: right-click any image on any webpage -> OCR directly in Side Panel
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "lao-ocr-image",
    title: "🇱🇦 MaeBounme Lao-OCR (ອ່ານຂໍ້ຄວາມໃນຮູບ)",
    contexts: ["image"]
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === "lao-ocr-image" && info.srcUrl && tab?.id) {
    try {
      await chrome.sidePanel.open({ tabId: tab.id });
    } catch (_) {}
    setTimeout(() => {
      chrome.runtime.sendMessage({ type: "OCR_IMAGE_URL", url: info.srcUrl }).catch(() => {});
    }, 350);
  }
});

// Relay messages between Side Panel and active tab (screen snip capture)
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "START_SNIP") {
    chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
      const tab = tabs[0];
      if (!tab?.id) {
        sendResponse({ ok: false, error: "ไม่พบแท็บที่เปิดอยู่" });
        return;
      }
      try {
        await chrome.scripting.executeScript({
          target: { tabId: tab.id },
          files: ["content_snip.js"]
        });
        sendResponse({ ok: true });
      } catch (err) {
        sendResponse({ ok: false, error: "ไม่สามารถครอบภาพบนหน้านี้ได้ (เช่น หน้า chrome://) ให้ใช้ Ctrl+V แทนครับ" });
      }
    });
    return true;
  }

  if (msg.type === "SNIP_RECT" && sender.tab) {
    chrome.tabs.captureVisibleTab(sender.tab.windowId, { format: "png" }, (dataUrl) => {
      if (chrome.runtime.lastError || !dataUrl) {
        chrome.runtime.sendMessage({ type: "SNIP_ERROR", error: chrome.runtime.lastError?.message || "capture failed" }).catch(() => {});
        return;
      }
      chrome.runtime.sendMessage({
        type: "SNIP_CAPTURED",
        dataUrl,
        rect: msg.rect,
        dpr: msg.dpr || 1
      }).catch(() => {});
    });
  }
});
