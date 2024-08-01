chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
  chrome.tabs.sendMessage(tabs[0].id, { type: "OPEN_FLOATING_WINDOW" });
  window.close();
});