// Add styles for CapFlow elements
function addStyles() {
  const style = document.createElement('style');
  style.textContent = `
    .capflow-dim-overlay {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background-color: rgba(0, 0, 0, 0.7);
      z-index: 9998;
      pointer-events: none;
      transition: opacity 0.3s ease;
    }
    .capflow-highlight-container {
      position: absolute;
      z-index: 9999;
      pointer-events: none;
      transition: all 0.3s ease;
      box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.8);
      overflow: hidden;
    }
    .capflow-highlight-inner {
      width: 100%;
      height: 100%;
      background-position: center;
      background-repeat: no-repeat;
      background-size: cover;
      filter: brightness(1.3);
    }
    @keyframes capflow-pulse {
      0%, 100% { box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.8); }
      50% { box-shadow: 0 0 0 4px rgba(52, 152, 219, 0.8); }
    }
    .capflow-highlight-loading {
      animation: capflow-pulse 1.5s infinite;
    }
    .capflow-spinner {
      border: 4px solid rgba(255, 255, 255, 0.3);
      border-top: 4px solid #ffffff;
      border-radius: 50%;
      width: 24px;
      height: 24px;
      animation: capflow-spin 1s linear infinite;
      margin: 10px auto;
    }
    @keyframes capflow-spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
  `;
  document.head.appendChild(style);
}

addStyles();

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "OPEN_FLOATING_WINDOW") {
    createFloatingWindow();
  } 
});

function createDimOverlay() {
  const overlay = document.createElement('div');
  overlay.className = 'capflow-dim-overlay';
  document.body.appendChild(overlay);
  overlay.offsetHeight; 
  overlay.style.opacity = '1';
}

function removeDimOverlay() {
  const overlay = document.querySelector('.capflow-dim-overlay');
  if (overlay) {
    overlay.style.opacity = '0';
    setTimeout(() => overlay.parentNode?.removeChild(overlay), 300);
  }
}

function createFloatingWindow() {
  const popup = document.createElement('div');
  popup.id = 'caption-popup';
  popup.style.cssText = `
    position: fixed;
    top: 20px;
    right: 20px;
    width: 350px;
    background-color: #2C3E50;
    color: white;
    padding: 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    z-index: 10000;
    border-radius: 12px;
    font-family: 'Roboto', sans-serif;`;

  popup.innerHTML = `
    <div id="popup-header" style="background-color: #34495E; padding: 15px; cursor: move; border-radius: 12px 12px 0 0; border-bottom: 1px solid #445566;">
      <span id="close-popup" style="position: absolute; right: 15px; top: 15px; color: white; opacity: 0.7; cursor: pointer; font-size: 20px;">✕</span>
      <h3 style="margin: 0; text-align: center; font-size: 18px; font-weight: 500; color: white;">CapFlow</h3>
    </div>
    <div id="content-area" style="padding: 20px;">
      <button id="start-button" style="
        background: linear-gradient(to bottom, #3498DB, #2980B9);
        color: white;
        border: none;
        padding: 12px 0;
        width: 100%;
        margin: 10px 0;
        display: block;
        border-radius: 8px;
        font-size: 16px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      ">Start</button>
      <p id="instruction-text" style="display: none; text-align: center; font-size: 16px; margin: 10px 0;">Click on an image</p>
      <div id="loading-spinner" class="capflow-spinner" style="display: none;"></div>
      <p id="caption-text" style="display: none; margin-top: 15px; font-size: 14px; line-height: 1.5; text-align: center;"></p>
    </div>`;

    document.body.appendChild(popup);

    const requiredElements = ['start-button', 'instruction-text', 'loading-spinner', 'caption-text'];
    requiredElements.forEach(id => {
      if (!document.getElementById(id)) {
        console.error(`Element with id '${id}' not found after creating floating window`);
      }
    });
  
    makeDraggable(popup);
    setupStartButton();
    setupCloseButton(popup);
}

function makeDraggable(popup) {
  const header = popup.querySelector('#popup-header');
  let isDragging = false;
  let offsetX, offsetY;

  header.addEventListener('mousedown', (e) => {
    isDragging = true;
    offsetX = e.clientX - popup.getBoundingClientRect().left;
    offsetY = e.clientY - popup.getBoundingClientRect().top;
  });

  document.addEventListener('mousemove', (e) => {
    if (isDragging) {
      popup.style.left = `${e.clientX - offsetX}px`;
      popup.style.top = `${e.clientY - offsetY}px`;
      popup.style.right = 'auto';
    }
  });

  document.addEventListener('mouseup', () => isDragging = false);
}

function setupStartButton() {
  const startButton = document.getElementById('start-button');
  const instructionText = document.getElementById('instruction-text');

  startButton.addEventListener('click', () => {
    if (startButton.textContent === "Start Again") {
      handleStartAgainClick();
    } else {
      startFlow();
    }
  });
}

function startFlow() {
  const startButton = document.getElementById('start-button');
  const instructionText = document.getElementById('instruction-text');
  
  startButton.style.display = 'none';
  instructionText.style.display = 'block';
  createDimOverlay();
  resetUI();
}

function setupCloseButton(popup) {
  document.getElementById('close-popup').addEventListener('click', () => {
    document.body.removeChild(popup);
    removeDimOverlay();
    resetUI();
  });
}

function resetUI() {
  const previousHighlight = document.querySelector('.capflow-highlight-container');
  if (previousHighlight) {
    previousHighlight.remove();
  }
  
  const captionText = document.getElementById('caption-text');
  if (captionText) {
    captionText.style.display = 'none';
  }
  
  const hashtagContainer = document.getElementById('hashtag-container');
  if (hashtagContainer) {
    hashtagContainer.remove();
  }
  
  const loadingSpinner = document.getElementById('loading-spinner');
  if (loadingSpinner) {
    loadingSpinner.style.display = 'none';
  }
}

function captureImage(img, maxWidth = 1024, quality = 0.9) {
  return new Promise((resolve) => {
    const canvas = document.createElement('canvas');
    let width = img.naturalWidth;
    let height = img.naturalHeight;

    // Resize if the image is too large
    if (width > maxWidth) {
      height = Math.round((height * maxWidth) / width);
      width = maxWidth;
    }

    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext('2d');
    
    try {
      ctx.drawImage(img, 0, 0, width, height);
      resolve(canvas.toDataURL('image/jpeg', quality));
    } catch (error) {
      resolve(img.src);
    }
  });
}

function createHighlightOverlay(img) {
  const rect = img.getBoundingClientRect();
  const overlay = document.createElement('div');
  overlay.className = 'capflow-highlight-container';
  overlay.style.top = `${rect.top + window.scrollY}px`;
  overlay.style.left = `${rect.left + window.scrollX}px`;
  overlay.style.width = `${rect.width}px`;
  overlay.style.height = `${rect.height}px`;
  
  const innerOverlay = document.createElement('div');
  innerOverlay.className = 'capflow-highlight-inner';
  innerOverlay.style.backgroundImage = `url("${img.src}")`;
  overlay.appendChild(innerOverlay);
  
  document.body.appendChild(overlay);
  return overlay;
}

function handleImageSelection(img) {
  console.log("Image selected:", img.src);
  resetUI();
  
  const highlightOverlay = createHighlightOverlay(img);
  highlightOverlay.classList.add('capflow-highlight-loading');
  
  document.getElementById('instruction-text').style.display = 'none';
  document.getElementById('loading-spinner').style.display = 'block';

  captureImage(img, 1024, 0.95)
    .then(imageData => {
      sendImageForCaption(imageData, highlightOverlay);
    });
}

function sendImageForCaption(imageData, highlightOverlay) {
  console.log("Sending message to background script with image data");
  chrome.runtime.sendMessage({ 
    type: "GET_CAPTION", 
    imageData: imageData 
  }, (response) => {
    console.log("Received response from background script:", response);
    if (response && response.caption) {
      displayCaption(response.caption, response.hashtags || []);
    } else {
      displayCaption("Error: Unable to generate caption", []);
    }
    highlightOverlay.classList.remove('capflow-highlight-loading');
    showStartAgainButton();
  });
}

function handleNonImageSelection() {
  displayCaption("An image wasn't selected");
  showStartAgainButton();
  removeDimOverlay();
}

function displayCaption(caption, hashtags) {
  console.log("Displaying caption:", caption);
  console.log("Displaying hashtags:", hashtags);
  const captionText = document.getElementById('caption-text');
  captionText.textContent = `Caption: ${caption}`;
  captionText.style.display = 'block';
  
  // Create hashtag buttons
  const hashtagContainer = document.createElement('div');
  hashtagContainer.id = 'hashtag-container';
  hashtagContainer.style.cssText = `
    display: flex;
    justify-content: center;
    gap: 10px;
    margin-top: 15px;
  `;
  
  hashtags.forEach(hashtag => {
    const button = document.createElement('button');
    button.textContent = hashtag;
    button.style.cssText = `
      background-color: #3498DB;
      color: white;
      border: none;
      padding: 5px 10px;
      border-radius: 5px;
      cursor: pointer;
    `;
    
    // Add click event listener to open Google Images search
    button.addEventListener('click', () => {
      const searchTerm = hashtag.slice(1); // Remove the # from the hashtag
      const googleImagesUrl = `https://www.google.com/search?tbm=isch&q=${encodeURIComponent(searchTerm)}`;
      window.open(googleImagesUrl, '_blank');
    });
    
    hashtagContainer.appendChild(button);
  });
  
  captionText.insertAdjacentElement('afterend', hashtagContainer);
  
  document.getElementById('loading-spinner').style.display = 'none';
}

function showStartAgainButton() {
  const startButton = document.getElementById('start-button');
  startButton.textContent = "Start Again";
  startButton.style.display = 'block';
  document.getElementById('instruction-text').style.display = 'none';
}

function handleStartAgainClick() {
  resetUI();
  removeDimOverlay();
  startFlow(); // Immediately restart the flow
}

// Update highlight position on scroll and resize
function updateHighlightPosition() {
  const highlight = document.querySelector('.capflow-highlight-container');
  if (highlight) {
    const img = document.elementFromPoint(
      parseInt(highlight.style.left) - window.scrollX + 1,
      parseInt(highlight.style.top) - window.scrollY + 1
    );
    if (img && img.tagName === 'IMG') {
      const rect = img.getBoundingClientRect();
      highlight.style.top = `${rect.top + window.scrollY}px`;
      highlight.style.left = `${rect.left + window.scrollX}px`;
      highlight.style.width = `${rect.width}px`;
      highlight.style.height = `${rect.height}px`;
      
      // Update the background image of the inner overlay
      const innerOverlay = highlight.querySelector('.capflow-highlight-inner');
      if (innerOverlay) {
        innerOverlay.style.backgroundImage = `url("${img.src}")`;
      }
    }
  }
}

window.addEventListener('scroll', updateHighlightPosition);
window.addEventListener('resize', updateHighlightPosition);

document.addEventListener('click', (event) => {
  const instructionText = document.getElementById('instruction-text');
  if (instructionText?.style.display === 'block') {
    event.preventDefault();
    event.stopPropagation();

    if (event.target.tagName === 'IMG') {
      handleImageSelection(event.target);
    } else {
      handleNonImageSelection();
    }
  }
}, true);