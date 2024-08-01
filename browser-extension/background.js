chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "GET_CAPTION") {
    console.log("Received GET_CAPTION message");
    const apiUrl = 'http://localhost:5000/generate_caption';

    console.log("Sending request to API:", apiUrl);
    
    let body;
    if (message.imageData.startsWith('data:image')) {
      body = JSON.stringify({ image_data: message.imageData });
    } else {
      body = JSON.stringify({ image_url: message.imageData });
    }

    fetch(apiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: body,
    })
    .then(response => {
      console.log("API Response status:", response.status);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return response.json();
    })
    .then(data => {
      console.log("API Data:", data);
      if (data && data.caption) {
        sendResponse({ 
          caption: data.caption,
          hashtags: data.hashtags || [] 
        });
      } else if (data && data.error) {
        throw new Error(data.error);
      } else {
        throw new Error("Unexpected response from server");
      }
    })
    .catch(error => {
      console.error('Error:', error);
      sendResponse({ 
        caption: `Error generating caption: ${error.message}`,
        hashtags: [] 
      });
    });

    return true; 
  }
});