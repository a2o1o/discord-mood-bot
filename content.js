
// map moods to CSS filters
// content.js - Discord Mood Highlighter

// 1. Log when the script loads
console.log("✅ mood-tint content.js loaded");

// 2. Define CSS filters for each mood
const MOOD_STYLES = {
  roast:      'hue-rotate(0deg) contrast(1.2)',
  sad:        'hue-rotate(200deg) brightness(0.7)',
  happy:      'hue-rotate(50deg) saturate(1.5)',
  jealousy:   'hue-rotate(-50deg) saturate(1.8)',
  excitement: 'hue-rotate(80deg) saturate(2) brightness(1.1)',
  neutral:    ''
};

// 3. Function to apply the filter and schedule reset
let clearMoodTimeout;
function applyMoodFilter(mood) {
  // Apply the tint
  document.documentElement.style.filter = MOOD_STYLES[mood] || '';
  
  // Clear any existing reset timer
  clearTimeout(clearMoodTimeout);
  
  // Reset back to neutral after 15 seconds
  clearMoodTimeout = setTimeout(() => {
    document.documentElement.style.filter = MOOD_STYLES['neutral'];
  }, 15000);
}

// 4. Observe all new DOM nodes for “Mood: <emotion>” and apply tint
const observer = new MutationObserver(mutations => {
  mutations.forEach(mutation => {
    mutation.addedNodes.forEach(node => {
      if (!(node instanceof HTMLElement)) return;
      
      // Read the text of each new node
      const text = node.innerText.toLowerCase();
      
      // Look for "mood: <word>"
      const match = text.match(/mood:\s*(\w+)/);
      if (match) {
        const mood = match[1];
        console.log("mood-tint: detected mood", mood);
        applyMoodFilter(mood);
      }
    });
  });
});

// Start observing the entire document
observer.observe(document.body, { childList: true, subtree: true });
