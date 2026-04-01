// ============================================
// SECTION 1 - TAB CAPTURE (existing)
// ============================================

const startBtn = document.getElementById('startBtn')
const stopBtn = document.getElementById('stopBtn')
const status = document.getElementById('status')

let mediaStream = null

startBtn.addEventListener('click', () => {
  chrome.tabCapture.capture({ audio: true, video: false }, (stream) => {
    if (!stream) {
      status.textContent = 'Error: ' + chrome.runtime.lastError.message
      status.style.color = '#e94560'
      return
    }
    mediaStream = stream
    status.textContent = 'Capturing audio...'
    status.style.color = '#4ecca3'
    startBtn.disabled = true
    startBtn.style.background = '#444'
    startBtn.style.cursor = 'not-allowed'
    stopBtn.disabled = false
    stopBtn.style.background = '#e94560'
    stopBtn.style.color = 'white'
    stopBtn.style.cursor = 'pointer'
    console.log('Stream started:', stream)
    console.log('Audio tracks:', stream.getAudioTracks())
  })
})

stopBtn.addEventListener('click', () => {
  if (mediaStream) {
    mediaStream.getTracks().forEach(track => track.stop())
    mediaStream = null
  }
  status.textContent = 'Not capturing'
  status.style.color = '#aaa'
  startBtn.disabled = false
  startBtn.style.background = '#e94560'
  startBtn.style.cursor = 'pointer'
  stopBtn.disabled = true
  stopBtn.style.background = '#444'
  stopBtn.style.color = '#aaa'
  stopBtn.style.cursor = 'not-allowed'
})


// ============================================
// SECTION 2 - WEBRTC CALL (opens as a tab)
// ============================================

const callBtn    = document.getElementById('callBtn')
const hangupBtn  = document.getElementById('hangupBtn')
const callStatus = document.getElementById('callStatus')

// Opens call.html as a new tab where mic access works
callBtn.addEventListener('click', () => {
  chrome.tabs.create({
    url: chrome.runtime.getURL('call.html')
  })
  window.close() // close the popup
})