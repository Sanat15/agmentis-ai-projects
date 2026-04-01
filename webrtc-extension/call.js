const callBtn     = document.getElementById('callBtn')
const hangupBtn   = document.getElementById('hangupBtn')
const callStatus  = document.getElementById('callStatus')
const roomInput   = document.getElementById('roomId')
const remoteAudio = document.getElementById('remoteAudio')

const SIGNAL_SERVER = 'ws://localhost:3000'

let peerConnection  = null
let signalingSocket = null
let localStream     = null

const iceConfig = {
  iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
}

callBtn.addEventListener('click', async () => {

  const room = roomInput.value.trim()
  if (!room) {
    callStatus.textContent = 'Please enter a Room ID'
    callStatus.style.color = '#e94560'
    return
  }

  callStatus.textContent = 'Getting microphone...'

  // getUserMedia works fine here because this is a full tab not a popup
  try {
    localStream = await navigator.mediaDevices.getUserMedia({ audio: true })
    console.log('Mic access granted')
    callStatus.textContent = 'Mic granted, connecting...'
  } catch (err) {
    callStatus.textContent = 'Mic denied: ' + err.message
    callStatus.style.color = '#e94560'
    return
  }

  // Create WebRTC peer connection
  peerConnection = new RTCPeerConnection(iceConfig)

  // Add local mic audio into the connection
  localStream.getTracks().forEach(track => {
    peerConnection.addTrack(track, localStream)
  })

  // When remote audio arrives play it
  peerConnection.ontrack = (event) => {
    console.log('Remote audio received')
    remoteAudio.srcObject = event.streams[0]
    callStatus.textContent = '✅ In call'
    callStatus.style.color = '#4ecca3'
  }

  // Send ICE candidates to other peer via signalling server
  peerConnection.onicecandidate = (event) => {
    if (event.candidate) {
      signalingSocket.send(JSON.stringify({
        type: 'candidate',
        candidate: event.candidate,
        room
      }))
    }
  }

  // Log connection state changes so we can see whats happening
  peerConnection.onconnectionstatechange = () => {
    console.log('Connection state:', peerConnection.connectionState)
    if (peerConnection.connectionState === 'connected') {
      callStatus.textContent = '✅ In call'
      callStatus.style.color = '#4ecca3'
    }
    if (peerConnection.connectionState === 'disconnected') {
      callStatus.textContent = 'Disconnected'
      callStatus.style.color = '#e94560'
    }
  }

  // Connect to signalling server
  signalingSocket = new WebSocket(SIGNAL_SERVER)

  signalingSocket.onopen = () => {
    console.log('Connected to signalling server')
    signalingSocket.send(JSON.stringify({ type: 'join', room }))
    callStatus.textContent = 'Waiting for other person...'
  }

  signalingSocket.onmessage = async (message) => {
    const data = JSON.parse(message.data)
    console.log('Signal received:', data.type)

    if (data.type === 'ready') {
      // We are the first person - create the offer
      const offer = await peerConnection.createOffer()
      await peerConnection.setLocalDescription(offer)
      signalingSocket.send(JSON.stringify({ type: 'offer', offer, room }))
      callStatus.textContent = 'Offer sent, waiting...'

    } else if (data.type === 'offer') {
      // We are the second person - respond with answer
      await peerConnection.setRemoteDescription(data.offer)
      const answer = await peerConnection.createAnswer()
      await peerConnection.setLocalDescription(answer)
      signalingSocket.send(JSON.stringify({ type: 'answer', answer, room }))
      callStatus.textContent = 'Answer sent, connecting...'

    } else if (data.type === 'answer') {
      await peerConnection.setRemoteDescription(data.answer)

    } else if (data.type === 'candidate') {
      await peerConnection.addIceCandidate(data.candidate)

    } else if (data.type === 'peer-left') {
      callStatus.textContent = 'Other person left'
      callStatus.style.color = '#e94560'

    } else if (data.type === 'error') {
      callStatus.textContent = 'Error: ' + data.message
      callStatus.style.color = '#e94560'
    }
  }

  signalingSocket.onerror = () => {
    callStatus.textContent = 'Cannot reach server - is it running?'
    callStatus.style.color = '#e94560'
  }

  // Update UI
  callBtn.disabled = true
  callBtn.style.background = '#444'
  callBtn.style.cursor = 'not-allowed'
  hangupBtn.disabled = false
  hangupBtn.style.background = '#e94560'
  hangupBtn.style.color = 'white'
  hangupBtn.style.cursor = 'pointer'

})

hangupBtn.addEventListener('click', () => {
  if (peerConnection)  { peerConnection.close(); peerConnection = null }
  if (localStream)     { localStream.getTracks().forEach(t => t.stop()); localStream = null }
  if (signalingSocket) { signalingSocket.close(); signalingSocket = null }
  remoteAudio.srcObject = null
  callStatus.textContent = 'Not in a call'
  callStatus.style.color = '#aaa'
  callBtn.disabled = false
  callBtn.style.background = '#4ecca3'
  callBtn.style.cursor = 'pointer'
  hangupBtn.disabled = true
  hangupBtn.style.background = '#444'
  hangupBtn.style.color = '#aaa'
  hangupBtn.style.cursor = 'not-allowed'
})