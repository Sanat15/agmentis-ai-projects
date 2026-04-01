# WebRTC Browser Extension

A Chrome browser extension that demonstrates live tab audio capture and peer-to-peer WebRTC calling — built as a learning project during Agmentis.ai internship evaluation.

---

## What It Does

The extension has two independent features:

### Feature 1 — Tab Audio Capture
Captures all audio playing in the active browser tab in real time using Chrome's `tabCapture` API. This includes WebRTC call audio from apps like Bitrix24, video audio, and any web app sounds.

### Feature 2 — WebRTC Call
Opens a full call page where two people can join the same room and make a live peer-to-peer audio call directly through the browser — no third party service needed.

---

## File Structure

```
webrtc-extension/
  manifest.json       — Extension config, name, permissions
  popup.html          — UI shown when clicking extension icon
  popup.js            — Tab capture logic
  call.html           — Full page for WebRTC call
  call.js             — WebRTC connection logic

signal-server/
  server.js           — Node.js WebSocket signalling server
```

---

## How to Run

### Step 1 — Load the Extension in Chrome
1. Open `chrome://extensions`
2. Enable **Developer Mode** (top right)
3. Click **Load Unpacked**
4. Select the `webrtc-extension` folder

### Step 2 — Start the Signalling Server
```bash
cd signal-server
npm install
node server.js
```
Server runs at `ws://localhost:3000`

### Step 3 — Test the Call
1. Click the extension icon → click **Open Call Window**
2. Open a second Chrome window, do the same
3. Both type the same room name (e.g. `room1`)
4. Click **Join / Start Call** in both
5. Allow microphone permission
6. Both windows are now on a live WebRTC call

---

## Tech Stack

| Technology | Purpose |
|---|---|
| Chrome Extensions (MV3) | Extension framework |
| `chrome.tabCapture` | Live tab audio capture |
| WebRTC (`RTCPeerConnection`) | Peer to peer audio call |
| `getUserMedia` | Microphone access |
| WebSocket (`ws` library) | Signalling server |
| Node.js | Signalling server runtime |
| Google STUN | NAT traversal / public IP discovery |

---

## How It Works — Core Concepts

### Tab Capture
`chrome.tabCapture` hooks into Chrome's internal audio rendering pipeline and creates a fork — your extension gets a live copy of all tab audio while the original plays normally. The captured audio arrives as a `MediaStream` object identical to what `getUserMedia()` returns.

### WebRTC Call Flow
```
Person A joins room → Server sends "ready" to A
A creates Offer (SDP) → Server forwards to B
B creates Answer (SDP) → Server forwards to A
Both exchange ICE candidates → best path found
Direct peer-to-peer audio connection established
Server no longer involved
```

### Why call.html is a Separate Tab
Chrome blocks `getUserMedia()` (microphone access) in extension popups — they are considered untrusted contexts. Opening `call.html` as a full browser tab gives it trusted context and mic access works normally.

---

## Key Decisions Explained

**Why `tabCapture` and not `getUserMedia` for call audio?**
`getUserMedia` only captures YOUR microphone. `tabCapture` captures the entire tab output — both sides of a call, merged.

**Why WebSocket for signalling?**
HTTP is request-response and closes. WebSocket stays permanently open so the server can push messages to you when the other person joins — essential for real time events.

**Why Google's STUN server?**
Peers don't know their own public IP (routers hide it). STUN server tells each peer what their public address looks like from the outside world so they can share it with each other.

---

## Requirements

- Google Chrome (any modern version)
- Node.js v16+
- `ws` npm package (`npm install ws`)
