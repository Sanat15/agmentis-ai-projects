# Technical Report — WebRTC Browser Extension

**Project:** WebRTC Browser Extension with Live Audio Capture  
**Author:** Sanat, IIT Indore  
**Context:** Agmentis.ai Internship Evaluation  

---

## 1. Overview

This project was built to understand and demonstrate how browser extensions can interact with WebRTC — both by capturing audio from existing WebRTC calls running in a tab, and by initiating entirely new WebRTC peer-to-peer calls directly from within the extension itself.

The motivation came from a real-world problem: cloud telephony platforms like Acefone and Tata Smartflo handle calls through their web portals, and the question was — how do you capture live audio from those calls for processing (transcription, AI analysis, etc.)? This extension explores one answer to that question.

---

## 2. Problem Statement

Cloud telephony platforms run calls inside browser tabs. Capturing that audio for real-time processing requires:

1. Getting access to the tab's audio stream while the call is live
2. Without interrupting the call
3. Without requiring changes to the telephony platform itself

Additionally, the task required understanding whether WebRTC could be used to initiate calls from within an extension — and what limitations exist.

---

## 3. Architecture

### 3.1 Tab Audio Capture

The extension uses `chrome.tabCapture` — a Chrome-exclusive API available only to browser extensions — to tap into the audio rendering pipeline of the active tab.

When a tab produces audio (from a WebRTC call, video, or any web audio), Chrome renders it through an internal Web Audio API graph before sending it to speakers. `tabCapture` creates a fork at this point:

```
Tab audio → Chrome renderer → FORK
                               ├── Speakers (unchanged)
                               └── MediaStream (extension copy)
```

The extension receives a `MediaStream` object identical in type to what `getUserMedia()` returns. This stream can then be:
- Analysed in real time using the Web Audio API (`ScriptProcessor`, `AudioContext`)
- Sent to a server via WebSocket for transcription or AI processing
- Recorded locally

**Latency:** The fork is a memory copy inside the same browser process — approximately 1ms. Buffer latency adds 16–32ms. Total end-to-end latency is approximately 20–40ms, faster than Acefone's own WebSocket streaming which sends chunks every 100ms.

### 3.2 WebRTC Call

The extension's second feature initiates a genuine WebRTC peer-to-peer audio call:

```
Extension (call.html) → getUserMedia → RTCPeerConnection
                      → WebSocket signalling → Other peer
                      → ICE negotiation → Direct P2P audio
```

**Key components:**

| Component | Role |
|---|---|
| `getUserMedia()` | Captures local microphone |
| `RTCPeerConnection` | Manages the P2P connection and audio encoding |
| STUN (Google) | Discovers each peer's public IP for NAT traversal |
| WebSocket signalling server | Exchanges SDP offer/answer and ICE candidates |

### 3.3 Signalling Server

A lightweight Node.js server using the `ws` library manages room-based peer discovery:

- Maintains a `rooms` object mapping room names to connected sockets
- When two peers join the same room, sends a `ready` signal to trigger offer creation
- Forwards `offer`, `answer`, and `candidate` messages between peers
- Cleans up rooms on disconnection

Once the WebRTC connection is established, the signalling server is no longer involved in the call.

---

## 4. Key Technical Findings

### 4.1 Popup vs Tab Context

Chrome's security model distinguishes between extension popups and full tab pages. `getUserMedia()` — which requests microphone access — is blocked in extension popups because they are considered untrusted rendering contexts.

**Solution:** The WebRTC call is opened as a full browser tab (`call.html`) via `chrome.tabs.create()`. Full tab pages opened by extensions have trusted context and microphone access works normally.

### 4.2 tabCapture vs getUserMedia

These are fundamentally different APIs capturing from different sources:

| | `getUserMedia` | `chrome.tabCapture` |
|---|---|---|
| Source | Hardware microphone | Tab's entire audio output |
| Who can use it | Any webpage | Extensions only |
| What it captures | Your voice only | Both sides of a call |
| Requires permission | User permission popup | `tabCapture` in manifest |

For call monitoring purposes, `tabCapture` is the correct choice because it captures the mixed audio of both participants.

### 4.3 ICE and NAT Traversal

WebRTC peers don't know their own public IP addresses because routers use NAT (Network Address Translation). The ICE (Interactive Connectivity Establishment) framework solves this:

1. Each peer queries the STUN server to learn its public IP/port
2. Multiple candidate paths are generated (local, public, relay)
3. Both peers exchange candidates via signalling
4. The best working path is selected automatically

---

## 5. Relationship to Acefone/Tata Smartflo

Both Acefone and Tata Smartflo (Smartflo) provide **bi-directional audio streaming via WebSocket** — a server-side equivalent of what this extension demonstrates client-side.

| Approach | Who captures | Protocol | Latency |
|---|---|---|---|
| This extension (tabCapture) | Client (browser) | Chrome internal API | ~20-40ms |
| Acefone WebSocket streaming | Server (Acefone) | WebSocket + mulaw G.711 | ~100ms chunks |
| Tata Smartflo Voice Streaming | Server (Tata) | WebSocket + mulaw G.711 | ~100ms chunks |

The extension approach works independently of the telephony provider — as long as the call is happening in a browser tab, the extension can capture it regardless of whether Acefone or Tata is used.

---

## 6. Limitations

- `chrome.tabCapture` captures **mixed** audio — both sides of the call are blended. Separating individual speakers requires the Insertable Streams API (advanced).
- The signalling server must have a public URL for production use — `localhost:3000` only works for local testing.
- Chrome blocks `getUserMedia` in extension popups — the call must run in a separate tab.
- `tabCapture` is Chrome-only — this approach does not work in Firefox or Safari.

---

## 7. Conclusion

The extension successfully demonstrates both objectives — capturing live tab audio including WebRTC call audio, and initiating a WebRTC call from within an extension. The main insight is that these are two separate layers: `tabCapture` sits above WebRTC and captures its output, while `RTCPeerConnection` is used to build a WebRTC call from scratch. Both can coexist within the same extension.

For the Agmentis.ai use case of live call audio capture and processing, the Acefone/Tata WebSocket streaming approach is preferable for production (server-side, no client install required), while the browser extension approach works as a client-side fallback that is telephony-provider agnostic.
