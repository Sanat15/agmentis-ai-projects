const WebSocket = require('ws')

// Start the server on port 3000
const server = new WebSocket.Server({ port: 3000 })

// Rooms object - stores who is in which room
// Example: { "room1": [socket1, socket2] }
const rooms = {}

console.log('Signalling server running on ws://localhost:3000')

server.on('connection', (socket) => {
  console.log('New peer connected')

  // Track which room this socket joined
  socket.currentRoom = null

  // Handle messages from peers
  socket.on('message', (message) => {
    const data = JSON.parse(message)
    console.log('Message received:', data.type, '| Room:', data.room)

    // ---- JOIN ----
    if (data.type === 'join') {
      const room = data.room

      // Create room if it doesnt exist
      if (!rooms[room]) {
        rooms[room] = []
      }

      // Add this socket to the room
      rooms[room].push(socket)
      socket.currentRoom = room

      console.log(`Peer joined room: ${room} | Total in room: ${rooms[room].length}`)

      // If two people are now in the room, tell the first one to start
      if (rooms[room].length === 2) {
        // Tell the first person (index 0) that someone is ready
        rooms[room][0].send(JSON.stringify({ type: 'ready' }))
        console.log('Two peers in room - sent ready signal')
      }

      // If more than 2 people try to join, reject them
      if (rooms[room].length > 2) {
        socket.send(JSON.stringify({ type: 'error', message: 'Room is full' }))
        rooms[room].pop()
      }
    }

    // ---- OFFER, ANSWER, CANDIDATE ----
    // Forward these messages to the OTHER person in the room
    else if (data.type === 'offer' || data.type === 'answer' || data.type === 'candidate') {
      const room = data.room

      if (rooms[room]) {
        // Find the other socket in the room (not the sender)
        rooms[room].forEach((peer) => {
          if (peer !== socket) {
            peer.send(JSON.stringify(data))
            console.log(`Forwarded ${data.type} to other peer`)
          }
        })
      }
    }

  })

  // Handle disconnection - clean up the room
  socket.on('close', () => {
    const room = socket.currentRoom

    if (room && rooms[room]) {
      // Remove this socket from the room
      rooms[room] = rooms[room].filter(peer => peer !== socket)
      console.log(`Peer left room: ${room} | Remaining: ${rooms[room].length}`)

      // Tell the other person their partner left
      if (rooms[room].length > 0) {
        rooms[room][0].send(JSON.stringify({ type: 'peer-left' }))
      }

      // Delete the room if empty
      if (rooms[room].length === 0) {
        delete rooms[room]
        console.log(`Room deleted: ${room}`)
      }
    }
  })

})