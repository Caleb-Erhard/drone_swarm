package main

import (
	"log"
	"net/http"
	"strings"
	"sync"

	"github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 512 * 1024,
	CheckOrigin:     func(r *http.Request) bool { return true },
}

var (
	frontendClients = make(map[*websocket.Conn]bool)
	clientsMutex    sync.Mutex
)

func handleUnity(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Println("Unity connection error:", err)
		return
	}
	defer conn.Close()
	log.Println("Unity Simulation Connected.")

	for {
		messageType, message, err := conn.ReadMessage()
		if err != nil {
			log.Println("Unity disconnected:", err)
			break
		}
		clientsMutex.Lock()
		for client := range frontendClients {
			if err := client.WriteMessage(messageType, message); err != nil {
				client.Close()
				delete(frontendClients, client)
			}
		}
		clientsMutex.Unlock()
	}
}

func handleFrontend(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Println("Frontend connection error:", err)
		return
	}
	defer conn.Close()

	clientsMutex.Lock()
	frontendClients[conn] = true
	clientsMutex.Unlock()
	log.Println("Next.js Dashboard Connected.")

	for {
		if _, _, err := conn.ReadMessage(); err != nil {
			break
		}
	}

	clientsMutex.Lock()
	delete(frontendClients, conn)
	clientsMutex.Unlock()
	log.Println("Next.js Dashboard Disconnected.")
}

// ── Video relay ───────────────────────────────────────────────────────────────

type viewer struct {
	conn      *websocket.Conn
	ch        chan []byte
	closeOnce sync.Once
}

func newViewer(conn *websocket.Conn) *viewer {
	v := &viewer{
		conn: conn,
		ch:   make(chan []byte, 4),
	}
	go v.writeLoop()
	return v
}

func (v *viewer) writeLoop() {
	for frame := range v.ch {
		if err := v.conn.WriteMessage(websocket.BinaryMessage, frame); err != nil {
			v.close()
			return
		}
	}
}

func (v *viewer) send(frame []byte) bool {
	select {
	case v.ch <- frame:
		return true
	default:
		return false // Channel full
	}
}

func (v *viewer) close() {
	v.closeOnce.Do(func() {
		close(v.ch)
		v.conn.Close()
	})
}

type videoHub struct {
	mu      sync.Mutex
	viewers map[*websocket.Conn]*viewer
}

var (
	hubs   = make(map[string]*videoHub)
	hubsMu sync.Mutex
)

func getOrCreateHub(streamID string) *videoHub {
	hubsMu.Lock()
	defer hubsMu.Unlock()
	h, ok := hubs[streamID]
	if !ok {
		h = &videoHub{viewers: make(map[*websocket.Conn]*viewer)}
		hubs[streamID] = h
	}
	return h
}

// Replace your handleVideoProducer function in main.go with this:
func handleVideoProducer(w http.ResponseWriter, r *http.Request) {
	streamID := strings.TrimPrefix(r.URL.Path, "/ws/video/")
	if streamID == "" {
		return
	}

	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		return
	}
	defer conn.Close()

	log.Printf("Video Producer Connected: %s\n", streamID)
	hub := getOrCreateHub(streamID)

	for {
		_, frame, err := conn.ReadMessage()
		if err != nil {
			log.Printf("Video Producer Disconnected (waiting for handover): %s\n", streamID)
			break
		}

		// Broadcast to viewers
		hub.mu.Lock()
		for viewerConn, v := range hub.viewers {
			if !v.send(frame) {
				v.close()
				delete(hub.viewers, viewerConn)
			}
		}
		hub.mu.Unlock()
	}

	// REMOVED: The loop that closes all viewers.
	// We now allow viewers to stay connected and wait for the Tracking twin to connect.
}

func handleVideoConsumer(w http.ResponseWriter, r *http.Request) {
	streamID := strings.TrimPrefix(r.URL.Path, "/ws/video-client/")
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		return
	}
	log.Printf("Consumer: %s\n", streamID)

	hub := getOrCreateHub(streamID)
	v := newViewer(conn)

	hub.mu.Lock()
	hub.viewers[conn] = v
	hub.mu.Unlock()

	for {
		if _, _, err := conn.ReadMessage(); err != nil {
			break
		}
	}

	hub.mu.Lock()
	delete(hub.viewers, conn)
	hub.mu.Unlock()
	v.close()
}

func main() {
	http.HandleFunc("/ws/unity", handleUnity)
	http.HandleFunc("/ws/frontend", handleFrontend)
	http.HandleFunc("/ws/video/", handleVideoProducer)
	http.HandleFunc("/ws/video-client/", handleVideoConsumer)

	log.Println("Go relay running on :8080")
	http.ListenAndServe(":8080", nil)
}
