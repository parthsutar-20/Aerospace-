import { useState, useRef, useEffect } from 'react'
import './index.css'

function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [latency, setLatency] = useState(0);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const startCamera = async () => {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error("Error accessing camera:", err);
        alert("Camera access denied or unavailable.");
      }
    }
  };

  const connectWebSocket = () => {
    wsRef.current = new WebSocket('ws://localhost:8000/ws/inference');
    
    wsRef.current.onopen = () => {
      setIsConnected(true);
      console.log('Connected to ML Engine');
    };
    
    wsRef.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.latency_ms !== undefined) {
          setLatency(data.latency_ms);
        }
        
        // Render bounding boxes on canvas
        if (data.defects && canvasRef.current && videoRef.current) {
          const ctx = canvasRef.current.getContext('2d');
          if (ctx) {
            ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
            data.defects.forEach((defect: any) => {
              const [x1, y1, x2, y2] = defect.bbox;
              ctx.strokeStyle = '#f85149';
              ctx.lineWidth = 3;
              ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
              ctx.fillStyle = '#f85149';
              ctx.font = '16px Arial';
              ctx.fillText(`${defect.type} (${Math.round(defect.confidence * 100)}%)`, x1, y1 - 5);
            });
          }
        }
      } catch (err) {
        console.error("Error parsing WebSocket message:", err);
      }
    };
    
    wsRef.current.onclose = () => {
      setIsConnected(false);
      console.log('Disconnected from ML Engine');
    };
  };

  useEffect(() => {
    const captureFrame = () => {
      if (isConnected && videoRef.current && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        const canvas = document.createElement('canvas');
        canvas.width = 640;
        canvas.height = 640;
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.drawImage(videoRef.current, 0, 0, 640, 640);
          const base64Data = canvas.toDataURL('image/jpeg', 0.8);
          wsRef.current.send(base64Data);
        }
      }
    };
    
    // Send frame every 100ms
    const interval = setInterval(captureFrame, 100);
    return () => clearInterval(interval);
  }, [isConnected]);

  // Adjust canvas size to match video
  useEffect(() => {
    if (videoRef.current && canvasRef.current) {
      canvasRef.current.width = videoRef.current.clientWidth;
      canvasRef.current.height = videoRef.current.clientHeight;
    }
  }, [videoRef.current?.clientWidth]);

  return (
    <div className="dashboard-container">
      <header className="header">
        <h1>AeroVision-MRO Dashboard</h1>
        <div className="latency-card">
          <span className="latency-label">Inference Latency</span>
          <span className={`latency-value ${latency > 15 ? 'high' : ''}`}>
            {latency.toFixed(1)} ms
          </span>
        </div>
      </header>
      
      <main className="main-content">
        <section className="video-panel">
          {!videoRef.current?.srcObject && (
            <div className="video-placeholder">
              <p>Camera Not Active</p>
            </div>
          )}
          <video 
            ref={videoRef} 
            autoPlay 
            playsInline 
            muted 
            style={{ width: '100%', height: '100%', objectFit: 'cover' }}
          />
          <canvas ref={canvasRef} />
        </section>
        
        <aside className="sidebar">
          <div className="control-panel">
            <h2>Controls</h2>
            <button className="btn" onClick={startCamera}>Start Camera</button>
            <button 
              className="btn" 
              onClick={connectWebSocket}
              disabled={isConnected}
              style={{ marginTop: '10px' }}
            >
              {isConnected ? 'ML Engine Connected' : 'Connect ML Engine'}
            </button>
          </div>
          
          <div className="log-panel">
            <h2>Audit Log</h2>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
              Waiting for defects...
            </p>
          </div>
        </aside>
      </main>
    </div>
  )
}

export default App
