"""
CareerLens AI - Application Launcher
Starts both backend API and frontend Streamlit app
"""
import subprocess
import sys
import time
import requests
from pathlib import Path

def check_port(port):
    """Check if port is available"""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', port))
    sock.close()
    return result == 0

def start_backend():
    """Start FastAPI backend"""
    print("🚀 Starting backend API server...")
    backend_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for backend to start
    print("⏳ Waiting for backend to be ready...")
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:8000/health", timeout=1)
            if response.status_code == 200:
                print("✅ Backend API is ready!")
                return backend_process
        except:
            pass
        time.sleep(1)
    
    print("❌ Backend failed to start")
    return None

def start_frontend():
    """Start Streamlit frontend"""
    print("🎨 Starting Streamlit frontend...")
    frontend_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "streamlit_app/Home.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return frontend_process

def main():
    print("""
    ╔══════════════════════════════════════╗
    ║      CareerLens AI Launcher          ║
    ║  AI-Powered Career Guidance Platform ║
    ╚══════════════════════════════════════╝
    """)
    
    # Check if ports are available
    if check_port(8000):
        print("⚠️  Port 8000 already in use. Backend may already be running.")
    
    if check_port(8501):
        print("⚠️  Port 8501 already in use. Frontend may already be running.")
    
    # Start backend
    backend = start_backend()
    if not backend:
        print("Failed to start backend. Exiting...")
        sys.exit(1)
    
    # Start frontend
    frontend = start_frontend()
    
    print("""
    ✅ Application started successfully!
    
    📊 Backend API:  http://localhost:8000
    🎨 Frontend UI:  http://localhost:8501
    
    Press Ctrl+C to stop all services
    """)
    
    try:
        # Keep running
        backend.wait()
        frontend.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
        backend.terminate()
        frontend.terminate()
        print("✅ Services stopped")

if __name__ == "__main__":
    main()