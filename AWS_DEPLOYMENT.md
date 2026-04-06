# SwitchIt-Pro · Scout AI Interviewer: AWS Deployment Guide

This guide covers deploying the `scout_ai_interviewer` Question Bank Manager — including the FastAPI backend, ChromaDB VectorDB, and Ollama inference engine — onto Amazon Web Services.

Because the system runs `qwen2.5:1.5b` locally for question generation and `nomic-embed-text` for embeddings, you need a GPU-enabled EC2 instance for production performance. For development/demo use, a CPU-only instance works but generation will be slower (~2–3 min per batch instead of ~30s).

---

## 1. Choose the Right EC2 Instance

| Use Case | Instance | GPU | Cost (est.) |
|----------|----------|-----|-------------|
| **Production** (recommended) | `g4dn.xlarge` | NVIDIA T4 (16 GB VRAM) | ~$0.53/hr |
| **Development / Demo** | `t3.large` | None (CPU only) | ~$0.08/hr |
| **High Volume** | `g4dn.2xlarge` | NVIDIA T4 (16 GB VRAM) | ~$0.75/hr |

**Storage:** Provision an EBS Volume of at least **80 GB (gp3)**:
- OS + dependencies: ~10 GB
- Ollama model binaries (`qwen2.5:1.5b` + `nomic-embed-text`): ~2 GB
- ChromaDB vector files: grows with question count
- Application + Excel: ~50 MB

**AMI:** Use **Ubuntu Server 22.04 LTS**

---

## 2. Security Group Configuration

Open the following inbound ports on your EC2 Security Group:

| Port | Protocol | Purpose |
|------|----------|---------|
| `22` | TCP | SSH access |
| `8050` | TCP | Scout AI Web Portal |
| `11434` | TCP | Ollama API (internal only — restrict to VPC) |
| `80` / `443` | TCP | Optional: Nginx reverse proxy + SSL |

> **Security tip:** Restrict port `11434` (Ollama) to your VPC CIDR only — do not expose it publicly.

---

## 3. Server Initialization

SSH into your instance:
```bash
ssh -i your-key.pem ubuntu@<your-ec2-public-ip>
```

### Install System Dependencies
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git curl
```

### Install NVIDIA Drivers (GPU instances only)
> Skip this if you selected an AWS Deep Learning AMI — drivers are pre-installed.

```bash
sudo apt install -y nvidia-driver-535
sudo reboot
# After reboot, verify:
nvidia-smi
```

---

## 4. Install & Configure Ollama

Ollama runs as a background systemd service on Linux and automatically uses the GPU if available.

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Verify Ollama is running
ollama list

# Pull required models
ollama pull qwen2.5:1.5b       # LLM for question generation (batches of 10)
ollama pull nomic-embed-text   # Embedding model for ChromaDB
```

Ollama listens on `http://localhost:11434` by default. The application connects to this automatically.

---

## 5. Deploy the Application

### Clone the repository
```bash
git clone <your-github-repo-url>
cd scout_ai_interviewer
```

### Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Python dependencies
```bash
pip install -r requirements.txt
```

### Verify data folder
The server auto-creates `data/questions.xlsx` on first startup if it doesn't exist. Ensure the `data/` directory exists:
```bash
mkdir -p data
```

---

## 6. Run as a Background Service (systemd)

Keep the FastAPI server alive after closing your SSH session using a `systemd` service.

### Create the service file
```bash
sudo nano /etc/systemd/system/switchit-scout.service
```

### Paste this configuration
```ini
[Unit]
Description=SwitchIt-Pro Scout AI Question Bank Manager
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/scout_ai_interviewer
ExecStart=/home/ubuntu/scout_ai_interviewer/venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8050
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

### Enable and start
```bash
sudo systemctl daemon-reload
sudo systemctl enable switchit-scout
sudo systemctl start switchit-scout

# Check status
sudo systemctl status switchit-scout

# View live logs
journalctl -u switchit-scout -f
```

---

## 7. (Optional) Nginx Reverse Proxy + SSL

For a production deployment with HTTPS:

```bash
sudo apt install -y nginx certbot python3-certbot-nginx
```

Create an Nginx config at `/etc/nginx/sites-available/switchit`:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8050;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 300s;   # Important: generation jobs can take 2-4 min
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/switchit /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# Add SSL
sudo certbot --nginx -d your-domain.com
```

> **Important:** Set `proxy_read_timeout 300s` — question generation batches can take 2–4 minutes on CPU instances. Without this, Nginx will kill the connection mid-generation.

---

## 8. Access the Portal

Open your browser:
```
http://<your-ec2-public-ip>:8050
```

Or with Nginx + SSL:
```
https://your-domain.com
```

The portal has 3 pages:
- **Question Bank** — browse all generated roles in the VectorDB
- **Add Questions** — enter role + level, Qwen generates and pushes to ChromaDB
- **Manage Roles** — search and delete role question sets

---

## 9. Useful Commands on Server

```bash
# Check Ollama model status
ollama list

# Check ChromaDB question count (Python)
python3 -c "import chromadb; c = chromadb.PersistentClient('./chroma_store'); print(c.get_collection('scout_questions').count(), 'questions')"

# Inspect the question store
python scripts/inspect_db.py

# Semantic search test
python scripts/inspect_db.py --search "cold calling objection handling"

# Manual bulk load from questions.xlsx to ChromaDB
python scripts/load_data.py --sync
```

---

## 10. Environment Variables (.env)

Create a `.env` file in the project root to override defaults:

```env
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_WORKERS=8
CHROMA_PERSIST_DIR=chroma_store
CHROMA_COLLECTION_NAME=scout_questions
BATCH_SIZE=50
SYNC_POLL_INTERVAL=10
```
