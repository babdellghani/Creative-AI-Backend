# Creative-AI: Comprehensive Audio Generation and Voice Conversion Suite

A powerful Docker-based AI audio platform that combines three state-of-the-art models for complete audio generation and voice conversion capabilities, with full AWS deployment support.

## 🎯 Overview

Creative-AI integrates three cutting-edge AI models into a unified platform:

- **StyleTTS2**: Human-level text-to-speech synthesis with fine-tuning capabilities
- **Seed-VC**: Zero-shot voice conversion and real-time voice cloning
- **Make-An-Audio**: Text-to-audio generation with diffusion models

All services are containerized and can be deployed locally or on AWS with ECR and EC2 integration.

## Frontend Interface

A complete web frontend is available for easy interaction with all services:
- **Frontend Repository**: [Creative-AI-Frontend](https://github.com/babdellghani/Creative-AI-Frontend.git)
- **Features**: Web-based interface for all three AI models
- **Integration**: Direct API communication with StyleTTS2, Seed-VC, and Make-An-Audio services

## 🚀 Features

### StyleTTS2 - Text-to-Speech
- Human-level TTS synthesis quality
- Single and multi-speaker support
- Style diffusion with adversarial training
- Zero-shot speaker adaptation
- Fine-tuning capabilities

### Seed-VC - Voice Conversion
- Zero-shot voice conversion (1-30 seconds reference)
- Real-time voice conversion (~300ms latency)
- Singing voice conversion
- Minimal training data requirement (1 utterance per speaker)
- Fast fine-tuning (100 steps, 2 minutes on T4)

### Make-An-Audio - Text-to-Audio
- High-fidelity audio generation from text
- Prompt-enhanced diffusion models
- Audio inpainting capabilities
- Multiple conditioning modalities

## 🛠️ Prerequisites & Installation

### System Requirements
- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.10.16 (recommended)
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **RAM**: 16GB+ system RAM
- **Storage**: 50GB+ free space for models and data
- **Tools**: Git, VSCode, Docker, FFmpeg

### Initial Setup

1. **Install Prerequisites:**
   ```powershell
   # Install Python 3.10.16
   # Download from: https://github.com/adang1345/PythonWindows/blob/master/3.10.16/python-3.10.16-amd64-full.exe
   
   # Install Microsoft C++ Build Tools
   # Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   
   # Install Git LFS
   git lfs install
   
   # Install FFmpeg (required for audio processing)
   # Follow: https://www.youtube.com/watch?v=JR36oH35Fgg
   ```

2. **Clone the repositories:**
   ```powershell
   # Create main project folder
   mkdir Creative-AI
   cd Creative-AI
   
   # Clone all required repositories
   git clone https://github.com/IIEleven11/StyleTTS2FineTune.git
   git clone https://github.com/yl4579/StyleTTS2.git
   git clone https://github.com/Plachtaa/seed-vc.git
   git clone https://github.com/Text-to-Audio/Make-An-Audio.git
   
   # Clone the frontend (optional - for web interface)
   git clone https://github.com/babdellghani/Creative-AI-Frontend.git
   ```

3. **Setup Model Dependencies:**
   ```powershell
   # For StyleTTS2
   cd StyleTTS2
   git clone https://huggingface.co/yl4579/StyleTTS2-LibriTTS
   
   # For Make-An-Audio
   cd ../Make-An-Audio
   git clone https://huggingface.co/microsoft/msclap
   # Download additional models from Google Drive (see Backend.txt for links)
   ```

### Environment Setup

Each service requires its own Python environment:

```powershell
# Enable script execution
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# For StyleTTS2
cd StyleTTS2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -U

# For StyleTTS2FineTune
cd ../StyleTTS2FineTune
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -U
pip install git+https://github.com/m-bain/whisperx.git
pip install phonemizer pydub pysrt tqdm

# For Seed-VC
cd ../seed-vc
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# For Make-An-Audio
cd ../Make-An-Audio
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 🚀 Quick Start - Local Development

### Running Individual Services

Each service can be run independently for development:

```powershell
# StyleTTS2 API
cd StyleTTS2
.\.venv\Scripts\Activate.ps1
uvicorn api:app --reload --port 8000

# Seed-VC API  
cd ../seed-vc
.\.venv\Scripts\Activate.ps1
uvicorn api:app --reload --port 8001

# Make-An-Audio API
cd ../Make-An-Audio
.\.venv\Scripts\Activate.ps1
uvicorn api:app --reload --port 8002
```

### Docker Deployment

For production deployment:

```powershell
# Build individual containers
docker build -f StyleTTS2/Dockerfile.api -t styletts2-api .
docker build -f seed-vc/Dockerfile.api -t seed-vc-api .
docker build -f Make-An-Audio/Dockerfile.api -t make-an-audio-api .

# Run with Docker Compose
docker-compose up -d
```

### Service Endpoints

- **StyleTTS2 API**: `http://localhost:8000` - Text-to-speech generation
- **Seed-VC API**: `http://localhost:8001` - Voice conversion and cloning  
- **Make-An-Audio API**: `http://localhost:8002` - Text-to-audio generation

## 📁 Project Structure

```
Creative-AI/
├── Docker-compose.yml          # Multi-service deployment configuration
├── Make-An-Audio/             # Text-to-audio generation
│   ├── api.py                 # FastAPI service
│   ├── main.py               # Core inference logic
│   └── requirements.txt      # Python dependencies
├── seed-vc/                  # Voice conversion and cloning
│   ├── api.py               # FastAPI service
│   ├── inference.py         # Voice conversion inference
│   └── real-time-gui.py     # Real-time GUI application
├── StyleTTS2/               # Text-to-speech synthesis
│   ├── api.py              # FastAPI service
│   ├── models.py           # Model architectures
│   └── train_*.py          # Training scripts
└── StyleTTS2FineTune/      # Fine-tuning utilities and datasets
```

## 🔧 Configuration & Troubleshooting

### Environment Variables
Create `.env` files in each service directory:

```env
# Common for all services
API_KEY=12345
AWS_REGION=eu-north-1
```

### GPU Configuration
- Ensure CUDA 11.8 compatibility
- Each service requires ~4-8GB VRAM
- Docker containers automatically use `--gpus all` flag

### Port Configuration
- StyleTTS2: Port 8000
- Seed-VC: Port 8001  
- Make-An-Audio: Port 8002

### Common Issues & Solutions

1. **NLTK Download Issues (StyleTTS2):**
   ```python
   # Create download_nltk_resources.py
   import nltk
   nltk.download('punkt')
   nltk.download('punkt_tab', raise_on_error=False)
   ```

2. **PyTorch Model Loading:**
   ```python
   # Add weights_only=False to torch.load() calls
   torch.load(checkpoint_path, weights_only=False)
   ```

3. **Import Errors (Make-An-Audio):**
   ```python
   # Replace deprecated import
   # from pytorch_lightning.utilities.distributed import rank_zero_only
   from pytorch_lightning.utilities.rank_zero import rank_zero_only
   ```

4. **FFmpeg Installation Required:**
   - Windows: Install from official website
   - Add to system PATH
   - Required for audio processing in all services

## ☁️ AWS Deployment

### Prerequisites
- AWS CLI installed and configured
- ECR repositories created for each service
- EC2 instances with GPU support (G5.xlarge recommended)

### Setup ECR Repositories
```powershell
# Create repositories in AWS ECR
aws ecr create-repository --repository-name styletts2-api --region eu-north-1
aws ecr create-repository --repository-name seed-vc-api --region eu-north-1  
aws ecr create-repository --repository-name make-an-audio-api --region eu-north-1
```

### Build and Push to ECR
```powershell
# Login to ECR
aws ecr get-login-password --region eu-north-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.eu-north-1.amazonaws.com

# Build and push each service
docker build -f StyleTTS2/Dockerfile.api -t styletts2-api .
docker tag styletts2-api:latest <account-id>.dkr.ecr.eu-north-1.amazonaws.com/styletts2-api:latest
docker push <account-id>.dkr.ecr.eu-north-1.amazonaws.com/styletts2-api:latest

# Repeat for other services...
```

### EC2 Deployment
```bash
# On EC2 instance
sudo usermod -a -G docker ec2-user
newgrp docker
aws ecr get-login-password --region eu-north-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.eu-north-1.amazonaws.com

# Deploy with Docker Compose
docker-compose up -d
```

## 📊 API Usage Examples

All APIs require authorization header: `Authorization: 12345`

### StyleTTS2 - Text-to-Speech
```powershell
# Health check
curl -X GET "http://localhost:8000/health" -H "Authorization: 12345"

# List voices
curl -X GET "http://localhost:8000/voices" -H "Authorization: 12345"

# Generate speech
curl -X POST "http://localhost:8000/generate" -H "Authorization: 12345" -H "Content-Type: application/json" -d '{"text": "Hello World! This is a test.", "target_voice": "Sam_Altman"}'

# Upload new voice
curl -X POST "http://localhost:8000/upload-voice" -H "Authorization: 12345" -F "file=@voice.wav" -F "voice_name=new_voice"

# Delete voice
curl -X DELETE "http://localhost:8000/voices/voice_name" -H "Authorization: 12345"
```

### Seed-VC - Voice Conversion
```powershell
# Health check
curl -X GET "http://localhost:8001/health" -H "Authorization: 12345"

# List available voices
curl -X GET "http://localhost:8001/voices" -H "Authorization: 12345"

# Convert voice
curl -X POST "http://localhost:8001/convert" -H "Authorization: 12345" -H "Content-Type: application/json" -d '{"source_audio_key": "seed-vc-audio-uploads/source.wav", "target_voice": "Sam_Altman"}'

# Upload reference voice
curl -X POST "http://localhost:8001/upload-voice" -H "Authorization: 12345" -F "file=@reference.wav" -F "voice_name=new_reference"
```

### Make-An-Audio - Text-to-Audio
```powershell
# Health check
curl -X GET "http://localhost:8002/health" -H "Authorization: 12345"

# Generate audio from text
curl -X POST "http://localhost:8002/generate" -H "Authorization: 12345" -H "Content-Type: application/json" -d '{"prompt": "Lion roaring in the desert"}'
```

## 🎨 Fine-tuning & Dataset Preparation

### StyleTTS2 Fine-tuning

1. **Prepare Audio Data:**
   ```powershell
   # Add 15-minute audio file to StyleTTS2FineTune/makeDataset/tools/audio/
   # Use WhisperX for transcription
   cd StyleTTS2FineTune/makeDataset/tools
   Get-ChildItem -Path './audio' -Filter *.wav | ForEach-Object { 
       whisperx $_.FullName --model large-v2 --align_model WAV2VEC2_ASR_LARGE_LV60K_960H --compute_type int8 
   }
   ```

2. **Process Dataset:**
   ```powershell
   python srtsegmenter.py
   python add_padding.py
   python phonemized.py --language en-us
   ```

3. **Setup Training Data:**
   - Copy segmented audio to `StyleTTS2/Data/wavs/`
   - Add training lists to `StyleTTS2/Data/`
   - Configure `StyleTTS2/Configs/config_ft.yml`

4. **Train Model:**
   ```powershell
   cd StyleTTS2
   accelerate launch --mixed_precision=fp16 --num_processes=1 train_finetune_accelerate.py --config_path ./Configs/config_ft.yml
   ```

### Seed-VC Custom Training

Minimal training requirements:
- Copy StyleTTS2 generated wavs to `seed-vc/dataset/`
- Fine-tune with just 100 steps (2 minutes on T4 GPU)
- Use provided Docker training container

### Make-An-Audio Model Setup

1. **Download Required Models:**
   - `maa1_full.ckpt` - Main model checkpoint
   - `bigvgan` folder with `args.yml` and `best_netG.pt`
   - CLAP weights: `CLAP_weights_2022.pth`

2. **File Structure:**
   ```
   useful_ckpts/
   ├── bigvgan/
   │   ├── args.yml
   │   └── best_netG.pt
   ├── CLAP/
   │   ├── config.yml
   │   └── CLAP_weights_2022.pth
   └── maa1_full.ckpt
   ```

## 🔍 Model Details

### StyleTTS2
- **Architecture**: Style diffusion + adversarial training
- **Quality**: Surpasses human recordings on LJSpeech
- **Capabilities**: Single/multi-speaker, zero-shot adaptation
- **Paper**: [StyleTTS 2: Towards Human-Level Text-to-Speech](https://arxiv.org/abs/2306.07691)

### Seed-VC  
- **Architecture**: Zero-shot voice conversion
- **Latency**: ~300ms algorithm + ~100ms device
- **Data Requirements**: 1-30 seconds reference audio
- **Paper**: [Seed-VC](https://arxiv.org/abs/2411.09943)

### Make-An-Audio
- **Architecture**: Prompt-enhanced diffusion models
- **Quality**: High-fidelity audio generation
- **Modalities**: Text-to-audio, audio inpainting
- **Paper**: [Make-An-Audio](https://arxiv.org/abs/2301.12661)

## � Project Structure (Updated)

```
Creative-AI/
├── Backend.txt                    # Complete setup instructions
├── Docker-compose.yml            # Multi-service deployment
├── README.md                     # This documentation
├── Creative-AI-Frontend/         # Web frontend interface (optional)
│   └── [Frontend files]          # React/Vue/HTML interface
├── Make-An-Audio/               # Text-to-audio generation
│   ├── api.py                   # FastAPI service
│   ├── gen_wav.py              # Core audio generation
│   ├── requirements.txt        # Python dependencies
│   ├── useful_ckpts/           # Model checkpoints
│   │   ├── maa1_full.ckpt
│   │   ├── bigvgan/
│   │   └── CLAP/
│   └── msclap/                 # CLAP model weights
├── seed-vc/                    # Voice conversion and cloning
│   ├── api.py                  # FastAPI service
│   ├── inference.py            # Voice conversion inference
│   ├── train.py               # Training scripts
│   ├── dataset/               # Training audio data
│   └── runs/                  # Training outputs
├── StyleTTS2/                 # Text-to-speech synthesis
│   ├── api.py                 # FastAPI service
│   ├── models.py              # Model architectures
│   ├── train_finetune_*.py    # Fine-tuning scripts
│   ├── Data/                  # Training data
│   │   └── wavs/             # Audio files
│   └── Models/LibriTTS/       # Pre-trained models
└── StyleTTS2FineTune/         # Dataset preparation tools
    ├── curate.ipynb           # Data curation
    ├── PhonemeCoverage.ipynb  # Phoneme analysis
    └── makeDataset/           # Audio processing tools
        └── tools/
            ├── srtsegmenter.py
            ├── add_padding.py
            └── phonemized.py
```

## � Performance & Monitoring

### Expected Performance
- **StyleTTS2**: ~2-5 seconds for 10-second audio generation
- **Seed-VC**: ~300ms algorithm latency + 100ms device latency
- **Make-An-Audio**: ~30-60 seconds for 10-second audio (100 DDIM steps)

### Resource Usage
- **Memory**: 4-8GB VRAM per service
- **Storage**: Models require ~20GB total space
- **CPU**: Multi-core recommended for preprocessing

### Monitoring
```powershell
# Check service logs
docker-compose logs -f styletts2-api
docker-compose logs -f seed-vc-api  
docker-compose logs -f make-an-audio-api

# Monitor resource usage
docker stats
```

## 📄 License

This project combines multiple models with different licenses:
- **StyleTTS2**: MIT License
- **Seed-VC**: Apache License 2.0  
- **Make-An-Audio**: Apache License 2.0

Please review individual model licenses in their respective directories.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description

## 🔗 Resources

- [StyleTTS2 Demo](https://styletts2.github.io/)
- [Seed-VC Demo](https://plachtaa.github.io/seed-vc/)
- [Make-An-Audio Demo](https://text-to-audio.github.io/)
- [Hugging Face Spaces](https://huggingface.co/spaces/AIGC-Audio/Make_An_Audio)

## � Additional Resources

### Model Documentation
- [StyleTTS2 Paper](https://arxiv.org/abs/2306.07691) - Style diffusion + adversarial training
- [Seed-VC Paper](https://arxiv.org/abs/2411.09943) - Zero-shot voice conversion
- [Make-An-Audio Paper](https://arxiv.org/abs/2301.12661) - Text-to-audio generation

### Live Demos
- [StyleTTS2 Demo](https://styletts2.github.io/)
- [Seed-VC Demo](https://plachtaa.github.io/seed-vc/)
- [Make-An-Audio Demo](https://text-to-audio.github.io/)

### Training Resources
- [WhisperX](https://github.com/m-bain/whisperx) - Speech recognition for dataset creation
- [FFmpeg Installation Guide](https://www.youtube.com/watch?v=JR36oH35Fgg)
- [AWS EC2 GPU Instances](https://aws.amazon.com/ec2/instance-types/g5/)

### Frontend & Integration
- [Creative-AI-Frontend](https://github.com/babdellghani/Creative-AI-Frontend.git) - Web interface for all services

## 📞 Support & Contributing

### Getting Help
1. Review individual service documentation in each folder
2. Check AWS deployment logs if using cloud deployment
3. Verify all model checkpoints are downloaded correctly

### Contributing
Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch  
3. Test thoroughly with all three services
4. Submit a pull request with detailed description

---

**Production Note**: This platform combines research models. For production deployment:
- Implement proper error handling and validation
- Add authentication beyond simple API keys  
- Monitor resource usage and implement auto-scaling
- Consider model optimization for your specific use case
- Test thoroughly with your expected workloads
