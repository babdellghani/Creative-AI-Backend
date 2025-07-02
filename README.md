# Creative-AI: Comprehensive Audio Generation and Voice Conversion Suite

A powerful Docker-based AI audio platform that combines three state-of-the-art models for complete audio generation and voice conversion capabilities.

## 🎯 Overview

Creative-AI integrates three cutting-edge AI models into a unified platform:

- **StyleTTS2**: Human-level text-to-speech synthesis
- **Seed-VC**: Zero-shot voice conversion and real-time voice cloning
- **Make-An-Audio**: Text-to-audio generation with diffusion models

All services are containerized and can be deployed with a single Docker Compose command.

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

## 🛠️ Quick Start

### Prerequisites
- Docker and Docker Compose
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit

### Installation

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd Creative-AI
```

2. **Start all services:**
```bash
docker-compose up -d
```

### Service Endpoints

Once running, the following APIs will be available:

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

## 🔧 Configuration

### GPU Requirements
Each service requires NVIDIA GPU access. The Docker Compose configuration automatically allocates one GPU per service.

### Port Configuration
- StyleTTS2: Port 8000
- Seed-VC: Port 8001  
- Make-An-Audio: Port 8002

To change ports, modify the `ports` section in `Docker-compose.yml`.

## 📊 Usage Examples

### StyleTTS2 - Text-to-Speech
```python
import requests

response = requests.post("http://localhost:8000/generate", 
    json={"text": "Hello, this is StyleTTS2 speaking!"})
```

### Seed-VC - Voice Conversion
```python
import requests

# Upload reference audio and target audio for voice conversion
files = {
    'reference_audio': open('reference.wav', 'rb'),
    'target_audio': open('target.wav', 'rb')
}
response = requests.post("http://localhost:8001/convert", files=files)
```

### Make-An-Audio - Text-to-Audio
```python
import requests

response = requests.post("http://localhost:8002/generate", 
    json={"prompt": "Sound of rain falling on leaves"})
```

## 🎨 Fine-tuning

### StyleTTS2 Fine-tuning
Located in `StyleTTS2FineTune/`, this section includes:
- Dataset preparation tools (`curate.ipynb`)
- Phoneme coverage analysis (`PhonemeCoverage.ipynb`)
- Sample datasets and configurations

### Seed-VC Custom Training
Minimal data requirements:
- 1 utterance per speaker minimum
- 100 training steps (2 minutes on T4 GPU)
- Custom dataset preparation tools included

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

## 🛡️ System Requirements

- **OS**: Linux, Windows, macOS
- **GPU**: NVIDIA GPU with 8GB+ VRAM recommended
- **RAM**: 16GB+ system RAM recommended
- **Storage**: 50GB+ free space for models and data
- **Docker**: Latest version with NVIDIA Container Toolkit

## 🐛 Troubleshooting

### Common Issues

1. **GPU not detected**: Ensure NVIDIA Container Toolkit is installed
2. **Port conflicts**: Check if ports 8000-8002 are available
3. **Memory issues**: Reduce batch size or use smaller models
4. **Model loading errors**: Verify model checkpoints are properly downloaded

### Logs
Check service logs:
```bash
docker-compose logs styletts2-api
docker-compose logs seedvc-api
docker-compose logs make-an-audio-api
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

## 📞 Support

For issues and questions:
- Check existing GitHub issues
- Review model-specific documentation
- Contact maintainers through GitHub discussions

---

**Note**: This is a research-oriented platform. Performance may vary based on hardware and input quality. For production use, additional optimization and testing are recommended.
