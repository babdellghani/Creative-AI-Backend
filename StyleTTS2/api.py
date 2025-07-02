from contextlib import asynccontextmanager
import uuid
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Header, UploadFile, File, Form
import logging
import os
import re
from fastapi.security import APIKeyHeader
import numpy as np
import soundfile as sf
import boto3
from pydantic import BaseModel
from libri_inference import StyleTTS2Inference
import tempfile
from urllib.parse import urlparse
import librosa
from typing import Dict


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
synthesizer = None
reference_style = None
API_KEY = os.getenv("API_KEY")

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


async def verify_api_key(authorization: str = Header(None)):
    if not authorization:
        logger.warning("No API key provided")
        raise HTTPException(status_code=401, detail="API key is missing")

    if authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "")
    else:
        token = authorization

    if token != API_KEY:
        logger.warning("Invalid API key provided")
        raise HTTPException(status_code=401, detail="Invalid API key")

    return token


def get_s3_client():
    client_kwargs = {'region_name': os.getenv("AWS_REGION", "us-east-1")}

    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
        client_kwargs.update({
            'aws_access_key_id': os.getenv("AWS_ACCESS_KEY_ID"),
            'aws_secret_access_key': os.getenv("AWS_SECRET_ACCESS_KEY")
        })

    return boto3.client('s3', **client_kwargs)


s3_client = get_s3_client()

S3_PREFIX = os.getenv("S3_PREFIX", "styletts2-output")
S3_BUCKET = os.getenv("S3_BUCKET", "creative-ai-tools")


def download_s3_file(s3_url, local_path):
    """Download a file from S3 to local path"""
    try:
        # Parse S3 URL to get bucket and key
        if s3_url.startswith('s3://'):
            parsed = urlparse(s3_url)
            bucket = parsed.netloc
            key = parsed.path.lstrip('/')
        else:
            # Assume it's already a local path
            return s3_url

        logger.info(f"Downloading {s3_url} to {local_path}")
        s3_client.download_file(bucket, key, local_path)
        return local_path
    except Exception as e:
        logger.error(f"Failed to download {s3_url}: {e}")
        raise


def get_local_voice_path(voice_key):
    """Get or download voice file and return local path"""
    s3_url = TARGET_VOICES[voice_key]

    # If it's already a local path, return as is
    if not s3_url.startswith('s3://'):
        return s3_url

    # Create a local temp file
    temp_dir = "/tmp/voice_cache"
    os.makedirs(temp_dir, exist_ok=True)

    # Create a safe filename
    safe_filename = f"{voice_key}.wav"
    local_path = os.path.join(temp_dir, safe_filename)

    # Download if not already cached
    if not os.path.exists(local_path):
        download_s3_file(s3_url, local_path)

    return local_path


def refresh_target_voices():
    """Refresh the TARGET_VOICES dictionary from S3"""
    global TARGET_VOICES
    try:
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix='target-voices/'
        )

        TARGET_VOICES = {}
        if 'Contents' in response:
            for item in response['Contents']:
                # Skip directories and empty objects
                if item['Key'].endswith('/') or item['Size'] == 0:
                    continue

                # Extract filename without extension and clean it up
                filename = item['Key'].split('/')[-1]
                if filename:  # Make sure it's not empty
                    key = os.path.splitext(filename)[0]  # Remove extension
                    # Clean up the key to make it more API-friendly
                    # Replace special chars with underscore
                    key = re.sub(r'[^\w\-_]', '_', key)
                    # Replace multiple underscores with single
                    key = re.sub(r'_+', '_', key)
                    key = key.strip('_')  # Remove leading/trailing underscores

                    if key:  # Make sure key is not empty after cleaning
                        TARGET_VOICES[key] = f"s3://{S3_BUCKET}/{item['Key']}"

        # If no voices found in S3, use default local voices
        if not TARGET_VOICES:
            TARGET_VOICES = {
                "amused": "Models/LibriTTS/Voice/amused.wav",
                "anger": "Models/LibriTTS/Voice/anger.wav",
                "computer": "Models/LibriTTS/Voice/computer.wav",
                "disgusted": "Models/LibriTTS/Voice/disgusted.wav",
                "gavin": "Models/LibriTTS/Voice/Gavin.wav",
                "man1": "Models/LibriTTS/Voice/man1.wav",
                "man2": "Models/LibriTTS/Voice/man2.wav",
                "man3": "Models/LibriTTS/Voice/man3.wav",
                "man4": "Models/LibriTTS/Voice/man4.wav",
                "man5": "Models/LibriTTS/Voice/man5.wav",
                "man6": "Models/LibriTTS/Voice/man6.wav",
                "man7": "Models/LibriTTS/Voice/man7.wav",
                "man8": "Models/LibriTTS/Voice/man8.wav",
                "manChinese1": "Models/LibriTTS/Voice/manChinese1.wav",
                "manChinese2": "Models/LibriTTS/Voice/manChinese2.wav",
                "manChinese3": "Models/LibriTTS/Voice/manChinese3.wav",
                "manChinese4": "Models/LibriTTS/Voice/manChinese4.wav",
                "manjp1": "Models/LibriTTS/Voice/manjp1.wav",
                "nima": "Models/LibriTTS/Voice/Nima.wav",
                "sam": "Models/LibriTTS/Voice/Sam Altman.wav",
                "sleepy": "Models/LibriTTS/Voice/sleepy.wav",
                "technopolis": "Models/LibriTTS/Voice/TECHNOPOLIS.wav",
                "trump": "Models/LibriTTS/Voice/trump.wav",
                "vinay": "Models/LibriTTS/Voice/Vinay.wav",
                "Wiz Khalifa": "Models/LibriTTS/Voice/Wiz Khalifa.wav",
                "woman1": "Models/LibriTTS/Voice/woman1.wav",
                "woman2": "Models/LibriTTS/Voice/woman2.wav",
                "woman3": "Models/LibriTTS/Voice/woman3.wav",
                "woman4": "Models/LibriTTS/Voice/woman4.wav",
                "woman5": "Models/LibriTTS/Voice/woman5.wav",
                "woman6": "Models/LibriTTS/Voice/woman6.wav",
                "woman7": "Models/LibriTTS/Voice/woman7.wav",
                "womanChinese1": "Models/LibriTTS/Voice/womanChinese1.wav",
                "womanChinese2": "Models/LibriTTS/Voice/womanChinese2.wav",
                "womanChinese3": "Models/LibriTTS/Voice/womanChinese3.wav",
                "womanChinese4": "Models/LibriTTS/Voice/womanChinese4.wav",
                "womanjp1": "Models/LibriTTS/Voice/womanjp1.wav",
                "womanjp2": "Models/LibriTTS/Voice/womanjp2.wav",
                "yinghao": "Models/LibriTTS/Voice/Yinghao.wav",
            }
            logger.warning(
                "No target voices found in S3, using default voices")

        logger.info(f"Refreshed {len(TARGET_VOICES)} target voices from S3")
        return TARGET_VOICES
    except Exception as e:
        logger.error(f"Failed to refresh target voices from S3: {e}")
        return TARGET_VOICES


def validate_audio_file(file_path: str) -> bool:
    """Validate if the uploaded file is a valid audio file"""
    try:
        # Try to load the audio file
        audio, sr = librosa.load(file_path, sr=None)

        # Check if audio has content
        if len(audio) == 0:
            return False

        # Check duration (should be at least 1 second, max 30 seconds for voice cloning)
        duration = len(audio) / sr
        if duration < 1 or duration > 30:
            logger.warning(
                f"Audio duration {duration:.2f}s is outside recommended range (1-30s)")
            return False

        return True
    except Exception as e:
        logger.error(f"Audio validation failed: {e}")
        return False


try:
    TARGET_VOICES = refresh_target_voices()
    logger.info(f"Available target voices: {list(TARGET_VOICES.keys())}")
except Exception as e:
    logger.error(f"Failed to list target voices from S3: {e}")
    TARGET_VOICES = {
        "amused": "Models/LibriTTS/Voice/amused.wav",
        "anger": "Models/LibriTTS/Voice/anger.wav",
        "computer": "Models/LibriTTS/Voice/computer.wav",
        "disgusted": "Models/LibriTTS/Voice/disgusted.wav",
        "gavin": "Models/LibriTTS/Voice/Gavin.wav",
        "man1": "Models/LibriTTS/Voice/man1.wav",
        "man2": "Models/LibriTTS/Voice/man2.wav",
        "man3": "Models/LibriTTS/Voice/man3.wav",
        "man4": "Models/LibriTTS/Voice/man4.wav",
        "man5": "Models/LibriTTS/Voice/man5.wav",
        "man6": "Models/LibriTTS/Voice/man6.wav",
        "man7": "Models/LibriTTS/Voice/man7.wav",
        "man8": "Models/LibriTTS/Voice/man8.wav",
        "manChinese1": "Models/LibriTTS/Voice/manChinese1.wav",
        "manChinese2": "Models/LibriTTS/Voice/manChinese2.wav",
        "manChinese3": "Models/LibriTTS/Voice/manChinese3.wav",
        "manChinese4": "Models/LibriTTS/Voice/manChinese4.wav",
        "manjp1": "Models/LibriTTS/Voice/manjp1.wav",
        "nima": "Models/LibriTTS/Voice/Nima.wav",
        "sam": "Models/LibriTTS/Voice/Sam Altman.wav",
        "sleepy": "Models/LibriTTS/Voice/sleepy.wav",
        "technopolis": "Models/LibriTTS/Voice/TECHNOPOLIS.wav",
        "trump": "Models/LibriTTS/Voice/trump.wav",
        "vinay": "Models/LibriTTS/Voice/Vinay.wav",
        "Wiz Khalifa": "Models/LibriTTS/Voice/Wiz Khalifa.wav",
        "woman1": "Models/LibriTTS/Voice/woman1.wav",
        "woman2": "Models/LibriTTS/Voice/woman2.wav",
        "woman3": "Models/LibriTTS/Voice/woman3.wav",
        "woman4": "Models/LibriTTS/Voice/woman4.wav",
        "woman5": "Models/LibriTTS/Voice/woman5.wav",
        "woman6": "Models/LibriTTS/Voice/woman6.wav",
        "woman7": "Models/LibriTTS/Voice/woman7.wav",
        "womanChinese1": "Models/LibriTTS/Voice/womanChinese1.wav",
        "womanChinese2": "Models/LibriTTS/Voice/womanChinese2.wav",
        "womanChinese3": "Models/LibriTTS/Voice/womanChinese3.wav",
        "womanChinese4": "Models/LibriTTS/Voice/womanChinese4.wav",
        "womanjp1": "Models/LibriTTS/Voice/womanjp1.wav",
        "womanjp2": "Models/LibriTTS/Voice/womanjp2.wav",
        "yinghao": "Models/LibriTTS/Voice/Yinghao.wav",
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global synthesizer, reference_style
    logger.info("Loading StyleTTS2 model...")
    try:
        synthesizer = StyleTTS2Inference(
            config_path=os.getenv("CONFIG_PATH", "Models/LibriTTS/config.yml"),
            model_path=os.getenv(
                "MODEL_PATH", "Models/LibriTTS/epochs_2nd_00020.pth")
        )

        logger.info("StyleTTS2 model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load StyleTTS2 model: {e}")
        raise

    yield

    logger.info("Shutting down StyleTTS2 API")

app = FastAPI(title="StyleTTS2 API",
              lifespan=lifespan)


def text_chunker(text, max_chunk_size=125):
    if len(text) <= max_chunk_size:
        return [text]

    chunks = []
    current_pos = 0
    text_len = len(text)

    while current_pos < text_len:
        if current_pos + max_chunk_size >= text_len:
            chunks.append(text[current_pos:])
            break

        chunk_end = current_pos + max_chunk_size
        search_text = text[current_pos:chunk_end]

        sentence_ends = [m.end() for m in re.finditer(r'[.!?]+', search_text)]

        if sentence_ends:
            last_sentence_end = sentence_ends[-1]
            chunks.append(text[current_pos:current_pos + last_sentence_end])
            current_pos += last_sentence_end
        else:
            last_space = search_text.rfind(' ')
            if last_space > 0:
                chunks.append(text[current_pos:current_pos + last_space])
                current_pos += last_space + 1
            else:
                chunks.append(text[current_pos:chunk_end])
                current_pos = chunk_end

        while current_pos < text_len and text[current_pos].isspace():
            current_pos += 1

    return chunks


class TextOnlyRequest(BaseModel):
    text: str
    target_voice: str


class VoiceUploadResponse(BaseModel):
    message: str
    voice_key: str
    s3_key: str
    voices: Dict[str, str] = None


@app.post("/upload-voice", dependencies=[Depends(verify_api_key)], response_model=VoiceUploadResponse)
async def upload_voice(
    background_tasks: BackgroundTasks,
    voice_name: str = Form(None),
    file: UploadFile = File(...)
):
    """
    Upload a voice file to be used as a target voice for speech synthesis.

    Args:
        voice_name: Optional custom name for the voice. If not provided, uses the filename.
        file: Audio file (WAV, MP3, FLAC, etc.) - recommended 3-10 seconds of clear speech

    Returns:
        Success message with the voice key that can be used in /generate endpoint
    """

    # Validate file type - check both content type and file extension
    valid_extensions = ['.wav', '.mp3',
                        '.flac', '.m4a', '.aac', '.ogg', '.wma']
    file_extension = os.path.splitext(file.filename.lower())[
        1] if file.filename else ''

    is_valid_content_type = file.content_type and file.content_type.startswith(
        'audio/')
    is_valid_extension = file_extension in valid_extensions

    if not (is_valid_content_type or is_valid_extension):
        raise HTTPException(
            status_code=400,
            detail=f"File must be an audio file. Supported formats: {', '.join(valid_extensions)}"
        )

    # Validate file size (max 10MB)
    if hasattr(file, 'size') and file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File size must be less than 10MB"
        )

    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            # Read and write the uploaded file
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Validate the audio file
        if not validate_audio_file(temp_file_path):
            background_tasks.add_task(os.remove, temp_file_path)
            raise HTTPException(
                status_code=400,
                detail="Invalid audio file or duration outside recommended range (1-30 seconds)"
            )

        # Determine voice name
        if voice_name:
            clean_voice_name = re.sub(r'[^\w\-_]', '_', voice_name)
            clean_voice_name = re.sub(r'_+', '_', clean_voice_name).strip('_')
        else:
            # Use original filename without extension
            original_name = os.path.splitext(file.filename)[
                0] if file.filename else "voice"
            clean_voice_name = re.sub(r'[^\w\-_]', '_', original_name)
            clean_voice_name = re.sub(r'_+', '_', clean_voice_name).strip('_')

        if not clean_voice_name:
            clean_voice_name = f"voice_{uuid.uuid4().hex[:8]}"

        # Check if voice name already exists
        if clean_voice_name in TARGET_VOICES:
            raise HTTPException(
                status_code=400,
                detail=f"Voice name '{clean_voice_name}' already exists. Please choose a different name."
            )

        # Convert to WAV format with standard settings for StyleTTS2
        output_path = temp_file_path.replace('.wav', '_processed.wav')
        try:
            # Load and resample to 24kHz (StyleTTS2's expected sample rate)
            audio, sr = librosa.load(temp_file_path, sr=24000)
            sf.write(output_path, audio, 24000)
        except Exception as e:
            background_tasks.add_task(os.remove, temp_file_path)
            raise HTTPException(
                status_code=400,
                detail=f"Failed to process audio file: {str(e)}"
            )

        # Upload to S3
        s3_key = f"target-voices/{clean_voice_name}.wav"
        try:
            s3_client.upload_file(output_path, S3_BUCKET, s3_key)
            logger.info(f"Uploaded voice '{clean_voice_name}' to S3: {s3_key}")
        except Exception as e:
            background_tasks.add_task(os.remove, temp_file_path)
            background_tasks.add_task(os.remove, output_path)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to upload to S3: {str(e)}"
            )

        # Clean up temp files
        background_tasks.add_task(os.remove, temp_file_path)
        background_tasks.add_task(os.remove, output_path)

        # Refresh the target voices list
        refresh_target_voices()

        return VoiceUploadResponse(
            message=f"Voice '{clean_voice_name}' uploaded successfully",
            voice_key=clean_voice_name,
            s3_key=s3_key,
            voices=TARGET_VOICES
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload voice: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload voice: {str(e)}"
        )


@app.post("/generate", dependencies=[Depends(verify_api_key)])
async def generate_speech(request: TextOnlyRequest, background_tasks: BackgroundTasks):
    if len(request.text) > 5000:
        raise HTTPException(
            status_code=400, detail="Text length exceeds the limit of 5000 characters")

    if not synthesizer:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if request.target_voice not in TARGET_VOICES:
        raise HTTPException(
            status_code=400, detail=f"Target voice not supported. Choose from: {', '.join(TARGET_VOICES.keys())}")

    try:
        # Get local path for the voice file (download if needed)
        ref_audio_path = get_local_voice_path(request.target_voice)

        logger.info(
            f"Using voice {request.target_voice} from local path: {ref_audio_path}")

        # Compute style for requested voice
        current_style = synthesizer.compute_style(ref_audio_path)

        # Generate a unique filename
        audio_id = str(uuid.uuid4())
        output_filename = f"{audio_id}.wav"
        local_path = f"/tmp/{output_filename}"

        # Split text into manageable chunks
        text_chunks = text_chunker(request.text)
        logger.info(f"Text split into chunks: {len(text_chunks)}")

        audio_segments = []

        for i, chunk in enumerate(text_chunks):
            logger.info(f"Processing chunk {i+1}/{len(text_chunks)}")

            audio_chunk = synthesizer.inference(
                text=chunk,
                ref_s=current_style
            )

            audio_segments.append(audio_chunk)

            if i < len(text_chunks) - 1:
                silence = np.zeros(int(24000 * 0.3))
                audio_segments.append(silence)

        if len(audio_segments) > 1:
            full_audio = np.concatenate(audio_segments)
        else:
            full_audio = audio_segments[0]

        sf.write(local_path, full_audio, 24000)

        # Upload to S3
        s3_key = f"{S3_PREFIX}/{output_filename}"
        s3_client.upload_file(local_path, S3_BUCKET, s3_key)

        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': S3_BUCKET, 'Key': s3_key},
            ExpiresIn=3600
        )

        background_tasks.add_task(os.remove, local_path)

        return {
            "audio_url": presigned_url,
            "s3_key": s3_key
        }
    except Exception as e:
        logger.error(f"Failed to generate speech: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate speech: {str(e)}")


@app.get("/voices", dependencies=[Depends(verify_api_key)])
async def list_voices():
    """List all available target voices"""
    # Refresh voices from S3 to get the latest list
    current_voices = refresh_target_voices()
    return {
        "voices": list(current_voices.keys()),
        "count": len(current_voices),
        "voice_details": {k: {"s3_url": v} for k, v in current_voices.items()}
    }


@app.delete("/voices/{voice_key}", dependencies=[Depends(verify_api_key)])
async def delete_voice(voice_key: str):
    """Delete a target voice"""
    if voice_key not in TARGET_VOICES:
        raise HTTPException(
            status_code=404,
            detail=f"Voice '{voice_key}' not found"
        )

    # Don't allow deletion of default voices
    if not TARGET_VOICES[voice_key].startswith('s3://'):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete default voice '{voice_key}'"
        )

    try:
        # Extract S3 key from URL
        s3_url = TARGET_VOICES[voice_key]
        parsed = urlparse(s3_url)
        s3_key = parsed.path.lstrip('/')

        # Delete from S3
        s3_client.delete_object(Bucket=S3_BUCKET, Key=s3_key)
        logger.info(f"Deleted voice '{voice_key}' from S3: {s3_key}")

        # Refresh the voices list
        refresh_target_voices()

        return {
            "message": f"Voice '{voice_key}' deleted successfully",
            "remaining_voices": list(TARGET_VOICES.keys())
        }

    except Exception as e:
        logger.error(f"Failed to delete voice {voice_key}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete voice: {str(e)}"
        )


@app.get("/health", dependencies=[Depends(verify_api_key)])
async def health_check():
    if synthesizer:
        return {"status": "healthy", "model": "loaded"}
    return {"status": "unhealthy", "model": "not loaded"}
