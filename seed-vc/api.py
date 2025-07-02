from inference import load_models, process_voice_conversion
import logging
import os
import uuid
from contextlib import asynccontextmanager
from tempfile import NamedTemporaryFile
import tempfile
import re
from urllib.parse import urlparse
from typing import Dict
import librosa
import boto3
import torchaudio
import soundfile as sf
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, UploadFile, File, Form
from fastapi.security import APIKeyHeader
from pydantic import BaseModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
models = None
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

S3_PREFIX = os.getenv("S3_PREFIX", "seedvc-outputs")
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
                "amused": "examples/reference/amused.wav",
                "anger": "examples/reference/anger.wav",
                "computer": "examples/reference/computer.wav",
                "disgusted": "examples/reference/disgusted.wav",
                "gavin": "examples/reference/Gavin.wav",
                "man1": "examples/reference/man1.wav",
                "man2": "examples/reference/man2.wav",
                "man3": "examples/reference/man3.wav",
                "man4": "examples/reference/man4.wav",
                "man5": "examples/reference/man5.wav",
                "man6": "examples/reference/man6.wav",
                "man7": "examples/reference/man7.wav",
                "man8": "examples/reference/man8.wav",
                "manChinese1": "examples/reference/manChinese1.wav",
                "manChinese2": "examples/reference/manChinese2.wav",
                "manChinese3": "examples/reference/manChinese3.wav",
                "manChinese4": "examples/reference/manChinese4.wav",
                "manjp1": "examples/reference/manjp1.wav",
                "nima": "examples/reference/Nima.wav",
                "sam": "examples/reference/Sam Altman.wav",
                "sleepy": "examples/reference/sleepy.wav",
                "technopolis": "examples/reference/TECHNOPOLIS.wav",
                "trump": "examples/reference/trump.wav",
                "vinay": "examples/reference/Vinay.wav",
                "Wiz Khalifa": "examples/reference/Wiz Khalifa.wav",
                "woman1": "examples/reference/woman1.wav",
                "woman2": "examples/reference/woman2.wav",
                "woman3": "examples/reference/woman3.wav",
                "woman4": "examples/reference/woman4.wav",
                "woman5": "examples/reference/woman5.wav",
                "woman6": "examples/reference/woman6.wav",
                "woman7": "examples/reference/woman7.wav",
                "womanChinese1": "examples/reference/womanChinese1.wav",
                "womanChinese2": "examples/reference/womanChinese2.wav",
                "womanChinese3": "examples/reference/womanChinese3.wav",
                "womanChinese4": "examples/reference/womanChinese4.wav",
                "womanjp1": "examples/reference/womanjp1.wav",
                "womanjp2": "examples/reference/womanjp2.wav",
                "yinghao": "examples/reference/Yinghao.wav",
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
        "amused": "examples/reference/amused.wav",
        "anger": "examples/reference/anger.wav",
        "computer": "examples/reference/computer.wav",
        "disgusted": "examples/reference/disgusted.wav",
        "gavin": "examples/reference/Gavin.wav",
        "man1": "examples/reference/man1.wav",
        "man2": "examples/reference/man2.wav",
        "man3": "examples/reference/man3.wav",
        "man4": "examples/reference/man4.wav",
        "man5": "examples/reference/man5.wav",
        "man6": "examples/reference/man6.wav",
        "man7": "examples/reference/man7.wav",
        "man8": "examples/reference/man8.wav",
        "manChinese1": "examples/reference/manChinese1.wav",
        "manChinese2": "examples/reference/manChinese2.wav",
        "manChinese3": "examples/reference/manChinese3.wav",
        "manChinese4": "examples/reference/manChinese4.wav",
        "manjp1": "examples/reference/manjp1.wav",
        "nima": "examples/reference/Nima.wav",
        "sam": "examples/reference/Sam Altman.wav",
        "sleepy": "examples/reference/sleepy.wav",
        "technopolis": "examples/reference/TECHNOPOLIS.wav",
        "trump": "examples/reference/trump.wav",
        "vinay": "examples/reference/Vinay.wav",
        "Wiz Khalifa": "examples/reference/Wiz Khalifa.wav",
        "woman1": "examples/reference/woman1.wav",
        "woman2": "examples/reference/woman2.wav",
        "woman3": "examples/reference/woman3.wav",
        "woman4": "examples/reference/woman4.wav",
        "woman5": "examples/reference/woman5.wav",
        "woman6": "examples/reference/woman6.wav",
        "woman7": "examples/reference/woman7.wav",
        "womanChinese1": "examples/reference/womanChinese1.wav",
        "womanChinese2": "examples/reference/womanChinese2.wav",
        "womanChinese3": "examples/reference/womanChinese3.wav",
        "womanChinese4": "examples/reference/womanChinese4.wav",
        "womanjp1": "examples/reference/womanjp1.wav",
        "womanjp2": "examples/reference/womanjp2.wav",
        "yinghao": "examples/reference/Yinghao.wav",
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global models
    logger.info("Loading Seed-VC model...")
    try:
        models = load_models()

        logger.info("Seed-VC model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    yield

    logger.info("Shutting down Seed-VC API")

app = FastAPI(title="Seed-VC API",
              lifespan=lifespan)


class VoiceConversionRequest(BaseModel):
    source_audio_key: str
    target_voice: str


class VoiceUploadResponse(BaseModel):
    message: str
    voice_key: str
    s3_key: str
    voices: Dict[str, str] = None

@app.post("/convert", dependencies=[Depends(verify_api_key)])
async def generate_speech(request: VoiceConversionRequest, background_tasks: BackgroundTasks):
    if not models:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if request.target_voice not in TARGET_VOICES:
        raise HTTPException(
            status_code=400, detail=f"Target voice not supported. Choose from: {', '.join(TARGET_VOICES.keys())}")

    try:
        target_audio_path = get_local_voice_path(request.target_voice)
        logger.info(
            f"Converting voice: {request.source_audio_key} to {request.target_voice}")
        logger.info(
            f"Using target voice from local path: {target_audio_path}")

        # Generate a unique filename
        audio_id = str(uuid.uuid4())
        output_filename = f"{audio_id}.wav"
        local_path = f"/tmp/{output_filename}"

        logger.info("Downloading source audio")
        source_temp = NamedTemporaryFile(delete=False, suffix=".wav")
        try:
            s3_client.download_fileobj(
                S3_BUCKET, Key=request.source_audio_key, Fileobj=source_temp)
            source_temp.close()
        except Exception as e:
            os.unlink(source_temp.name)
            raise HTTPException(
                status_code=404, detail="Source audio not found")

        vc_wave, sr = process_voice_conversion(
            models=models, source=source_temp.name, target_name=target_audio_path, output=None)

        os.unlink(source_temp.name)

        torchaudio.save(local_path, vc_wave, sr)

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
        logger.error(f"Error in voice conversion: {e}")
        raise HTTPException(
            status_code=500, detail="Error in voice conversion")


@app.post("/upload-voice", dependencies=[Depends(verify_api_key)], response_model=VoiceUploadResponse)
async def upload_voice(
    background_tasks: BackgroundTasks,
    voice_name: str = Form(None),
    file: UploadFile = File(...)
):
    """
    Upload a voice file to be used as a target voice for voice conversion.

    Args:
        voice_name: Optional custom name for the voice. If not provided, uses the filename.
        file: Audio file (WAV, MP3, FLAC, etc.) - recommended 3-10 seconds of clear speech

    Returns:
        Success message with the voice key that can be used in /convert endpoint
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

        # Convert to WAV format with standard settings
        output_path = temp_file_path.replace('.wav', '_processed.wav')
        try:
            # Load and resample to 16kHz (common for voice conversion)
            audio, sr = librosa.load(temp_file_path, sr=16000)
            sf.write(output_path, audio, 16000)
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

        # Add to TARGET_VOICES and refresh
        TARGET_VOICES[clean_voice_name] = f"s3://{S3_BUCKET}/{s3_key}"
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

    # Don't allow deletion of default voices (local paths)
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

        # Remove from TARGET_VOICES and refresh
        del TARGET_VOICES[voice_key]
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
    if models:
        return {"status": "healthy", "model": "loaded"}
    return {"status": "unhealthy", "model": "not loaded"}
