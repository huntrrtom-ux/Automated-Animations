import os
import json
import time
import uuid
import math
import subprocess
import logging
import random
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_file, send_from_directory
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
import openai
import base64
import requests as req
from PIL import Image
from io import BytesIO
from gevent.pool import Pool as GeventPool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
# Use /data for persistent storage on Railway (volume mount), fallback to local for dev
PERSISTENT_DIR = '/data' if os.path.isdir('/data') else os.path.dirname(os.path.abspath(__file__))

app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
app.config['PRESET_FOLDER'] = os.path.join(PERSISTENT_DIR, 'presets')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['PRESET_FOLDER'], exist_ok=True)

logger.info(f"Persistent storage: {PERSISTENT_DIR} ({'Railway volume' if PERSISTENT_DIR == '/data' else 'local'})")

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='gevent')

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'flac', 'm4a', 'aac', 'webm', 'mp4'}
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_image(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def emit_progress(session_id, step, progress, message, data=None):
    payload = {'session_id': session_id, 'step': step, 'progress': progress, 'message': message}
    if data:
        payload['data'] = data
    socketio.emit('progress', payload)
    logger.info(f"[{session_id}] {step}: {message} ({progress}%)")


# ===================== PRESET MANAGEMENT =====================
def get_preset_order():
    order_path = os.path.join(app.config['PRESET_FOLDER'], '_order.json')
    if os.path.exists(order_path):
        try:
            with open(order_path, 'r') as f:
                return json.load(f)
        except:
            pass
    return []

def save_preset_order(order):
    order_path = os.path.join(app.config['PRESET_FOLDER'], '_order.json')
    with open(order_path, 'w') as f:
        json.dump(order, f)

def get_all_presets():
    presets = []
    preset_dir = app.config['PRESET_FOLDER']
    preset_map = {}
    for name in os.listdir(preset_dir):
        if name.startswith('_'):
            continue
        config_path = os.path.join(preset_dir, name, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            config['id'] = name
            preset_map[name] = config
    # Sort by saved order, then append any new ones
    order = get_preset_order()
    for pid in order:
        if pid in preset_map:
            presets.append(preset_map.pop(pid))
    # Append remaining (new presets not yet in order)
    for pid in sorted(preset_map.keys()):
        presets.append(preset_map[pid])
    return presets

def get_preset(preset_id):
    preset_dir = os.path.join(app.config['PRESET_FOLDER'], preset_id)
    config_path = os.path.join(preset_dir, 'config.json')
    if not os.path.exists(config_path):
        return None
    with open(config_path, 'r') as f:
        config = json.load(f)
    style_path = os.path.join(preset_dir, 'style.png')
    if os.path.exists(style_path):
        with open(style_path, 'rb') as f:
            config['style_base64'] = base64.b64encode(f.read()).decode('utf-8')
    subject_path = os.path.join(preset_dir, 'subject.png')
    if os.path.exists(subject_path):
        with open(subject_path, 'rb') as f:
            config['subject_base64'] = base64.b64encode(f.read()).decode('utf-8')
    return config

def save_preset(name, style_data=None, subject_data=None, style_text='', tags=''):
    preset_id = str(uuid.uuid4())[:8]
    preset_dir = os.path.join(app.config['PRESET_FOLDER'], preset_id)
    os.makedirs(preset_dir, exist_ok=True)
    if style_data:
        style_bytes = base64.b64decode(style_data)
        with open(os.path.join(preset_dir, 'style.png'), 'wb') as f:
            f.write(style_bytes)
    tag_list = [t.strip() for t in tags.split(',') if t.strip()] if tags else []
    config = {'name': name, 'has_subject': subject_data is not None,
              'has_style_image': style_data is not None, 'style_text': style_text,
              'tags': tag_list,
              'created_at': time.strftime('%Y-%m-%d %H:%M')}
    if subject_data:
        subject_bytes = base64.b64decode(subject_data)
        with open(os.path.join(preset_dir, 'subject.png'), 'wb') as f:
            f.write(subject_bytes)
    with open(os.path.join(preset_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    return preset_id

def delete_preset(preset_id):
    import shutil
    preset_dir = os.path.join(app.config['PRESET_FOLDER'], preset_id)
    if os.path.exists(preset_dir):
        shutil.rmtree(preset_dir)
        return True
    return False


# ===================== AUDIO =====================
def get_audio_duration(filepath):
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
           '-of', 'default=noprint_wrappers=1:nokey=1', filepath]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return float(result.stdout.strip())

def split_audio_for_whisper(filepath, max_size_mb=24):
    file_size = os.path.getsize(filepath)
    max_bytes = max_size_mb * 1024 * 1024
    if file_size <= max_bytes:
        return [filepath]
    duration = get_audio_duration(filepath)
    num_chunks = math.ceil(file_size / max_bytes)
    chunk_duration = duration / num_chunks
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_duration
        chunk_path = os.path.join(os.path.dirname(filepath), f'chunk_{i}.mp3')
        cmd = ['ffmpeg', '-y', '-i', filepath, '-ss', str(start), '-t', str(chunk_duration),
               '-c:a', 'libmp3lame', '-b:a', '128k', chunk_path]
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        chunks.append(chunk_path)
    return chunks

def transcribe_audio(filepath, session_id):
    emit_progress(session_id, 'transcription', 2, 'Preparing audio...')
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    chunks = split_audio_for_whisper(filepath)
    all_segments = []
    full_text_parts = []
    audio_duration = get_audio_duration(filepath)
    
    for ci, chunk_path in enumerate(chunks):
        emit_progress(session_id, 'transcription', int(2 + 12 * ci / len(chunks)),
                     f'Transcribing part {ci+1}/{len(chunks)}...')
        # Calculate the true time offset based on chunk position, not Whisper's last segment
        if len(chunks) > 1:
            chunk_duration = audio_duration / len(chunks)
            time_offset = ci * chunk_duration
        else:
            time_offset = 0.0
        
        with open(chunk_path, 'rb') as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", file=audio_file,
                response_format="verbose_json", timestamp_granularities=["segment"]
            )
        chunk_seg_count = 0
        if hasattr(transcript, 'segments') and transcript.segments:
            for seg in transcript.segments:
                start = (seg.start if hasattr(seg, 'start') else seg['start']) + time_offset
                end = (seg.end if hasattr(seg, 'end') else seg['end']) + time_offset
                text = (seg.text if hasattr(seg, 'text') else seg['text']).strip()
                if text:  # skip empty segments
                    all_segments.append({'start': start, 'end': end, 'text': text})
                    chunk_seg_count += 1
        full_text_parts.append(transcript.text if hasattr(transcript, 'text') else str(transcript))
        logger.info(f"Whisper chunk {ci+1}/{len(chunks)}: {chunk_seg_count} segments, offset={time_offset:.1f}s")
        if chunk_path != filepath and os.path.exists(chunk_path):
            os.remove(chunk_path)
    
    # Log coverage
    if all_segments:
        last_seg_end = all_segments[-1]['end']
        logger.info(f"Whisper complete: {len(all_segments)} segments, last segment ends at {last_seg_end:.1f}s (audio={audio_duration:.1f}s)")
        if last_seg_end < audio_duration - 10:
            logger.warning(f"Whisper transcription ends {audio_duration - last_seg_end:.1f}s before audio end!")
    
    emit_progress(session_id, 'transcription', 15, f'Transcribed {len(all_segments)} segments')
    return {'full_text': ' '.join(full_text_parts), 'segments': all_segments}


# ===================== SCENE DETECTION =====================
def get_format_scene_rules(video_format, audio_duration):
    """Return format-specific scene rules for GPT prompt."""
    if video_format == 'pulse':
        return (
            "FORMAT: PULSE (fast-paced entertainment)\n\n"
            "INTRO (first ~30 seconds — flexible, end at the nearest natural scene break):\n"
            "- ALL intro scenes must have is_video: true\n"
            "- Prioritise SHORT SNAPPY scenes: 1-4 seconds each\n"
            "- Include ONE hero scene that is 5-8 seconds long\n"
            "- Each intro scene needs its own unique visual matching the narration\n\n"
            "BODY (after intro):\n"
            "- ALL body scenes must have is_video: false\n"
            "- Create a NEW scene every 2-6 seconds — keep it fast-paced\n"
            "- NEVER exceed 10 seconds for any single scene\n"
            "- EXCEPTION: Place ONE animated scene (is_video: true, 4-8 seconds) near every 10-minute mark "
            "(can be anywhere within ±30 seconds of each 10-min mark)\n\n"
            "COVERAGE:\n"
            "- Every second of the audio MUST be covered — NO gaps between scenes\n"
            "- Each scene's start_time MUST equal the previous scene's end_time\n"
            "- The FIRST scene MUST start at 0.0\n"
            "- The LAST scene's end_time MUST equal the final transcript timestamp\n"
        )
    elif video_format == 'flash':
        return (
            "FORMAT: FLASH (animated educational)\n\n"
            "INTRO (first ~2 minutes — flexible within ±10 seconds of the 2-minute mark):\n"
            "- ALL intro scenes must have is_video: true\n"
            "- Scene durations: 3-8 seconds, but PRIORITISE 3-5 second scenes for high tempo\n"
            "- Each intro scene needs its own unique visual matching the narration\n\n"
            "BODY (after intro):\n"
            "- ALL body scenes must have is_video: false\n"
            "- Create a NEW scene every 5-15 seconds based on what the narrator is saying\n"
            "- Short scenes (5-7s) when narrator makes quick points or transitions\n"
            "- Longer scenes (10-15s) when narrator explains one concept in depth\n"
            "- NEVER exceed 15 seconds for any single scene\n"
            "- NO animated scenes after intro\n\n"
            "COVERAGE:\n"
            "- Every second of the audio MUST be covered — NO gaps between scenes\n"
            "- Each scene's start_time MUST equal the previous scene's end_time\n"
            "- The FIRST scene MUST start at 0.0\n"
            "- The LAST scene's end_time MUST equal the final transcript timestamp\n"
        )
    elif video_format == 'deep':
        mid_point = audio_duration / 2 if audio_duration else 1800
        return (
            "FORMAT: DEEP (longform educational, ~60 minutes)\n\n"
            "INTRO (first ~45-60 seconds — analyse the transcript to find where the hook ends):\n"
            "- NO animated scenes — all is_video: false\n"
            "- Scene durations: 2-10 seconds\n"
            "- FRONT-LOAD with short snappy 2-4 second scenes for the first ~40 seconds\n"
            "- The last few intro scenes can be slightly longer (5-10s)\n\n"
            f"FIRST HALF (intro to ~{mid_point:.0f}s):\n"
            "- Create a NEW scene every 2-7 seconds, PRIORITISE shorter scenes to maintain momentum\n"
            "- All is_video: false\n\n"
            f"SECOND HALF (~{mid_point:.0f}s to end):\n"
            "- Create a NEW scene every 8-15 seconds, more relaxed pacing\n"
            "- NEVER exceed 15 seconds for any single scene\n"
            "- All is_video: false\n\n"
            "COVERAGE:\n"
            "- Every second of the audio MUST be covered — NO gaps between scenes\n"
            "- Each scene's start_time MUST equal the previous scene's end_time\n"
            "- The FIRST scene MUST start at 0.0\n"
            "- The LAST scene's end_time MUST equal the final transcript timestamp\n"
        )
    return ""

def get_format_subject_rules(video_format, has_subject):
    """Return format-specific subject/character rules."""
    if not has_subject:
        return ""
    
    if video_format == 'flash':
        return (
            "\n\nMAIN CHARACTER (SUBJECT REFERENCE):\n"
            "A character reference image has been uploaded.\n"
            "- Set has_subject: true for EVERY scene — the character MUST appear in ALL scenes\n"
            "- Describe the character's SPECIFIC emotion, body language, and action in every scene:\n"
            "  GOOD: 'looking frustrated while gripping a desk, furrowed brow, clenched jaw'\n"
            "  GOOD: 'pointing at a diagram with enthusiasm, wide smile'\n"
            "  BAD: 'the main character appears' (too vague)\n"
            "- The character should feel ALIVE — never stiff or static\n"
        )
    elif video_format == 'deep':
        return (
            "\n\nMAIN CHARACTER (SUBJECT REFERENCE):\n"
            "A character reference image has been uploaded.\n"
            "- Use the character SPARINGLY — roughly once every 5 minutes\n"
            "- Set has_subject: true only for scenes near 5-minute intervals (e.g. ~5min, ~10min, ~15min etc.)\n"
            "- When used, the character can be an educator pointing at something, reacting, or presenting\n"
            "- All other scenes: has_subject: false\n"
            "- When has_subject is true, describe emotion, body language, and action richly\n"
        )
    else:  # pulse — let GPT decide
        return (
            "\n\nMAIN CHARACTER (SUBJECT REFERENCE):\n"
            "A character reference image has been uploaded. You MUST follow these rules:\n"
            "- Set has_subject: true for ANY scene that contains a person, human figure, character, narrator, "
            "host, presenter, worker, businessman, or any living being\n"
            "- ONLY set has_subject: false for pure establishing shots (landscape, building exterior), "
            "object-only close-ups, or abstract visuals with absolutely NO people\n"
            "- When has_subject is true, describe the character's SPECIFIC emotion, body language, and action:\n"
            "  GOOD: 'looking frustrated while gripping a desk, furrowed brow, clenched jaw'\n"
            "  GOOD: 'leaning back in chair with a relieved smile, arms behind head'\n"
            "  GOOD: 'pointing urgently at a whiteboard, mouth open mid-speech'\n"
            "  BAD: 'the main character appears' (too vague)\n"
            "  BAD: 'a man stands in a room' (no emotion/action)\n"
            "- The character should feel ALIVE — never stiff or static\n"
            "- When in doubt, set has_subject: true\n"
        )

def detect_scene_changes(transcript_data, session_id, has_subject=False, video_format='pulse', audio_duration=0):
    emit_progress(session_id, 'scene_detection', 16, 'Analyzing script...')
    # Use Gemini via OpenAI-compatible endpoint (cheaper + higher rate limits)
    if GEMINI_API_KEY:
        client = openai.OpenAI(
            api_key=GEMINI_API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        scene_model = "gemini-2.5-flash"
        logger.info("Scene detection using Gemini 2.5 Flash")
    else:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        scene_model = "gpt-4o"
        logger.info("Scene detection using GPT-4o (no GEMINI_API_KEY set)")
    segments = transcript_data['segments']
    # FIX 1: Smaller chunks — 50 segments (~3-4 min) so Gemini can't skip sections
    CHUNK_SIZE = 50
    all_scenes = []

    format_rules = get_format_scene_rules(video_format, audio_duration)
    subject_note = get_format_subject_rules(video_format, has_subject)

    for chunk_start in range(0, len(segments), CHUNK_SIZE):
        chunk_segments = segments[chunk_start:chunk_start + CHUNK_SIZE]
        chunk_num = chunk_start // CHUNK_SIZE + 1
        total_chunks = math.ceil(len(segments) / CHUNK_SIZE)
        emit_progress(session_id, 'scene_detection', int(16 + 8 * chunk_start / len(segments)),
                     f'Analyzing section {chunk_num}/{total_chunks}...')
        segments_text = "\n".join([f"[{s['start']:.1f}s - {s['end']:.1f}s]: {s['text']}" for s in chunk_segments])

        # FIX 3: Calculate chunk time range so Gemini knows exactly what to cover
        chunk_time_start = chunk_segments[0]['start']
        chunk_time_end = chunk_segments[-1]['end']
        chunk_duration = chunk_time_end - chunk_time_start
        is_last_chunk = chunk_num == total_chunks

        system_prompt = (
            "You are a visual director creating scene breakdowns for an illustrated video.\n\n"
            "YOUR CORE TASK:\n"
            "Read through the transcript line by line. Every time the narrator moves to a new idea, "
            "concept, example, or subject — that's a new scene with its own unique visual. "
            "Each scene should last 5-15 seconds based on how long the narrator spends on that point.\n\n"
            "HOW TO CREATE SCENES:\n"
            "1. Read each line of the transcript\n"
            "2. When the narrator changes topic or moves to a new point, start a new scene\n"
            "3. Each scene's visual_description is a UNIQUE IMAGE PROMPT — it illustrates ONLY what the "
            "narrator says during those specific seconds, nothing else\n"
            "4. Scenes don't need to relate to each other — each one is a standalone visual for that moment\n"
            "5. Short points = short scenes (5-7s). Detailed explanations = longer scenes (10-15s)\n"
            f"6. You MUST create scenes covering from {chunk_time_start:.1f}s to {chunk_time_end:.1f}s with NO gaps\n\n"
            "SCENE VISUAL DESCRIPTION RULES:\n"
            "- Each visual_description is an IMAGE PROMPT — it must paint a specific, vivid picture\n"
            "- Every scene MUST look DIFFERENT — different setting, action, composition, mood, camera angle\n"
            "- Base each scene's visual ONLY on what the narrator says during those seconds\n"
            "- Use metaphorical visuals for abstract concepts: money = piles of coins/bills, debt = heavy chains, "
            "growth = rising stairs, risk = stormy skies, success = golden light\n"
            "- Describe: camera angle, lighting, mood, character emotions/poses, environment details, props\n"
            f"{subject_note}\n"
            "FORBIDDEN IN VISUAL DESCRIPTIONS:\n"
            "- NO text, words, numbers, labels, signs, letters, titles, captions, or writing anywhere\n"
            "- NO dollar amounts, percentages, statistics, charts, graphs, or data\n"
            "- NO art style words: 'cartoon', 'animated', 'illustrated', 'drawn', 'realistic', '3D'\n"
            "- The art style is controlled separately — only describe scene CONTENT\n\n"
            "TIMING RULES:\n"
            f"- Your scenes MUST start at {chunk_time_start:.1f}s and end at {chunk_time_end:.1f}s\n"
            "- Each scene's start_time MUST equal the previous scene's end_time (no gaps)\n"
            "- Vary durations: 5s, 7s, 10s, 12s — match the pacing of the speech\n"
            "- NEVER exceed 15 seconds for any single scene\n"
            "- NEVER create a scene shorter than 3 seconds\n\n"
            "Return valid JSON only, no markdown:\n"
            '{"scenes": [{"scene_number": 1, "start_time": 0.0, "end_time": 7.5, '
            '"narration_summary": "what narrator says", "visual_description": "unique image prompt for this scene only", '
            '"has_subject": true, "is_video": false}]}\n\n'
            f"{format_rules}"
        )

        user_msg = (
            f"Total audio duration: {audio_duration:.1f} seconds\n"
            f"This is section {chunk_num} of {total_chunks}, covering {chunk_time_start:.1f}s to {chunk_time_end:.1f}s.\n"
        )
        if is_last_chunk:
            user_msg += f"IMPORTANT: This is the LAST section — your final scene's end_time MUST reach {chunk_time_end:.1f}s.\n"
        user_msg += f"\nTranscript:\n\n{segments_text}"

        # FIX 2: Higher max_tokens (32000) + FIX 5: Lower temperature (0.4) for consistency
        response = client.chat.completions.create(
            model=scene_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.4, max_tokens=32000
        )
        response_text = response.choices[0].message.content.strip()
        if response_text.startswith('```'):
            response_text = response_text.split('\n', 1)[1]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
        try:
            chunk_scenes = json.loads(response_text)
            scenes_list = chunk_scenes.get('scenes', [])
            for scene in scenes_list:
                scene['scene_number'] = len(all_scenes) + 1
                all_scenes.append(scene)
            
            # FIX 4: Validation + retry — check if Gemini covered the full chunk
            if scenes_list:
                last_scene_end = scenes_list[-1].get('end_time', 0)
                uncovered = chunk_time_end - last_scene_end
                if uncovered > 10:
                    logger.warning(f"Chunk {chunk_num}: Gemini stopped at {last_scene_end:.1f}s, {uncovered:.1f}s uncovered. Retrying uncovered section.")
                    # Find segments in the uncovered range
                    retry_segments = [s for s in chunk_segments if s['start'] >= last_scene_end - 1]
                    if retry_segments:
                        retry_text = "\n".join([f"[{s['start']:.1f}s - {s['end']:.1f}s]: {s['text']}" for s in retry_segments])
                        retry_start = last_scene_end
                        retry_end = chunk_time_end
                        retry_msg = (
                            f"You need to create scenes covering {retry_start:.1f}s to {retry_end:.1f}s.\n"
                            f"The first scene MUST start at {retry_start:.1f}s. The last scene MUST end at {retry_end:.1f}s.\n\n"
                            f"Transcript:\n\n{retry_text}"
                        )
                        retry_response = client.chat.completions.create(
                            model=scene_model,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": retry_msg}
                            ],
                            temperature=0.4, max_tokens=32000
                        )
                        retry_text_resp = retry_response.choices[0].message.content.strip()
                        if retry_text_resp.startswith('```'):
                            retry_text_resp = retry_text_resp.split('\n', 1)[1]
                            if retry_text_resp.endswith('```'):
                                retry_text_resp = retry_text_resp[:-3]
                        try:
                            retry_scenes = json.loads(retry_text_resp)
                            for scene in retry_scenes.get('scenes', []):
                                scene['scene_number'] = len(all_scenes) + 1
                                all_scenes.append(scene)
                            logger.info(f"Retry added {len(retry_scenes.get('scenes', []))} scenes for uncovered section")
                        except json.JSONDecodeError as e:
                            logger.error(f"Retry JSON parse error in chunk {chunk_num}: {e}")
            
            logger.info(f"Chunk {chunk_num}/{total_chunks}: {len(scenes_list)} scenes, "
                        f"{chunk_time_start:.1f}s-{chunk_time_end:.1f}s")
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error in chunk {chunk_num}: {e}")
    
    # Log scene coverage info
    if all_scenes:
        last_scene_end = max(s.get('end_time', 0) for s in all_scenes)
        logger.info(f"Gemini scenes: {len(all_scenes)} scenes, covering 0-{last_scene_end:.1f}s (audio={audio_duration:.1f}s)")
        if last_scene_end < audio_duration - 5:
            logger.warning(f"Gemini left {audio_duration - last_scene_end:.1f}s uncovered at end of audio!")
    
    emit_progress(session_id, 'scene_detection', 25, f'Detected {len(all_scenes)} scenes')
    return all_scenes


# ===================== WHISK AUTH =====================
def get_whisk_token():
    return os.environ.get('WHISK_API_KEY') or os.environ.get('WHISK_API_TOKEN') or ''

def get_whisk_cookie():
    return os.environ.get('WHISK_COOKIE') or ''

def whisk_bearer_headers():
    return {
        "authorization": f"Bearer {get_whisk_token()}",
        "content-type": "application/json",
        "origin": "https://labs.google",
        "referer": "https://labs.google/",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
    }

def whisk_cookie_headers():
    return {
        "content-type": "application/json",
        "cookie": get_whisk_cookie(),
        "origin": "https://labs.google",
        "referer": "https://labs.google/fx/tools/whisk",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
    }


# ===================== WHISK CAPTION & UPLOAD =====================
def caption_image_whisk(image_base64, media_category, workflow_id, session_ts):
    headers = whisk_cookie_headers()
    if not image_base64.startswith('data:'):
        image_base64 = f"data:image/png;base64,{image_base64}"
    payload = {"json": {"captionInput": {"candidatesCount": 1, "mediaInput": {"mediaCategory": media_category}},
               "mediaInput": {"mediaCategory": media_category, "rawBytes": image_base64},
               "clientContext": {"sessionId": session_ts, "workflowId": workflow_id}}}
    try:
        response = req.post("https://labs.google/fx/api/trpc/backbone.captionImage", json=payload, headers=headers, timeout=60)
        if response.status_code == 200:
            result = response.json()
            try:
                candidates = result["result"]["data"]["json"]["result"]["candidates"]
                if candidates:
                    return candidates[0].get("output", "")
            except (KeyError, IndexError):
                pass
        else:
            logger.error(f"Caption failed: {response.status_code}")
    except Exception as e:
        logger.error(f"Caption error: {e}")
    return ""

def upload_image_to_whisk(image_base64, media_category, caption, workflow_id, session_ts):
    headers = whisk_cookie_headers()
    if not image_base64.startswith('data:'):
        image_base64 = f"data:image/png;base64,{image_base64}"
    payload = {"json": {"clientContext": {"workflowId": workflow_id, "sessionId": session_ts},
               "uploadMediaInput": {"mediaCategory": media_category, "caption": caption, "rawBytes": image_base64}}}
    try:
        response = req.post("https://labs.google/fx/api/trpc/backbone.uploadImage", json=payload, headers=headers, timeout=120)
        logger.info(f"Upload response ({media_category}): status={response.status_code}")
        if response.status_code == 401:
            return "TOKEN_EXPIRED"
        if response.status_code != 200:
            logger.error(f"Upload failed ({media_category}): {response.status_code}")
            return None
        result = response.json()
        try:
            gen_id = result["result"]["data"]["json"]["result"]["uploadMediaGenerationId"]
            logger.info(f"Got mediaGenerationId for {media_category}: {gen_id[:60]}...")
            return gen_id
        except (KeyError, TypeError) as e:
            logger.error(f"Could not extract mediaGenerationId: {e}")
            return None
    except Exception as e:
        logger.error(f"Upload exception: {e}")
        return None


def upload_preset_images_to_whisk(preset_config, session_id):
    workflow_id = str(uuid.uuid4())
    session_ts = f";{int(time.time() * 1000)}"
    result = {"workflow_id": workflow_id, "session_ts": session_ts}
    
    # Pass style_text through to generation
    style_text = preset_config.get('style_text', '')
    result['style_text'] = style_text

    if preset_config.get('style_base64'):
        emit_progress(session_id, 'generation', 26, 'Captioning style image...')
        auto_caption = caption_image_whisk(preset_config['style_base64'], "MEDIA_CATEGORY_SCENE", workflow_id, session_ts)
        style_caption = (
            "ABSOLUTE STYLE LOCK. Every generated image MUST exactly match this art style: "
            "the line work, outlines, color palette, shading technique, and rendering aesthetic. "
            "FORBIDDEN styles: photorealistic, realistic, 3D render, CGI, cinematic, lifelike, hyper-realistic, photo. "
            "If any element looks realistic, make it MORE stylized. "
            "DO NOT reproduce any objects, characters, or scenes from this image."
        )
        if style_text:
            style_caption += f" User style notes: {style_text}"
        if auto_caption:
            style_caption += f" Art style features: {auto_caption[:200]}"
        result['style_caption'] = style_caption

        emit_progress(session_id, 'generation', 28, 'Uploading style reference...')
        style_id = upload_image_to_whisk(preset_config['style_base64'], "MEDIA_CATEGORY_SCENE", style_caption, workflow_id, session_ts)
        if style_id == "TOKEN_EXPIRED":
            return "TOKEN_EXPIRED"
        result['style_media_id'] = style_id
    elif style_text:
        # Text-only style — no image upload needed, style applied via userInstruction
        emit_progress(session_id, 'generation', 28, f'Using text style: {style_text[:50]}...')

    if preset_config.get('subject_base64'):
        emit_progress(session_id, 'generation', 29, 'Captioning subject...')
        auto_caption = caption_image_whisk(preset_config['subject_base64'], "MEDIA_CATEGORY_SUBJECT", workflow_id, session_ts)
        subject_caption = (
            "CHARACTER IDENTITY REFERENCE. This character's face, body type, hair, and clothing should be used "
            "as identity reference ONLY. The character MUST be drawn with different poses, facial expressions, "
            "gestures, and emotions as needed by each scene. Do NOT keep the character stiff or in the same pose. "
            "Adapt the character dynamically — they can smile, frown, point, sit, run, gesture, etc."
        )
        if auto_caption:
            subject_caption += f" Character identity details: {auto_caption[:200]}"
        result['subject_caption'] = subject_caption

        emit_progress(session_id, 'generation', 29, 'Uploading subject character...')
        subject_id = upload_image_to_whisk(preset_config['subject_base64'], "MEDIA_CATEGORY_SUBJECT", subject_caption, workflow_id, session_ts)
        if subject_id == "TOKEN_EXPIRED":
            return "TOKEN_EXPIRED"
        result['subject_media_id'] = subject_id

    return result


# ===================== WHISK IMAGE GENERATION =====================
def generate_image_whisk(prompt, output_path, session_id, scene_num, whisk_session=None, scene_has_subject=False):
    # Route to recipe if we have uploaded images
    if whisk_session and (whisk_session.get('style_media_id') or whisk_session.get('subject_media_id')):
        for attempt in range(3):
            result = generate_image_with_recipe(prompt, output_path, session_id, scene_num, whisk_session, scene_has_subject)
            if result == "TOKEN_EXPIRED":
                return result
            if result is not None:
                return result
            if attempt < 2:
                logger.warning(f"Retry {attempt+2}/3 for scene {scene_num}")
                time.sleep(3)
        logger.error(f"All 3 attempts failed for scene {scene_num}, using placeholder")
        create_placeholder_image(prompt, output_path)
        return None

    # Text-only style or no style — use basic generateImage with style in prompt
    style_text = whisk_session.get('style_text', '') if whisk_session else ''
    if style_text:
        full_prompt = f"ART STYLE: {style_text}. Scene: {prompt}"
    else:
        full_prompt = prompt

    headers = whisk_bearer_headers()
    json_data = {
        "clientContext": {"workflowId": str(uuid.uuid4()), "tool": "BACKBONE", "sessionId": f";{int(time.time()*1000)}"},
        "imageModelSettings": {"imageModel": "IMAGEN_3_5", "aspectRatio": "IMAGE_ASPECT_RATIO_LANDSCAPE"},
        "mediaCategory": "MEDIA_CATEGORY_BOARD", "prompt": full_prompt, "seed": 0
    }
    response = req.post("https://aisandbox-pa.googleapis.com/v1/whisk:generateImage", json=json_data, headers=headers, timeout=120)
    if response.status_code == 401:
        return "TOKEN_EXPIRED"
    if response.status_code != 200:
        create_placeholder_image(prompt, output_path)
        return None
    result = response.json()
    if "imagePanels" in result and result["imagePanels"]:
        panel = result["imagePanels"][0]
        if "generatedImages" in panel and panel["generatedImages"]:
            img_data = panel["generatedImages"][0]
            encoded_image = img_data["encodedImage"]
            if "," in encoded_image:
                encoded_image = encoded_image.split(",", 1)[1]
            with open(output_path, 'wb') as f:
                f.write(base64.b64decode(encoded_image))
            enforce_resolution(output_path)
            return {"media_id": img_data.get("mediaGenerationId", ""), "prompt": img_data.get("prompt", prompt),
                    "encoded_image": img_data["encodedImage"], "workflow_id": img_data.get("workflowId", "")}
    create_placeholder_image(prompt, output_path)
    return None


def generate_image_with_recipe(prompt, output_path, session_id, scene_num, whisk_session, scene_has_subject=False):
    token = get_whisk_token()
    headers = {
        "authorization": f"Bearer {token}",
        "content-type": "text/plain;charset=UTF-8",
        "origin": "https://labs.google",
        "referer": "https://labs.google/",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
    }

    workflow_id = whisk_session.get('workflow_id', str(uuid.uuid4()))
    session_ts = whisk_session.get('session_ts', f";{int(time.time() * 1000)}")

    recipe_inputs = []
    if scene_has_subject and whisk_session.get('subject_media_id'):
        recipe_inputs.append({
            "caption": whisk_session.get('subject_caption', 'Character identity reference — adapt pose and expression to scene'),
            "mediaInput": {
                "mediaCategory": "MEDIA_CATEGORY_SUBJECT",
                "mediaGenerationId": whisk_session['subject_media_id']
            }
        })
    if whisk_session.get('style_media_id'):
        recipe_inputs.append({
            "caption": whisk_session.get('style_caption', 'Art style reference only — match visual style not content'),
            "mediaInput": {
                "mediaCategory": "MEDIA_CATEGORY_SCENE",
                "mediaGenerationId": whisk_session['style_media_id']
            }
        })

    # Build style instruction from text + image
    style_text = whisk_session.get('style_text', '')
    has_style_image = whisk_session.get('style_media_id') is not None
    
    if has_style_image and style_text:
        # Both image and text — strongest combination
        styled_prompt = (
            f"ABSOLUTE STYLE LOCK: You MUST match the exact art style from the style reference image. "
            f"Additional style details: {style_text}. "
            f"FORBIDDEN: photorealistic, realistic, 3D render, CGI, cinematic, lifelike, hyper-realistic, photo, photograph. "
            f"CRITICAL: Do NOT render ANY text, letters, numbers, words, labels, signs, or writing anywhere in the image. "
            f"If unsure, make it MORE stylized, not less. "
            f"The scene to create: {prompt}"
        )
    elif style_text:
        # Text only — no style image
        styled_prompt = (
            f"ABSOLUTE STYLE LOCK — ART STYLE: {style_text}. "
            f"Apply this style consistently to every element. "
            f"FORBIDDEN: photorealistic, realistic, 3D render, CGI, cinematic, lifelike, hyper-realistic, photo, photograph. "
            f"CRITICAL: Do NOT render ANY text, letters, numbers, words, labels, signs, or writing anywhere in the image. "
            f"If unsure, make it MORE stylized, not less. "
            f"The scene to create: {prompt}"
        )
    else:
        # Image only
        styled_prompt = (
            f"ABSOLUTE STYLE LOCK: Match the exact art style from the style reference — "
            f"same line work, coloring, shading, detail level. "
            f"FORBIDDEN: photorealistic, realistic, 3D render, CGI, cinematic, lifelike, hyper-realistic, photo, photograph. "
            f"CRITICAL: Do NOT render ANY text, letters, numbers, words, labels, signs, or writing anywhere in the image. "
            f"If unsure, make it MORE stylized, not less. "
            f"The scene to create: {prompt}"
        )

    json_data = {
        "clientContext": {"workflowId": workflow_id, "tool": "BACKBONE", "sessionId": session_ts},
        "imageModelSettings": {"imageModel": "R2I", "aspectRatio": "IMAGE_ASPECT_RATIO_LANDSCAPE"},
        "recipeMediaInputs": recipe_inputs,
        "seed": random.randint(100000, 999999),
        "userInstruction": styled_prompt
    }

    logger.info(f"Whisk runImageRecipe for scene {scene_num} (subject={scene_has_subject}, inputs={len(recipe_inputs)})")
    response = req.post("https://aisandbox-pa.googleapis.com/v1/whisk:runImageRecipe",
                        data=json.dumps(json_data), headers=headers, timeout=120)
    logger.info(f"Whisk recipe response for scene {scene_num}: {response.status_code}")

    # Token expired — wait and retry with fresh env vars (allows mid-generation token update)
    if response.status_code == 401:
        logger.warning(f"Token expired at scene {scene_num}. Waiting 60s for token refresh...")
        emit_progress(session_id, 'generation', -1, f'Token expired — update in Railway. Retrying in 60s...')
        time.sleep(60)
        # Re-read headers from env vars (picks up Railway variable changes)
        fresh_headers = whisk_cookie_headers()
        response = req.post("https://aisandbox-pa.googleapis.com/v1/whisk:runImageRecipe",
                            data=json.dumps(json_data), headers=fresh_headers, timeout=120)
        logger.info(f"Whisk retry after token refresh for scene {scene_num}: {response.status_code}")
        if response.status_code == 401:
            return "TOKEN_EXPIRED"
    if response.status_code != 200:
        logger.error(f"Whisk recipe error {response.status_code}: {response.text[:500]}")
        return None

    result = response.json()
    encoded_image = None
    media_id = ""
    img_prompt = prompt

    if "imagePanels" in result:
        for panel in result["imagePanels"]:
            for img in panel.get("generatedImages", []):
                encoded_image = img.get("encodedImage", "")
                media_id = img.get("mediaGenerationId", "")
                img_prompt = img.get("prompt", prompt)
                break
            if encoded_image:
                break
    if not encoded_image:
        encoded_image = result.get("encodedImage", "") or result.get("rawBytes", "")
        media_id = result.get("mediaGenerationId", "")

    if encoded_image:
        if "," in encoded_image:
            encoded_image = encoded_image.split(",", 1)[1]
        with open(output_path, 'wb') as f:
            f.write(base64.b64decode(encoded_image))
        enforce_resolution(output_path)
        logger.info(f"Whisk recipe generated image for scene {scene_num}")
        return {"media_id": media_id, "prompt": img_prompt, "encoded_image": encoded_image, "workflow_id": workflow_id}

    logger.warning(f"No image in recipe response for scene {scene_num}")
    return None


# ===================== ANIMATION =====================
def animate_image_whisk(image_info, script, output_path, session_id, scene_num):
    headers = whisk_bearer_headers()
    session_ts = f";{int(time.time() * 1000)}"
    raw_bytes = image_info.get("encoded_image", "")
    if raw_bytes and "," in raw_bytes[:100]:
        raw_bytes = raw_bytes.split(",", 1)[1]
    animate_data = {
        "clientContext": {"sessionId": session_ts, "tool": "BACKBONE", "workflowId": image_info.get("workflow_id", str(uuid.uuid4()))},
        "loopVideo": False, "modelKey": "", "modelNameType": "VEO_3_1_I2V_12STEP",
        "promptImageInput": {"mediaGenerationId": image_info.get("media_id", ""),
                             "prompt": f"ORIGINAL IMAGE DESCRIPTION:\n{image_info.get('prompt', script)}", "rawBytes": raw_bytes},
        "userInstructions": ""
    }
    logger.info(f"Whisk Animate starting for scene {scene_num} (media_id={image_info.get('media_id', 'none')}, has_raw_bytes={bool(raw_bytes)})")
    response = None
    for attempt in range(5):
        response = req.post("https://aisandbox-pa.googleapis.com/v1/whisk:generateVideo", json=animate_data, headers=headers, timeout=60)
        logger.info(f"Animate response scene {scene_num}: status={response.status_code} (attempt {attempt+1})")
        if response.status_code == 401:
            return "TOKEN_EXPIRED"
        if response.status_code == 429:
            time.sleep(30 * (attempt + 1))
            continue
        if response.status_code != 200:
            logger.warning(f"Animate scene {scene_num} HTTP error: {response.status_code} — body: {response.text[:300]}")
        break
    if response.status_code != 200:
        logger.error(f"Animate scene {scene_num} FAILED: HTTP {response.status_code} after all attempts — reason: {response.text[:500]}")
        return False
    result = response.json()
    operation_name = None
    if "operation" in result:
        op = result["operation"]
        if isinstance(op, dict):
            operation_name = op.get("operation", {}).get("name", "") or op.get("name", "")
    if not operation_name:
        operation_name = result.get("name", "")
    if not operation_name:
        logger.error(f"Animate scene {scene_num} FAILED: No operation_name in response — keys: {list(result.keys())}, response: {json.dumps(result)[:500]}")
        return False

    for i in range(90):
        time.sleep(2)
        emit_progress(session_id, 'generation', -1, f'Animating scene {scene_num}... ({(i+1)*2}s)')
        poll_resp = req.post("https://aisandbox-pa.googleapis.com/v1:runVideoFxSingleClipsStatusCheck",
                             json={"operations": [{"operation": {"name": operation_name}}]}, headers=headers, timeout=30)
        if poll_resp.status_code == 401:
            return "TOKEN_EXPIRED"
        if poll_resp.status_code != 200:
            logger.warning(f"Poll {i+1} scene {scene_num}: HTTP {poll_resp.status_code}")
            continue
        poll_result = poll_resp.json()
        status = poll_result.get("status", "")
        logger.info(f"Poll {i+1} scene {scene_num}: {status}")
        if status == "MEDIA_GENERATION_STATUS_SUCCESSFUL":
            raw_bytes = poll_result.get("rawBytes", "")
            if not raw_bytes:
                try:
                    ops = poll_result.get("operations", [])
                    if isinstance(ops, list) and ops:
                        raw_bytes = ops[0].get("rawBytes", "")
                except:
                    pass
            if raw_bytes:
                if "," in raw_bytes:
                    raw_bytes = raw_bytes.split(",", 1)[1]
                with open(output_path, 'wb') as f:
                    f.write(base64.b64decode(raw_bytes))
                return True
            logger.error(f"Animate scene {scene_num} FAILED: Status successful but NO raw_bytes — keys: {list(poll_result.keys())}")
            return False
        if status == "MEDIA_GENERATION_STATUS_FAILED":
            fail_reason = poll_result.get("failureReason", poll_result.get("error", "unknown"))
            logger.error(f"Animate scene {scene_num} FAILED: Veo returned FAILED — reason: {fail_reason}, full: {json.dumps(poll_result)[:500]}")
            return False
    logger.error(f"Animate scene {scene_num} FAILED: Polling timed out after 180s — last status: {status}")
    return False


# ===================== HELPERS =====================
def enforce_resolution(image_path, target_w=1920, target_h=1080):
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if img.size == (target_w, target_h):
            img.save(image_path, 'PNG')
            return
        src_w, src_h = img.size
        scale = max(target_w / src_w, target_h / src_h)
        new_w, new_h = int(src_w * scale), int(src_h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        img = img.crop((left, top, left + target_w, top + target_h))
        img.save(image_path, 'PNG')
    except Exception as e:
        logger.error(f"Resolution enforcement failed: {e}")

def create_placeholder_image(prompt, output_path):
    from PIL import ImageDraw, ImageFont
    img = Image.new('RGB', (1920, 1080), color=(255, 253, 245))
    draw = ImageDraw.Draw(img)
    cx, cy = 960, 400
    draw.ellipse([cx-40, cy-140, cx+40, cy-60], outline='#333333', width=3)
    draw.line([cx, cy-60, cx, cy+60], fill='#333333', width=3)
    draw.line([cx-60, cy-20, cx+60, cy-20], fill='#333333', width=3)
    draw.line([cx, cy+60, cx-40, cy+140], fill='#333333', width=3)
    draw.line([cx, cy+60, cx+40, cy+140], fill='#333333', width=3)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except:
        font = ImageFont.load_default()
    words = prompt[:200].split()
    lines, cur = [], ""
    for w in words:
        if len(cur + " " + w) < 60:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    y = 600
    for line in lines[:4]:
        bbox = draw.textbbox((0, 0), line, font=font)
        draw.text(((1920 - bbox[2] + bbox[0]) / 2, y), line, fill='#666666', font=font)
        y += 35
    img.save(output_path, 'PNG')


def create_video_from_image(image_path, video_path, duration, effect='none', scene_index=0):
    """Create a video from a still image with optional smooth Ken Burns effect.
    
    Uses scale + crop with frame counter (n) for smooth movement on looped images.
    Zoom rate is 0.5% per second, capped at 15%.
    """
    try:
        if effect == 'rotate_pulse':
            effects_cycle = ['pan_right', 'zoom_in', 'zoom_out']
            actual_effect = effects_cycle[scene_index % 3]
        elif effect == 'zoom_in':
            actual_effect = 'zoom_in'
        else:
            actual_effect = 'none'
        
        frames = max(int(duration * 25), 25)
        
        if actual_effect != 'none':
            # Pre-scale image to 110% with PIL so we have room to crop
            from PIL import Image as PILImage
            img = PILImage.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            scaled_w, scaled_h = 2112, 1188  # 1920*1.1, 1080*1.1
            img = img.resize((scaled_w, scaled_h), PILImage.LANCZOS)
            oversized_path = image_path.replace('.png', '_oversized.png')
            img.save(oversized_path, 'PNG')
            
            # Pixels to move across the full duration
            extra_w = scaled_w - 1920  # 192 pixels
            extra_h = scaled_h - 1080  # 108 pixels
            
            if actual_effect == 'zoom_in':
                # Start showing full oversized, end cropped to centre
                # crop width shrinks from scaled_w to 1920 over frames
                vf = (f"crop='trunc(({scaled_w}-({extra_w}*n/{frames}))/2)*2:"
                      f"trunc(({scaled_h}-({extra_h}*n/{frames}))/2)*2:"
                      f"trunc(({extra_w}*n/{frames})/4)*2:"
                      f"trunc(({extra_h}*n/{frames})/4)*2',"
                      f"scale=1920:1080")
            elif actual_effect == 'zoom_out':
                # Start cropped tight at centre, end showing full oversized
                vf = (f"crop='trunc((1920+({extra_w}*n/{frames}))/2)*2:"
                      f"trunc((1080+({extra_h}*n/{frames}))/2)*2:"
                      f"trunc(({extra_w}-{extra_w}*n/{frames})/4)*2:"
                      f"trunc(({extra_h}-{extra_h}*n/{frames})/4)*2',"
                      f"scale=1920:1080")
            elif actual_effect == 'pan_right':
                # Fixed crop size, x position moves left to right
                vf = (f"crop=1920:1080:"
                      f"trunc(({extra_w}*n/{frames})/2)*2:"
                      f"trunc({extra_h}/4)*2,"
                      f"scale=1920:1080")
            
            cmd = ['ffmpeg', '-y', '-loop', '1', '-i', oversized_path,
                   '-c:v', 'libx264', '-preset', 'ultrafast', '-b:v', '5M', '-t', str(duration),
                   '-pix_fmt', 'yuv420p',
                   '-vf', vf,
                   '-r', '25', video_path]
            subprocess.run(cmd, check=True, capture_output=True, timeout=300)
            # Clean up oversized image
            try:
                os.remove(oversized_path)
            except:
                pass
        else:
            cmd = ['ffmpeg', '-y', '-loop', '1', '-i', image_path,
                   '-c:v', 'libx264', '-preset', 'ultrafast', '-b:v', '5M', '-t', str(duration),
                   '-pix_fmt', 'yuv420p',
                   '-vf', 'scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080',
                   '-r', '25', video_path]
            subprocess.run(cmd, check=True, capture_output=True, timeout=180)
    except Exception as e:
        logger.error(f"Video from image failed: {e}")
        # Fallback to static if effects fail
        try:
            cmd = ['ffmpeg', '-y', '-loop', '1', '-i', image_path,
                   '-c:v', 'libx264', '-preset', 'ultrafast', '-b:v', '5M', '-t', str(duration),
                   '-pix_fmt', 'yuv420p',
                   '-vf', 'scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080',
                   '-r', '25', video_path]
            subprocess.run(cmd, check=True, capture_output=True, timeout=180)
        except Exception as e2:
            logger.error(f"Static fallback also failed: {e2}")


def compose_final_video(scene_videos, audio_path, output_path, session_id, audio_duration=None):
    emit_progress(session_id, 'compositing', 86, 'Compositing video...')
    if len(scene_videos) < 2:
        if scene_videos:
            cmd = ['ffmpeg', '-y', '-i', scene_videos[0], '-i', audio_path, '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
                   '-map', '0:v:0', '-map', '1:a:0', output_path]
            subprocess.run(cmd, check=True, capture_output=True, timeout=600)
        return output_path

    work_dir = os.path.dirname(output_path)
    total_clips = len(scene_videos)

    # Parallel normalize all clips to 1920x1080 @ 25fps using ultrafast
    NORM_BATCH_SIZE = 5  # 5 concurrent FFmpeg processes
    normalized_clips = [None] * total_clips
    norm_completed = [0]  # mutable counter for closure

    def normalize_single_clip(idx):
        clip = scene_videos[idx]
        norm_path = os.path.join(work_dir, f'norm_{idx:04d}.mp4')
        try:
            cmd = ['ffmpeg', '-y', '-i', clip,
                   '-vf', 'scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,format=yuv420p',
                   '-c:v', 'libx264', '-preset', 'ultrafast', '-b:v', '5M', '-r', '25', '-pix_fmt', 'yuv420p',
                   norm_path]
            subprocess.run(cmd, check=True, capture_output=True, timeout=120)
            return idx, norm_path
        except Exception as e:
            logger.warning(f"Normalize failed for clip {idx+1}/{total_clips}, using original: {e}")
            return idx, clip

    logger.info(f"=== NORMALIZE: Parallel normalization of {total_clips} clips (batches of {NORM_BATCH_SIZE}) ===")

    for batch_start in range(0, total_clips, NORM_BATCH_SIZE):
        batch_end = min(batch_start + NORM_BATCH_SIZE, total_clips)
        batch_indices = list(range(batch_start, batch_end))

        pool = GeventPool(size=NORM_BATCH_SIZE)
        batch_results = pool.map(normalize_single_clip, batch_indices)

        for idx, norm_path in batch_results:
            normalized_clips[idx] = norm_path
            norm_completed[0] += 1

        clip_progress = 88 + int(3 * norm_completed[0] / total_clips)
        emit_progress(session_id, 'compositing', clip_progress, f'Normalizing clips ({norm_completed[0]}/{total_clips})...')
        logger.info(f"Normalize batch: {norm_completed[0]}/{total_clips} complete")

    logger.info(f"=== NORMALIZE COMPLETE: {norm_completed[0]}/{total_clips} clips ===")

    # Concat all clips using stream copy (all clips are now identical format)
    emit_progress(session_id, 'compositing', 91, 'Joining scenes...')
    concat_file = os.path.join(work_dir, 'concat_list.txt')
    with open(concat_file, 'w') as f:
        for clip in normalized_clips:
            f.write(f"file '{clip}'\n")

    temp_video = os.path.join(work_dir, 'concat_video.mp4')
    cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_file,
           '-c', 'copy', temp_video]
    subprocess.run(cmd, check=True, capture_output=True, timeout=600)

    # Sync with audio duration
    emit_progress(session_id, 'compositing', 94, 'Syncing with audio...')

    # Add audio
    emit_progress(session_id, 'compositing', 97, 'Adding audio track...')
    if audio_duration:
        cmd = ['ffmpeg', '-y', '-i', temp_video, '-i', audio_path,
               '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
               '-map', '0:v:0', '-map', '1:a:0', '-t', str(audio_duration),
               output_path]
    else:
        cmd = ['ffmpeg', '-y', '-i', temp_video, '-i', audio_path, '-c:v', 'copy', '-c:a', 'aac', '-b:v', '192k',
               '-map', '0:v:0', '-map', '1:a:0', output_path]
    subprocess.run(cmd, check=True, capture_output=True, timeout=1800)
    emit_progress(session_id, 'compositing', 100, 'Final video complete!')
    return output_path


# ===================== MAIN PIPELINE =====================
def process_voiceover(filepath, session_id, preset_id=None, video_format='pulse', project_title=''):
    try:
        start_time_ts = time.time()
        work_dir = os.path.join(app.config['OUTPUT_FOLDER'], session_id)
        os.makedirs(work_dir, exist_ok=True)

        preset_config = None
        has_subject = False
        if preset_id:
            preset_config = get_preset(preset_id)
            if preset_config:
                has_subject = preset_config.get('has_subject', False)

        audio_duration = get_audio_duration(filepath)
        emit_progress(session_id, 'init', 1, f'Audio: {audio_duration/60:.1f} min')

        transcript_data = transcribe_audio(filepath, session_id)
        scenes = detect_scene_changes(transcript_data, session_id, has_subject, video_format, audio_duration)

        with open(os.path.join(work_dir, 'scenes.json'), 'w') as f:
            json.dump({'transcript': transcript_data, 'scenes': scenes, 'audio_duration': audio_duration}, f, indent=2)

        # Merge short scenes
        MIN_DUR = 5.0
        merged = []
        for scene in scenes:
            if merged and (merged[-1]['end_time'] - merged[-1]['start_time']) < MIN_DUR:
                merged[-1]['end_time'] = scene['end_time']
                merged[-1]['visual_description'] += " " + scene['visual_description']
                merged[-1]['has_subject'] = merged[-1].get('has_subject', False) or scene.get('has_subject', False)
                continue
            merged.append(scene)
        if merged and (merged[-1]['end_time'] - merged[-1]['start_time']) < MIN_DUR and len(merged) >= 2:
            merged[-2]['end_time'] = merged[-1]['end_time']
            merged[-2]['has_subject'] = merged[-2].get('has_subject', False) or merged[-1].get('has_subject', False)
            merged.pop()
        logger.info(f"Merged {len(scenes)} -> {len(merged)} scenes")
        scenes = merged

        # FIX #1: Renumber sequentially
        for i, scene in enumerate(scenes):
            scene['scene_number'] = i + 1
        total = len(scenes)

        # FIX #7: Ensure scenes cover full audio with no gaps
        if scenes:
            # Handle end-of-audio gap — DON'T just extend last scene
            if scenes[-1]['end_time'] < audio_duration:
                gap = audio_duration - scenes[-1]['end_time']
                if gap <= 5.0:
                    # Small gap — safe to extend last scene
                    logger.info(f"Extending last scene by {gap:.1f}s to match audio duration")
                    scenes[-1]['end_time'] = audio_duration
                else:
                    # Large gap — create new scenes from transcript segments
                    logger.warning(f"Large gap at end: {scenes[-1]['end_time']:.1f}s to {audio_duration:.1f}s ({gap:.1f}s uncovered)")
                    gap_start = scenes[-1]['end_time']
                    segments = transcript_data['segments']
                    # Find transcript segments in the uncovered range
                    gap_segments = [s for s in segments if s['end'] > gap_start and s['start'] < audio_duration]
                    
                    if gap_segments:
                        # Group gap segments into ~10s scenes
                        current_scene_start = gap_start
                        current_texts = []
                        for seg in gap_segments:
                            current_texts.append(seg['text'].strip())
                            seg_time = seg['end'] - current_scene_start
                            # Create a new scene every ~10s or at the last segment
                            if seg_time >= 10.0 or seg == gap_segments[-1]:
                                narration = ' '.join(current_texts)
                                new_scene = {
                                    'scene_number': len(scenes) + 1,
                                    'start_time': current_scene_start,
                                    'end_time': min(seg['end'], audio_duration),
                                    'narration_summary': narration[:100],
                                    'visual_description': f"The main character in a new setting that visually represents: {narration[:200]}",
                                    'has_subject': scenes[-1].get('has_subject', True),
                                    'is_video': False
                                }
                                scenes.append(new_scene)
                                logger.info(f"Created gap scene {new_scene['scene_number']}: {new_scene['start_time']:.1f}-{new_scene['end_time']:.1f}s")
                                current_scene_start = seg['end']
                                current_texts = []
                    
                    # Final extension if still not reaching audio end
                    if scenes[-1]['end_time'] < audio_duration:
                        remaining = audio_duration - scenes[-1]['end_time']
                        if remaining > 15.0:
                            # Still a big gap with no transcript — create filler scenes
                            num_fillers = math.ceil(remaining / 10.0)
                            filler_dur = remaining / num_fillers
                            for f in range(num_fillers):
                                filler = {
                                    'scene_number': len(scenes) + 1,
                                    'start_time': scenes[-1]['end_time'],
                                    'end_time': scenes[-1]['end_time'] + filler_dur,
                                    'narration_summary': 'Continuation',
                                    'visual_description': f"A wide establishing shot of a peaceful, contemplative landscape — soft golden light, open sky, sense of resolution and calm",
                                    'has_subject': False,
                                    'is_video': False
                                }
                                scenes.append(filler)
                                logger.info(f"Created filler scene {filler['scene_number']}: {filler['start_time']:.1f}-{filler['end_time']:.1f}s")
                        else:
                            scenes[-1]['end_time'] = audio_duration
            
            # Fill any gaps between scenes
            filled_scenes = [scenes[0]]
            for i in range(1, len(scenes)):
                prev_end = filled_scenes[-1]['end_time']
                curr_start = scenes[i]['start_time']
                if curr_start > prev_end + 0.5:
                    gap = curr_start - prev_end
                    if gap <= 10.0:
                        # Small gap — extend previous scene
                        logger.info(f"Filling gap: {prev_end:.1f}s to {curr_start:.1f}s by extending scene {filled_scenes[-1]['scene_number']}")
                        filled_scenes[-1]['end_time'] = curr_start
                    else:
                        # Large gap — create a new scene from transcript
                        gap_segs = [s for s in transcript_data['segments'] if s['end'] > prev_end and s['start'] < curr_start]
                        gap_text = ' '.join(s['text'].strip() for s in gap_segs) if gap_segs else 'A moment of reflection and transition'
                        gap_scene = {
                            'scene_number': 0,  # will be renumbered
                            'start_time': prev_end,
                            'end_time': curr_start,
                            'narration_summary': gap_text[:100],
                            'visual_description': f"The main character in a new environment that represents: {gap_text[:200]}",
                            'has_subject': True,
                            'is_video': False
                        }
                        filled_scenes[-1]['end_time'] = prev_end  # don't extend
                        filled_scenes.append(gap_scene)
                        logger.info(f"Created gap-fill scene: {prev_end:.1f}s to {curr_start:.1f}s")
                filled_scenes.append(scenes[i])
            
            # Ensure first scene starts at 0
            if filled_scenes[0]['start_time'] > 0.5:
                filled_scenes[0]['start_time'] = 0.0
            
            scenes = filled_scenes
            total = len(scenes)

        # Validate scene coverage - log any gaps or oversized scenes
        for i, scene in enumerate(scenes):
            dur = scene['end_time'] - scene['start_time']
            if dur > 20.0:
                logger.warning(f"Scene {scene.get('scene_number', '?')} is {dur:.1f}s — Gemini created an oversized scene")
            if i > 0:
                gap = scene['start_time'] - scenes[i-1]['end_time']
                if gap > 1.0:
                    logger.warning(f"Gap of {gap:.1f}s between scenes {i} and {i+1}")

        # Re-renumber after split
        for i, scene in enumerate(scenes):
            scene['scene_number'] = i + 1
        total = len(scenes)

        # Animation flags per format — enforce from pipeline side
        logger.info(f"Animation selection for format={video_format}, {len(scenes)} scenes:")
        for i, scene in enumerate(scenes[:20]):  # Log first 20 scenes' timing
            logger.info(f"  Scene {i}: start={scene['start_time']:.1f}s, end={scene['end_time']:.1f}s, dur={scene['end_time']-scene['start_time']:.1f}s")
        if video_format == 'pulse':
            # Intro: all scenes before ~30s animated, find natural break
            intro_end = 30.0
            last_intro_idx = -1
            for i, scene in enumerate(scenes):
                if scene['start_time'] < intro_end:
                    last_intro_idx = i
            for i, scene in enumerate(scenes):
                if i <= last_intro_idx:
                    scene['is_video'] = True
                else:
                    # Periodic animation every ~10 minutes
                    mid = (scene['start_time'] + scene['end_time']) / 2
                    near_10min = any(abs(mid - mark) < 30 for mark in range(600, int(audio_duration), 600))
                    scene['is_video'] = near_10min and scene.get('is_video', False)
        elif video_format == 'flash':
            # Flash: animate ALL scenes in the first ~2 minutes
            # Simple rule: any scene that STARTS before 120s gets animated
            intro_cutoff = 120.0
            animated_count = 0
            for i, scene in enumerate(scenes):
                if scene['start_time'] < intro_cutoff:
                    scene['is_video'] = True
                    animated_count += 1
                else:
                    scene['is_video'] = False
            logger.info(f"Flash intro: {animated_count} scenes marked for animation (all starting before {intro_cutoff}s)")
            if animated_count > 0:
                logger.info(f"Flash intro: first animated scene ends at {scenes[0]['end_time']:.1f}s, last animated scene ends at {scenes[animated_count-1]['end_time']:.1f}s")
            # Force subject on every scene for flash
            if has_subject:
                for scene in scenes:
                    scene['has_subject'] = True
        elif video_format == 'deep':
            # No animation at all
            for scene in scenes:
                scene['is_video'] = False

        # Upload preset images
        whisk_session = None
        if preset_config:
            emit_progress(session_id, 'generation', 28, 'Uploading style/subject to Whisk...')
            whisk_session = upload_preset_images_to_whisk(preset_config, session_id)
            if whisk_session == "TOKEN_EXPIRED":
                emit_progress(session_id, 'error', 0, 'Token expired — update in Railway settings.')
                return None

        # Generate visuals
        scene_videos = []
        # Determine Ken Burns effect per format
        def get_scene_effect(fmt, scene_is_video):
            if scene_is_video:
                return 'none'  # Animated scenes don't need Ken Burns
            if fmt == 'pulse':
                return 'rotate_pulse'
            else:  # flash and deep — no effects
                return 'none'
        
        for i, scene in enumerate(scenes):
            scene_num = scene['scene_number']
            start, end = scene['start_time'], scene['end_time']
            duration = end - start
            is_video = scene.get('is_video', False)
            scene_has_subject = scene.get('has_subject', False) and has_subject
            scene_effect = get_scene_effect(video_format, is_video)

            logger.info(f"Scene {scene_num}/{total}: {start:.1f}-{end:.1f}s, video={is_video}, subject={scene_has_subject}, effect={scene_effect}")

        # ===================== PHASE 1: BATCH IMAGE GENERATION =====================
        IMAGE_BATCH_SIZE = 10
        IMAGE_BATCH_PAUSE_EVERY = 20
        image_results = [None] * len(scenes)
        images_generated = 0

        def generate_single_image(idx):
            scene = scenes[idx]
            scene_num = scene['scene_number']
            scene_has_subject = scene.get('has_subject', False) and has_subject
            img_path = os.path.join(work_dir, f'scene_{scene_num:04d}.png')
            result = generate_image_whisk(scene['visual_description'], img_path, session_id, scene_num, whisk_session, scene_has_subject)
            add_credits(1, f'Image generation — scene {scene_num}', session_id)
            return idx, result

        logger.info(f"=== PHASE 1: Batch image generation for {total} scenes (batches of {IMAGE_BATCH_SIZE}) ===")
        emit_progress(session_id, 'generation', 30, f'Generating images (0/{total})...')

        for batch_start in range(0, len(scenes), IMAGE_BATCH_SIZE):
            batch_end = min(batch_start + IMAGE_BATCH_SIZE, len(scenes))
            batch_indices = list(range(batch_start, batch_end))

            pool = GeventPool(size=IMAGE_BATCH_SIZE)
            batch_results = pool.map(generate_single_image, batch_indices)

            token_expired = False
            for idx, result in batch_results:
                image_results[idx] = result
                images_generated += 1
                if result == "TOKEN_EXPIRED":
                    token_expired = True

            if token_expired:
                emit_progress(session_id, 'error', 0, 'Token expired — update in Railway settings.')
                return None

            progress = 30 + (30 * images_generated / total)
            emit_progress(session_id, 'generation', int(progress), f'Generating images ({images_generated}/{total})...')
            logger.info(f"Image batch {batch_start//IMAGE_BATCH_SIZE + 1}: {images_generated}/{total} complete")

            # Pause every N images to avoid throttling
            if images_generated % IMAGE_BATCH_PAUSE_EVERY == 0 and images_generated < len(scenes):
                logger.info(f"Throttle pause after {images_generated} images (2s)")
                time.sleep(2)

        logger.info(f"=== PHASE 1 COMPLETE: {images_generated}/{total} images generated ===")

        # ===================== PHASE 2: BATCH ANIMATION (Veo) =====================
        ANIM_BATCH_SIZE = 3
        ANIM_BATCH_DELAY = 2

        # Collect scenes that need animation
        anim_scenes = []
        for i, scene in enumerate(scenes):
            if scene.get('is_video', False) and image_results[i] and isinstance(image_results[i], dict):
                anim_scenes.append(i)

        total_anims = len(anim_scenes)
        anims_completed = 0
        anim_results = {}  # idx -> (animated: bool, video_path: str or None)

        def animate_single_scene(idx):
            scene = scenes[idx]
            scene_num = scene['scene_number']
            video_path = os.path.join(work_dir, f'scene_{scene_num:04d}_animated.mp4')
            image_info = image_results[idx]
            max_anim_retries = 5
            animated = False

            for anim_attempt in range(max_anim_retries):
                try:
                    animated = animate_image_whisk(image_info, scene['visual_description'], video_path, session_id, scene_num)
                except Exception as e:
                    logger.error(f"Animation error scene {scene_num} (attempt {anim_attempt+1}/{max_anim_retries}): {e}")
                    animated = False
                if animated == "TOKEN_EXPIRED":
                    return idx, "TOKEN_EXPIRED", None
                if animated:
                    break
                logger.warning(f"Animation attempt {anim_attempt+1}/{max_anim_retries} FAILED for scene {scene_num}")
                if anim_attempt < max_anim_retries - 1:
                    time.sleep(3)
                # On second-to-last retry, regenerate the image
                if anim_attempt == max_anim_retries - 2:
                    logger.info(f"Regenerating image for scene {scene_num} before final animation attempt")
                    scene_has_subject = scene.get('has_subject', False) and has_subject
                    img_path = os.path.join(work_dir, f'scene_{scene_num:04d}.png')
                    new_image = generate_image_whisk(scene['visual_description'], img_path, session_id, scene_num, whisk_session, scene_has_subject)
                    if new_image == "TOKEN_EXPIRED":
                        return idx, "TOKEN_EXPIRED", None
                    if new_image and isinstance(new_image, dict):
                        image_results[idx] = new_image

            return idx, animated, video_path if animated else None

        if total_anims > 0:
            logger.info(f"=== PHASE 2: Batch animation for {total_anims} scenes (batches of {ANIM_BATCH_SIZE}) ===")
            emit_progress(session_id, 'generation', 60, f'Animating scenes (0/{total_anims})...')

            for batch_start in range(0, total_anims, ANIM_BATCH_SIZE):
                batch_end = min(batch_start + ANIM_BATCH_SIZE, total_anims)
                batch_indices = [anim_scenes[j] for j in range(batch_start, batch_end)]

                pool = GeventPool(size=ANIM_BATCH_SIZE)
                batch_results = pool.map(animate_single_scene, batch_indices)

                token_expired = False
                for idx, animated, video_path in batch_results:
                    anim_results[idx] = (animated, video_path)
                    anims_completed += 1
                    if animated == "TOKEN_EXPIRED":
                        token_expired = True

                if token_expired:
                    emit_progress(session_id, 'error', 0, 'Token expired — update in Railway settings.')
                    return None

                progress = 60 + (25 * anims_completed / total_anims)
                emit_progress(session_id, 'generation', int(progress), f'Animating scenes ({anims_completed}/{total_anims})...')
                logger.info(f"Animation batch {batch_start//ANIM_BATCH_SIZE + 1}: {anims_completed}/{total_anims} complete")

                if batch_end < total_anims:
                    time.sleep(ANIM_BATCH_DELAY)

            logger.info(f"=== PHASE 2 COMPLETE: {anims_completed}/{total_anims} animations ===")
        else:
            logger.info("=== PHASE 2 SKIPPED: No scenes require animation ===")

        # ===================== PHASE 3: ASSEMBLE CLIPS (PARALLEL) =====================
        CLIP_BATCH_SIZE = 5
        scene_videos = [None] * len(scenes)
        clips_completed = [0]

        def assemble_single_clip(i):
            scene = scenes[i]
            scene_num = scene['scene_number']
            start, end = scene['start_time'], scene['end_time']
            duration = end - start
            is_video = scene.get('is_video', False)
            scene_effect = get_scene_effect(video_format, is_video)
            img_path = os.path.join(work_dir, f'scene_{scene_num:04d}.png')

            if i in anim_results:
                animated, video_path = anim_results[i]
                if animated and video_path:
                    add_credits(2, f'Animation (Veo) — scene {scene_num}', session_id)
                    trimmed = os.path.join(work_dir, f'scene_{scene_num:04d}_trimmed.mp4')
                    try:
                        cmd = ['ffmpeg', '-y', '-i', video_path, '-t', str(duration),
                               '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', '-r', '25',
                               '-vf', 'scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080',
                               trimmed]
                        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
                        return i, trimmed
                    except:
                        return i, video_path
                else:
                    logger.error(f"=== ANIMATION DIAGNOSTIC DUMP for scene {scene_num} ===")
                    logger.error(f"  Scene: {scene_num}/{total}, start={start:.1f}s, end={end:.1f}s, duration={duration:.1f}s")
                    logger.error(f"  is_video={is_video}, format={video_format}")
                    logger.error(f"  image_info type={type(image_results[i]).__name__}, media_id={image_results[i].get('media_id', 'none') if isinstance(image_results[i], dict) else 'N/A'}")
                    logger.error(f"  All animation attempts FAILED — falling back to still image")
                    logger.error(f"=== END DIAGNOSTIC ===")
                    vid = os.path.join(work_dir, f'scene_{scene_num:04d}_video.mp4')
                    create_video_from_image(img_path, vid, duration, effect=scene_effect, scene_index=i)
                    return i, vid
            elif is_video and (not image_results[i] or not isinstance(image_results[i], dict)):
                logger.warning(f"Scene {scene_num} marked is_video=True but image_info invalid — falling back to still")
                vid = os.path.join(work_dir, f'scene_{scene_num:04d}_video.mp4')
                create_video_from_image(img_path, vid, duration, effect=scene_effect, scene_index=i)
                return i, vid
            else:
                vid = os.path.join(work_dir, f'scene_{scene_num:04d}_video.mp4')
                create_video_from_image(img_path, vid, duration, effect=scene_effect, scene_index=i)
                return i, vid

        logger.info(f"=== PHASE 3: Parallel clip assembly for {total} scenes (batches of {CLIP_BATCH_SIZE}) ===")
        emit_progress(session_id, 'generation', 85, f'Assembling clips (0/{total})...')

        for batch_start in range(0, len(scenes), CLIP_BATCH_SIZE):
            batch_end = min(batch_start + CLIP_BATCH_SIZE, len(scenes))
            batch_indices = list(range(batch_start, batch_end))

            pool = GeventPool(size=CLIP_BATCH_SIZE)
            batch_results = pool.map(assemble_single_clip, batch_indices)

            for idx, vid_path in batch_results:
                scene_videos[idx] = vid_path
                clips_completed[0] += 1

            progress = 85 + (3 * clips_completed[0] / total)
            emit_progress(session_id, 'generation', int(progress), f'Assembling clips ({clips_completed[0]}/{total})...')
            logger.info(f"Assembly batch: {clips_completed[0]}/{total} complete")

        logger.info(f"=== PHASE 3 COMPLETE: {clips_completed[0]}/{total} clips assembled ===")

        # Compose
        if project_title:
            output_filename = f'{project_title}.mp4'
        else:
            output_filename = f'visualized_{session_id}.mp4'
        output_path = os.path.join(work_dir, output_filename)
        compose_final_video(scene_videos, filepath, output_path, session_id, audio_duration=audio_duration)

        # Save generation state for scene regeneration
        generation_state = {
            'session_id': session_id,
            'scenes': scenes,
            'scene_videos': scene_videos,
            'audio_path': filepath,
            'audio_duration': audio_duration,
            'output_filename': output_filename,
            'preset_id': preset_id,
            'video_format': video_format,
            'has_subject': has_subject,
            'project_title': project_title
        }
        with open(os.path.join(work_dir, 'generation_state.json'), 'w') as f:
            json.dump(generation_state, f, indent=2)

        emit_progress(session_id, 'complete', 100, 'Processing complete!', {
            'video_url': f'/download/{session_id}/{output_filename}',
            'scenes': scenes,
            'session_id': session_id
        })
        
        # Log generation to history
        processing_time = round(time.time() - start_time_ts, 1)
        credits_data = load_credits()
        # Count credits for this session
        session_credits = sum(e['amount'] for e in credits_data.get('log', []) if e.get('session_id') == session_id)
        log_generation({
            'session_id': session_id,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'filename': os.path.basename(filepath),
            'audio_duration': round(audio_duration, 1),
            'scene_count': total,
            'preset_id': preset_id or 'none',
            'preset_name': preset_config.get('name', 'Unknown') if preset_config else 'None',
            'animate_intro': video_format,
            'credits_used': session_credits,
            'processing_time': processing_time,
            'status': 'complete',
            'video_url': f'/download/{session_id}/{output_filename}'
        })
        
        return output_path
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        emit_progress(session_id, 'error', 0, f'Error: {str(e)}')
        
        # Log failed generation
        log_generation({
            'session_id': session_id,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'filename': os.path.basename(filepath),
            'preset_id': preset_id or 'none',
            'status': 'error',
            'error': str(e)
        })
        raise


# ===================== GENERATION HISTORY =====================
HISTORY_FILE = os.path.join(PERSISTENT_DIR, 'generation_history.json')

def log_generation(entry):
    history = load_history()
    history.insert(0, entry)  # newest first
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to log generation: {e}")

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []


ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'hunter2026')


@app.route('/admin')
def admin_page():
    auth = request.cookies.get('admin_auth')
    if auth != ADMIN_PASSWORD:
        return render_template('admin_login.html')
    history = load_history()
    return render_template('admin.html', history=history)


@app.route('/admin/login', methods=['POST'])
def admin_login():
    password = request.form.get('password', '')
    if password == ADMIN_PASSWORD:
        resp = jsonify({'ok': True})
        resp.set_cookie('admin_auth', password, max_age=60*60*24*90, httponly=True, samesite='Lax')
        return resp
    return jsonify({'error': 'Wrong password'}), 401


@app.route('/admin/history')
def admin_history_api():
    auth = request.cookies.get('admin_auth')
    if auth != ADMIN_PASSWORD:
        return jsonify({'error': 'Unauthorized'}), 401
    return jsonify(load_history())


# ===================== CREDITS TRACKING =====================
CREDITS_FILE = os.path.join(PERSISTENT_DIR, 'credits.json')

def load_credits():
    if os.path.exists(CREDITS_FILE):
        try:
            with open(CREDITS_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {'used': 0, 'log': []}

def save_credits(data):
    try:
        with open(CREDITS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save credits: {e}")

def add_credits(amount, description, session_id=''):
    data = load_credits()
    data['used'] += amount
    data['log'].append({
        'amount': amount,
        'description': description,
        'session_id': session_id,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    })
    save_credits(data)

@app.route('/admin/credits')
def admin_credits_api():
    auth = request.cookies.get('admin_auth')
    if auth != ADMIN_PASSWORD:
        return jsonify({'error': 'Unauthorized'}), 401
    return jsonify(load_credits())

@app.route('/admin/credits/reset', methods=['POST'])
def admin_credits_reset():
    auth = request.cookies.get('admin_auth')
    if auth != ADMIN_PASSWORD:
        return jsonify({'error': 'Unauthorized'}), 401
    save_credits({'used': 0, 'log': []})
    return jsonify({'ok': True, 'message': 'Credits reset to 0'})

@app.route('/admin/credits/set', methods=['POST'])
def admin_credits_set():
    auth = request.cookies.get('admin_auth')
    if auth != ADMIN_PASSWORD:
        return jsonify({'error': 'Unauthorized'}), 401
    amount = request.json.get('amount', 0)
    try:
        amount = int(amount)
    except:
        return jsonify({'error': 'Invalid amount'}), 400
    data = load_credits()
    data['used'] = amount
    data['log'].append({
        'amount': 0,
        'description': f'Manual set to {amount}',
        'session_id': 'admin',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    })
    save_credits(data)
    return jsonify({'ok': True, 'used': amount})


# ===================== ROUTES =====================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400
    file = request.files['audio']
    if not file.filename or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    preset_id = request.form.get('preset_id', '')
    video_format = request.form.get('format', 'pulse')
    project_title = request.form.get('project_title', '').strip()
    session_id = str(uuid.uuid4())[:12]
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'{session_id}_{filename}')
    file.save(filepath)
    socketio.start_background_task(process_voiceover, filepath, session_id, preset_id, video_format, project_title)
    return jsonify({'session_id': session_id, 'message': 'Processing started', 'filename': filename})

@app.route('/download/<session_id>/<filename>')
def download_file(session_id, filename):
    return send_from_directory(os.path.join(app.config['OUTPUT_FOLDER'], session_id), filename, as_attachment=True)

@app.route('/scene-image/<session_id>/<int:scene_num>')
def scene_image(session_id, scene_num):
    """Serve the generated image for a specific scene."""
    work_dir = os.path.join(app.config['OUTPUT_FOLDER'], session_id)
    img_path = os.path.join(work_dir, f'scene_{scene_num:04d}.png')
    if os.path.exists(img_path):
        return send_from_directory(work_dir, f'scene_{scene_num:04d}.png')
    return '', 404

@app.route('/api/regenerate-scene', methods=['POST'])
def regenerate_scene():
    """Regenerate a single scene's image and recompose the video."""
    data = request.json
    session_id = data.get('session_id')
    scene_num = data.get('scene_number')
    custom_prompt = data.get('custom_prompt')  # optional: user can edit the prompt
    
    if not session_id or not scene_num:
        return jsonify({'error': 'Missing session_id or scene_number'}), 400
    
    work_dir = os.path.join(app.config['OUTPUT_FOLDER'], session_id)
    state_path = os.path.join(work_dir, 'generation_state.json')
    
    if not os.path.exists(state_path):
        return jsonify({'error': 'Generation state not found — video may have been cleaned up'}), 404
    
    with open(state_path) as f:
        state = json.load(f)
    
    scenes = state['scenes']
    scene_videos = state['scene_videos']
    audio_path = state['audio_path']
    audio_duration = state['audio_duration']
    output_filename = state['output_filename']
    preset_id = state.get('preset_id')
    video_format = state.get('video_format', 'flash')
    has_subject = state.get('has_subject', False)
    
    # Find the scene
    scene_idx = None
    scene = None
    for i, s in enumerate(scenes):
        if s['scene_number'] == scene_num:
            scene_idx = i
            scene = s
            break
    
    if scene is None:
        return jsonify({'error': f'Scene {scene_num} not found'}), 404
    
    # Use custom prompt if provided, otherwise use existing
    prompt = custom_prompt if custom_prompt else scene['visual_description']
    
    # Setup Whisk session
    preset_config = get_preset(preset_id) if preset_id else None
    whisk_session = None
    if preset_config:
        whisk_session = upload_preset_images_to_whisk(preset_config, session_id)
        if whisk_session == "TOKEN_EXPIRED":
            return jsonify({'error': 'Whisk token expired — update in Railway settings'}), 401
    
    scene_has_subject = scene.get('has_subject', False) and has_subject
    img_path = os.path.join(work_dir, f'scene_{scene_num:04d}.png')
    
    # Regenerate image
    logger.info(f"Regenerating scene {scene_num} for session {session_id}")
    result = generate_image_whisk(prompt, img_path, session_id, scene_num, whisk_session, scene_has_subject)
    
    if result == "TOKEN_EXPIRED":
        return jsonify({'error': 'Whisk token expired'}), 401
    if not result:
        return jsonify({'error': 'Image generation failed'}), 500
    
    # Rebuild clip for this scene
    is_video = scene.get('is_video', False)
    duration = scene['end_time'] - scene['start_time']
    scene_effect = 'none'
    if not is_video and video_format == 'pulse':
        scene_effect = 'rotate_pulse'
    
    vid_path = os.path.join(work_dir, f'scene_{scene_num:04d}_video.mp4')
    
    if is_video and result and isinstance(result, dict):
        # Animate the new image
        anim_path = os.path.join(work_dir, f'scene_{scene_num:04d}_anim.mp4')
        animated = animate_image_whisk(result, prompt, anim_path, session_id, scene_num)
        if animated:
            trimmed = os.path.join(work_dir, f'scene_{scene_num:04d}_trimmed.mp4')
            cmd = ['ffmpeg', '-y', '-i', anim_path, '-t', str(duration),
                   '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', '-r', '25',
                   '-vf', 'scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080',
                   trimmed]
            subprocess.run(cmd, check=True, capture_output=True, timeout=120)
            vid_path = trimmed
        else:
            create_video_from_image(img_path, vid_path, duration, effect=scene_effect, scene_index=scene_idx)
    else:
        create_video_from_image(img_path, vid_path, duration, effect=scene_effect, scene_index=scene_idx)
    
    # Update scene_videos list
    scene_videos[scene_idx] = vid_path
    
    # Update the prompt if custom
    if custom_prompt:
        scenes[scene_idx]['visual_description'] = custom_prompt
    
    # Save updated state
    state['scenes'] = scenes
    state['scene_videos'] = scene_videos
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)
    
    # Recompose the full video
    logger.info(f"Recomposing video after scene {scene_num} regeneration")
    output_path = os.path.join(work_dir, output_filename)
    compose_final_video(scene_videos, audio_path, output_path, session_id, audio_duration=audio_duration)
    
    logger.info(f"Scene {scene_num} regenerated and video recomposed")
    return jsonify({
        'success': True,
        'scene_number': scene_num,
        'video_url': f'/download/{session_id}/{output_filename}',
        'image_url': f'/scene-image/{session_id}/{scene_num}',
        'scenes': scenes
    })

@app.route('/api/presets', methods=['GET'])
def list_presets():
    return jsonify(get_all_presets())

@app.route('/api/presets', methods=['POST'])
def create_preset():
    name = request.form.get('name', 'Untitled')
    style_text = request.form.get('style_text', '').strip()
    tags = request.form.get('tags', '').strip()
    style_b64 = None
    if 'style' in request.files:
        style_file = request.files['style']
        if style_file.filename and allowed_image(style_file.filename):
            style_b64 = base64.b64encode(style_file.read()).decode('utf-8')
    if not style_b64 and not style_text:
        return jsonify({'error': 'Provide a style image, text description, or both.'}), 400
    subject_b64 = None
    if 'subject' in request.files:
        sf = request.files['subject']
        if sf.filename and allowed_image(sf.filename):
            subject_b64 = base64.b64encode(sf.read()).decode('utf-8')
    preset_id = save_preset(name, style_b64, subject_b64, style_text, tags)
    return jsonify({'id': preset_id, 'message': 'Preset saved'})

@app.route('/api/presets/<preset_id>', methods=['DELETE'])
def remove_preset(preset_id):
    return jsonify({'message': 'Deleted'}) if delete_preset(preset_id) else (jsonify({'error': 'Not found'}), 404)

@app.route('/api/presets/reorder', methods=['POST'])
def reorder_presets():
    order = request.json.get('order', [])
    if not isinstance(order, list):
        return jsonify({'error': 'Invalid order'}), 400
    save_preset_order(order)
    return jsonify({'ok': True})

@app.route('/api/presets/<preset_id>/style.png')
def preset_style_image(preset_id):
    path = os.path.join(app.config['PRESET_FOLDER'], preset_id, 'style.png')
    return send_file(path, mimetype='image/png') if os.path.exists(path) else ('', 404)

@app.route('/api/presets/<preset_id>/subject.png')
def preset_subject_image(preset_id):
    path = os.path.join(app.config['PRESET_FOLDER'], preset_id, 'subject.png')
    return send_file(path, mimetype='image/png') if os.path.exists(path) else ('', 404)

@app.route('/api/presets/export')
def export_presets():
    """Export all presets as a JSON file with embedded base64 images."""
    presets = get_all_presets()
    export_data = []
    for p in presets:
        preset_dir = os.path.join(app.config['PRESET_FOLDER'], p['id'])
        entry = {'name': p['name'], 'has_subject': p.get('has_subject', False),
                 'style_text': p.get('style_text', ''), 'tags': p.get('tags', [])}
        # Embed style image
        style_path = os.path.join(preset_dir, 'style.png')
        if os.path.exists(style_path):
            with open(style_path, 'rb') as f:
                entry['style_b64'] = base64.b64encode(f.read()).decode('utf-8')
        # Embed subject image
        subject_path = os.path.join(preset_dir, 'subject.png')
        if os.path.exists(subject_path):
            with open(subject_path, 'rb') as f:
                entry['subject_b64'] = base64.b64encode(f.read()).decode('utf-8')
        export_data.append(entry)
    response = app.response_class(
        response=json.dumps(export_data, indent=2),
        mimetype='application/json',
        headers={'Content-Disposition': 'attachment; filename=presets-export.json'}
    )
    return response

@app.route('/api/presets/import', methods=['POST'])
def import_presets():
    """Import presets from a JSON export file."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    try:
        data = json.loads(file.read().decode('utf-8'))
    except Exception as e:
        return jsonify({'error': f'Invalid JSON file: {str(e)}'}), 400
    if not isinstance(data, list):
        return jsonify({'error': 'Invalid format — expected a list of presets'}), 400
    count = 0
    for entry in data:
        name = entry.get('name', 'Imported')
        style_b64 = entry.get('style_b64')
        subject_b64 = entry.get('subject_b64')
        style_text = entry.get('style_text', '')
        tags = ','.join(entry.get('tags', []))
        if not style_b64 and not style_text:
            continue
        save_preset(name, style_b64, subject_b64 if subject_b64 else None, style_text, tags)
        count += 1
    return jsonify({'ok': True, 'count': count})

@app.route('/version')
def version():
    return jsonify({"version": "v43", "features": ["presets", "subject", "long_form",
        "style_enforcement", "retry_logic", "resolution_1080p", "dark_mode", "audio_fix",
        "subject_detection", "scene_renumber", "style_text", "admin_dashboard"]})

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'openai': bool(OPENAI_API_KEY), 'gemini': bool(GEMINI_API_KEY),
                    'whisk_token': bool(get_whisk_token()), 'whisk_cookie': bool(get_whisk_cookie())})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)
