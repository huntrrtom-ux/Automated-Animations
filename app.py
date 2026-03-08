from gevent import monkey
monkey.patch_all()

import os
import sys
import json
import time
import uuid
import math
import copy
import shutil
import subprocess
import logging
import random
import tempfile
import threading
import bisect
import hashlib
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

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
# Use /data for persistent storage on Railway (volume mount), fallback to local for dev
PERSISTENT_DIR = '/data' if os.path.isdir('/data') else os.path.dirname(os.path.abspath(__file__))

app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
app.config['PRESET_FOLDER'] = os.path.join(PERSISTENT_DIR, 'presets')
app.config['CHANNEL_FOLDER'] = os.path.join(PERSISTENT_DIR, 'channels')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['PRESET_FOLDER'], exist_ok=True)
os.makedirs(app.config['CHANNEL_FOLDER'], exist_ok=True)

logger.info(f"Persistent storage: {PERSISTENT_DIR} ({'Railway volume' if PERSISTENT_DIR == '/data' else 'local'})")

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='gevent')

# Track active (in-progress) generation sessions
active_sessions = {}
_active_sessions_lock = threading.Lock()

# Cancellation flags — set session_id: True to request cancellation
_cancelled_sessions = {}
_cancelled_lock = threading.Lock()

def request_cancel(session_id):
    """Mark a session for cancellation."""
    with _cancelled_lock:
        _cancelled_sessions[session_id] = True

def is_cancelled(session_id):
    """Check if a session has been cancelled."""
    with _cancelled_lock:
        return _cancelled_sessions.get(session_id, False)

def clear_cancel(session_id):
    """Clear cancellation flag after cleanup."""
    with _cancelled_lock:
        _cancelled_sessions.pop(session_id, None)

class GenerationCancelled(Exception):
    """Raised when a user cancels an in-progress generation."""
    pass

def check_cancelled(session_id):
    """Raise GenerationCancelled if session was cancelled."""
    if is_cancelled(session_id):
        raise GenerationCancelled(f"Generation {session_id} cancelled by user")

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
ASSEMBLYAI_API_KEY = os.environ.get('ASSEMBLYAI_API_KEY')
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'flac', 'm4a', 'aac', 'webm', 'mp4'}
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_image(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def emit_progress(session_id, step, progress, message, data=None, log_type=None):
    payload = {'session_id': session_id, 'step': step, 'progress': progress, 'message': message}
    if data:
        payload['data'] = data
    if log_type:
        payload['log_type'] = log_type
    socketio.emit('progress', payload)
    logger.info(f"[{session_id}] {step}: {message} ({progress}%)")


# ===================== BASE FORMAT DEFINITIONS =====================
# Each channel gets a deep copy of one of these as its standalone format.
# Changes to a channel's format never affect these defaults.
BASE_FORMATS = {
    'pulse': {
        'base': 'pulse',
        'label': 'Pulse',
        'description': 'Fast-paced entertainment',
        'intro_duration': 30,
        'intro_animated': False,
        'intro_scene_min_duration': 1,
        'intro_scene_max_duration': 8,
        'body_scene_min_duration': 2,
        'body_scene_max_duration': 10,
        'body_animated': False,
        'periodic_animation_interval': 0,
        'periodic_animation_window': 30,
        'ken_burns_effect': 'none',
        'subject_mode': 'auto',
        'subject_interval': 0,
        'scene_detection_temperature': 0.4,
        'max_scene_duration': 10,
    },
    'flash': {
        'base': 'flash',
        'label': 'Flash',
        'description': 'Animated educational',
        'intro_duration': 120,
        'intro_animated': True,
        'intro_scene_min_duration': 3,
        'intro_scene_max_duration': 8,
        'body_scene_min_duration': 5,
        'body_scene_max_duration': 15,
        'body_animated': False,
        'periodic_animation_interval': 0,
        'periodic_animation_window': 0,
        'ken_burns_effect': 'none',
        'subject_mode': 'all',
        'subject_interval': 0,
        'scene_detection_temperature': 0.4,
        'max_scene_duration': 15,
    },
    'deep': {
        'base': 'deep',
        'label': 'Deep',
        'description': 'Longform educational',
        'intro_duration': 60,
        'intro_animated': False,
        'intro_scene_min_duration': 2,
        'intro_scene_max_duration': 10,
        'body_scene_min_duration': 2,
        'body_scene_max_duration': 15,
        'body_scene_min_duration_first_half': 2,
        'body_scene_max_duration_first_half': 7,
        'body_scene_min_duration_second_half': 8,
        'body_scene_max_duration_second_half': 15,
        'body_animated': False,
        'periodic_animation_interval': 0,
        'periodic_animation_window': 0,
        'ken_burns_effect': 'none',
        'subject_mode': 'sparse',
        'subject_interval': 300,
        'scene_detection_temperature': 0.4,
        'max_scene_duration': 15,
    },
    'botanical': {
        'base': 'botanical',
        'label': 'Botanical',
        'description': 'Plant-focused educational with subject character',
        'intro_duration': 30,
        'intro_animated': False,
        'intro_scene_min_duration': 2,
        'intro_scene_max_duration': 8,
        'body_scene_min_duration': 2,
        'body_scene_max_duration': 8,
        'body_animated': False,
        'periodic_animation_interval': 0,
        'periodic_animation_window': 0,
        'ken_burns_effect': 'none',
        'ken_burns_scale': 1.03,
        'subject_mode': 'botanical',
        'subject_interval': 0,
        'scene_detection_temperature': 0.4,
        'max_scene_duration': 8,
    },
    'tv-show-pov': {
        'base': 'tv-show-pov',
        'label': 'TV Show POV',
        'description': 'PIL frame-by-frame with persistent POV character',
        'intro_duration': 30,
        'intro_animated': False,
        'intro_scene_min_duration': 2,
        'intro_scene_max_duration': 8,
        'body_scene_min_duration': 2,
        'body_scene_max_duration': 8,
        'body_animated': False,
        'periodic_animation_interval': 0,
        'periodic_animation_window': 0,
        'ken_burns_effect': 'none',
        'subject_mode': 'pov',
        'subject_interval': 0,
        'scene_detection_temperature': 0.4,
        'max_scene_duration': 8,
    },
}


# ===================== CHANNEL MANAGEMENT =====================
def get_channel_registry():
    registry_path = os.path.join(app.config['CHANNEL_FOLDER'], '_registry.json')
    if os.path.exists(registry_path):
        try:
            with open(registry_path, 'r') as f:
                return json.load(f)
        except:
            pass
    return []

def save_channel_registry(order):
    registry_path = os.path.join(app.config['CHANNEL_FOLDER'], '_registry.json')
    with open(registry_path, 'w') as f:
        json.dump(order, f)

def get_all_channels():
    channels = []
    channel_dir = app.config['CHANNEL_FOLDER']
    channel_map = {}
    for name in os.listdir(channel_dir):
        if name.startswith('_') or not name.startswith('ch_'):
            continue
        config_path = os.path.join(channel_dir, name, 'config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                config['id'] = name
                config['has_logo'] = os.path.exists(os.path.join(channel_dir, name, 'logo.png'))
                channel_map[name] = config
            except Exception as e:
                logger.error(f"Failed to load channel {name}: {e}")
    # Collect all channels then sort alphabetically by first tag (no tags at end)
    channels = list(channel_map.values())
    def _tag_sort_key(ch):
        tags = ch.get('tags', [])
        if tags:
            return (0, tags[0].lower())
        return (1, '')
    channels.sort(key=_tag_sort_key)
    # Add format_tailored flag: True if channel format differs from its base defaults
    for ch in channels:
        fmt = ch.get('format', {})
        base_key = fmt.get('base', 'flash')
        base_fmt = BASE_FORMATS.get(base_key, BASE_FORMATS['flash'])
        ch['format_tailored'] = any(fmt.get(k) != base_fmt.get(k) for k in base_fmt if k not in ('base', 'label', 'description'))
    return channels

def get_channel(channel_id):
    channel_dir = os.path.join(app.config['CHANNEL_FOLDER'], channel_id)
    config_path = os.path.join(channel_dir, 'config.json')
    if not os.path.exists(config_path):
        return None
    with open(config_path, 'r') as f:
        config = json.load(f)
    config['id'] = channel_id
    config['has_logo'] = os.path.exists(os.path.join(channel_dir, 'logo.png'))
    style_path = os.path.join(channel_dir, 'style.png')
    if os.path.exists(style_path):
        with open(style_path, 'rb') as f:
            config['style_base64'] = base64.b64encode(f.read()).decode('utf-8')
    subject_path = os.path.join(channel_dir, 'subject.png')
    if os.path.exists(subject_path):
        with open(subject_path, 'rb') as f:
            config['subject_base64'] = base64.b64encode(f.read()).decode('utf-8')
    return config

def _resize_logo(path, size=800):
    """Resize a logo image to size×size pixels for consistent quality."""
    try:
        img = Image.open(path)
        img = img.convert('RGBA')
        img = img.resize((size, size), Image.LANCZOS)
        img.save(path, 'PNG')
    except Exception as e:
        logger.warning(f"Logo resize failed: {e}")


def save_channel(name, base_format='pulse', style_data=None, subject_data=None, logo_data=None,
                 style_text='', tags='', tag_colors=None, scene_instructions='', image_instructions='',
                 character_instructions=''):
    channel_id = 'ch_' + str(uuid.uuid4())[:8]
    channel_dir = os.path.join(app.config['CHANNEL_FOLDER'], channel_id)
    os.makedirs(channel_dir, exist_ok=True)
    if style_data:
        with open(os.path.join(channel_dir, 'style.png'), 'wb') as f:
            f.write(base64.b64decode(style_data))
    if subject_data:
        with open(os.path.join(channel_dir, 'subject.png'), 'wb') as f:
            f.write(base64.b64decode(subject_data))
    if logo_data:
        logo_path = os.path.join(channel_dir, 'logo.png')
        with open(logo_path, 'wb') as f:
            f.write(base64.b64decode(logo_data))
        _resize_logo(logo_path)
    tag_list = [t.strip() for t in tags.split(',') if t.strip()] if isinstance(tags, str) else (tags or [])
    format_config = copy.deepcopy(BASE_FORMATS.get(base_format, BASE_FORMATS['pulse']))
    config = {
        'name': name,
        'tags': tag_list,
        'tag_colors': tag_colors or {},
        'style_text': style_text,
        'scene_instructions': scene_instructions,
        'image_instructions': image_instructions,
        'character_instructions': character_instructions,
        'format': format_config,
        'has_style_image': style_data is not None,
        'has_subject': subject_data is not None,
        'created_at': time.strftime('%Y-%m-%d %H:%M'),
        'updated_at': time.strftime('%Y-%m-%d %H:%M'),
    }
    with open(os.path.join(channel_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    registry = get_channel_registry()
    registry.append(channel_id)
    save_channel_registry(registry)
    return channel_id

def update_channel(channel_id, **fields):
    channel_dir = os.path.join(app.config['CHANNEL_FOLDER'], channel_id)
    config_path = os.path.join(channel_dir, 'config.json')
    if not os.path.exists(config_path):
        return False
    with open(config_path, 'r') as f:
        config = json.load(f)
    for key in ['name', 'tags', 'tag_colors', 'style_text', 'scene_instructions', 'image_instructions', 'character_instructions']:
        if key in fields and fields[key] is not None:
            config[key] = fields[key]
    if 'format' in fields and fields['format'] is not None:
        config['format'] = fields['format']
    if 'animation_pattern' in fields:
        config.setdefault('format', {})['animation_pattern'] = fields['animation_pattern']
    if fields.get('style_data'):
        with open(os.path.join(channel_dir, 'style.png'), 'wb') as f:
            f.write(base64.b64decode(fields['style_data']))
        config['has_style_image'] = True
    if fields.get('subject_data'):
        with open(os.path.join(channel_dir, 'subject.png'), 'wb') as f:
            f.write(base64.b64decode(fields['subject_data']))
        config['has_subject'] = True
    if fields.get('logo_data'):
        logo_path = os.path.join(channel_dir, 'logo.png')
        with open(logo_path, 'wb') as f:
            f.write(base64.b64decode(fields['logo_data']))
        _resize_logo(logo_path)
    if fields.get('remove_subject'):
        subject_path = os.path.join(channel_dir, 'subject.png')
        if os.path.exists(subject_path):
            os.remove(subject_path)
        config['has_subject'] = False
    if fields.get('remove_logo'):
        logo_path = os.path.join(channel_dir, 'logo.png')
        if os.path.exists(logo_path):
            os.remove(logo_path)
    if fields.get('remove_style'):
        style_path = os.path.join(channel_dir, 'style.png')
        if os.path.exists(style_path):
            os.remove(style_path)
        config['has_style_image'] = False
    config['updated_at'] = time.strftime('%Y-%m-%d %H:%M')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    return True

def delete_channel(channel_id):
    channel_dir = os.path.join(app.config['CHANNEL_FOLDER'], channel_id)
    if os.path.exists(channel_dir):
        shutil.rmtree(channel_dir)
        registry = get_channel_registry()
        if channel_id in registry:
            registry.remove(channel_id)
            save_channel_registry(registry)
        return True
    return False


# ===================== PRESET MIGRATION =====================
def _get_preset_order():
    """Read old preset order for migration only."""
    order_path = os.path.join(app.config['PRESET_FOLDER'], '_order.json')
    if os.path.exists(order_path):
        try:
            with open(order_path, 'r') as f:
                return json.load(f)
        except:
            pass
    return []

def migrate_presets_to_channels():
    """One-time migration: convert /data/presets/ to /data/channels/."""
    registry_path = os.path.join(app.config['CHANNEL_FOLDER'], '_registry.json')
    if os.path.exists(registry_path):
        return  # Already migrated

    preset_dir = app.config['PRESET_FOLDER']
    if not os.path.isdir(preset_dir):
        save_channel_registry([])
        return

    logger.info("=== MIGRATING PRESETS TO CHANNELS ===")
    order = _get_preset_order()
    preset_ids = []
    for name in os.listdir(preset_dir):
        if name.startswith('_'):
            continue
        if os.path.exists(os.path.join(preset_dir, name, 'config.json')):
            preset_ids.append(name)

    ordered = [pid for pid in order if pid in preset_ids]
    for pid in preset_ids:
        if pid not in ordered:
            ordered.append(pid)

    if not ordered:
        save_channel_registry([])
        logger.info("No presets to migrate")
        return

    channel_ids = []
    for preset_id in ordered:
        preset_path = os.path.join(preset_dir, preset_id)
        config_path = os.path.join(preset_path, 'config.json')
        try:
            with open(config_path) as f:
                preset_config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read preset {preset_id}: {e}")
            continue

        channel_id = f'ch_{preset_id}'
        channel_path = os.path.join(app.config['CHANNEL_FOLDER'], channel_id)
        os.makedirs(channel_path, exist_ok=True)

        for img_name in ['style.png', 'subject.png']:
            src = os.path.join(preset_path, img_name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(channel_path, img_name))

        channel_config = {
            'name': preset_config.get('name', 'Untitled'),
            'tags': preset_config.get('tags', []),
            'tag_colors': preset_config.get('tag_colors', {}),
            'style_text': preset_config.get('style_text', ''),
            'scene_instructions': '',
            'image_instructions': '',
            'format': copy.deepcopy(BASE_FORMATS['pulse']),
            'has_style_image': preset_config.get('has_style_image', False),
            'has_subject': preset_config.get('has_subject', False),
            'created_at': preset_config.get('created_at', time.strftime('%Y-%m-%d %H:%M')),
            'updated_at': time.strftime('%Y-%m-%d %H:%M'),
            'migrated_from_preset': preset_id,
        }
        with open(os.path.join(channel_path, 'config.json'), 'w') as f:
            json.dump(channel_config, f, indent=2)
        channel_ids.append(channel_id)
        logger.info(f"  Migrated preset '{preset_config.get('name')}' -> {channel_id}")

    save_channel_registry(channel_ids)
    logger.info(f"=== MIGRATION COMPLETE: {len(channel_ids)} channels ===")

# Run migration on startup
migrate_presets_to_channels()


def migrate_v53_channel_updates():
    """One-time migration: switch all channels to flash, copy+rename Pet Psychology, rename others."""
    flag_path = os.path.join(app.config['CHANNEL_FOLDER'], '_migration_v53.done')
    if os.path.exists(flag_path):
        return

    logger.info("=== V53 MIGRATION: Starting channel updates ===")
    channel_dir = app.config['CHANNEL_FOLDER']

    # Build name->id map
    name_to_id = {}
    for name in os.listdir(channel_dir):
        if not name.startswith('ch_'):
            continue
        config_path = os.path.join(channel_dir, name, 'config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    cfg = json.load(f)
                name_to_id[cfg.get('name', '')] = name
            except Exception:
                pass

    # 1. Switch ALL channels to flash format
    flash_format = copy.deepcopy(BASE_FORMATS['flash'])
    for ch_name, ch_id in name_to_id.items():
        config_path = os.path.join(channel_dir, ch_id, 'config.json')
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        cfg['format'] = copy.deepcopy(flash_format)
        with open(config_path, 'w') as f:
            json.dump(cfg, f, indent=2)
        logger.info(f"  Switched '{ch_name}' to flash format")

    # 2. Copy+rename Pet Psychology - Flash -> 3 channels
    pet_psych_id = name_to_id.get('Pet Psychology - Flash')
    if pet_psych_id:
        pet_src = os.path.join(channel_dir, pet_psych_id)
        new_names = ['Psychology of Cats', 'Whisker Theory', 'Psychology of Dogs']
        registry = get_channel_registry()

        # Rename the original to the first new name
        config_path = os.path.join(pet_src, 'config.json')
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        cfg['name'] = new_names[0]
        cfg['updated_at'] = time.strftime('%Y-%m-%d %H:%M')
        with open(config_path, 'w') as f:
            json.dump(cfg, f, indent=2)
        logger.info(f"  Renamed 'Pet Psychology - Flash' -> '{new_names[0]}'")

        # Create 2 copies for the remaining names
        for copy_name in new_names[1:]:
            new_id = 'ch_' + str(uuid.uuid4())[:8]
            new_dir = os.path.join(channel_dir, new_id)
            shutil.copytree(pet_src, new_dir)
            cp_config_path = os.path.join(new_dir, 'config.json')
            with open(cp_config_path, 'r') as f:
                cp_cfg = json.load(f)
            cp_cfg['name'] = copy_name
            cp_cfg['updated_at'] = time.strftime('%Y-%m-%d %H:%M')
            with open(cp_config_path, 'w') as f:
                json.dump(cp_cfg, f, indent=2)
            registry.append(new_id)
            logger.info(f"  Created copy '{copy_name}' as {new_id}")

        save_channel_registry(registry)
    else:
        logger.warning("  'Pet Psychology - Flash' not found, skipping copies")

    # 3. Rename channels
    renames = {
        'Pastel - Pregnancy Advice - Flash': 'Mama Knowledge',
        'Watercolour - Pregnancy Advice - Flash': 'Pregnancy with Grace',
        'Fatherhood Advice - Flash': 'Coach Dan',
        'Sally - Housing Market - Flash': 'Sally Saves',
        'Betty - Housing Market - Flash': 'Financial Betty',
        'Howard - Housing Market - Flash': 'Housing Howard',
    }
    # Re-read name map since Pet Psychology was renamed
    name_to_id_fresh = {}
    for dname in os.listdir(channel_dir):
        if not dname.startswith('ch_'):
            continue
        cp = os.path.join(channel_dir, dname, 'config.json')
        if os.path.exists(cp):
            try:
                with open(cp, 'r') as f:
                    c = json.load(f)
                name_to_id_fresh[c.get('name', '')] = dname
            except Exception:
                pass

    for old_name, new_name in renames.items():
        ch_id = name_to_id_fresh.get(old_name)
        if ch_id:
            cp = os.path.join(channel_dir, ch_id, 'config.json')
            with open(cp, 'r') as f:
                cfg = json.load(f)
            cfg['name'] = new_name
            cfg['updated_at'] = time.strftime('%Y-%m-%d %H:%M')
            with open(cp, 'w') as f:
                json.dump(cfg, f, indent=2)
            logger.info(f"  Renamed '{old_name}' -> '{new_name}'")
        else:
            logger.warning(f"  '{old_name}' not found, skipping rename")

    # Write flag file
    with open(flag_path, 'w') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S'))
    logger.info("=== V53 MIGRATION COMPLETE ===")

migrate_v53_channel_updates()


def migrate_pregnancy_explainer_channel():
    """One-time migration: duplicate Mama Knowledge as Pregnancy Explainer with topic title cards."""
    flag_path = os.path.join(app.config['CHANNEL_FOLDER'], '_migration_pregnancy_explainer.done')
    if os.path.exists(flag_path):
        return

    logger.info("=== PREGNANCY EXPLAINER MIGRATION: Starting ===")
    channel_dir = app.config['CHANNEL_FOLDER']

    # Find Mama Knowledge by name
    mama_id = None
    for name in os.listdir(channel_dir):
        if not name.startswith('ch_'):
            continue
        config_path = os.path.join(channel_dir, name, 'config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    cfg = json.load(f)
                if cfg.get('name') == 'Mama Knowledge':
                    mama_id = name
                    break
            except Exception:
                pass

    if not mama_id:
        logger.warning("  'Mama Knowledge' not found, skipping Pregnancy Explainer creation")
        with open(flag_path, 'w') as f:
            f.write(time.strftime('%Y-%m-%d %H:%M:%S'))
        return

    # Copy entire channel directory (config, style, subject, logo)
    new_id = 'ch_' + str(uuid.uuid4())[:8]
    src_dir = os.path.join(channel_dir, mama_id)
    dst_dir = os.path.join(channel_dir, new_id)
    shutil.copytree(src_dir, dst_dir)

    # Update config
    cp_config_path = os.path.join(dst_dir, 'config.json')
    with open(cp_config_path, 'r') as f:
        cfg = json.load(f)
    cfg['name'] = 'Pregnancy Explainer'
    cfg['tags'] = []  # User will add tags via admin dashboard
    cfg['tag_colors'] = {}
    cfg.setdefault('format', {})['topic_title_cards'] = True
    cfg['updated_at'] = time.strftime('%Y-%m-%d %H:%M')
    with open(cp_config_path, 'w') as f:
        json.dump(cfg, f, indent=2)

    # Add to registry
    registry = get_channel_registry()
    registry.append(new_id)
    save_channel_registry(registry)

    logger.info(f"  Created 'Pregnancy Explainer' as {new_id} (copied from Mama Knowledge {mama_id})")

    with open(flag_path, 'w') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S'))
    logger.info("=== PREGNANCY EXPLAINER MIGRATION COMPLETE ===")

migrate_pregnancy_explainer_channel()


def migrate_pregnancy_explainer_safety():
    """One-time migration: add content safety image_instructions to Pregnancy Explainer."""
    flag_path = os.path.join(app.config['CHANNEL_FOLDER'], '_migration_pregnancy_safety.done')
    if os.path.exists(flag_path):
        return

    logger.info("=== PREGNANCY EXPLAINER SAFETY MIGRATION ===")
    channel_dir = app.config['CHANNEL_FOLDER']

    for name in os.listdir(channel_dir):
        if not name.startswith('ch_'):
            continue
        config_path = os.path.join(channel_dir, name, 'config.json')
        if not os.path.exists(config_path):
            continue
        try:
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            if cfg.get('name') != 'Pregnancy Explainer':
                continue

            cfg['image_instructions'] = (
                "Family-friendly medical education illustration. "
                "All characters must be fully clothed and modest at all times. "
                "No nudity, no exposed bodies, no genitals, no revealing clothing, "
                "no graphic medical imagery, no anatomical diagrams, no surgical scenes. "
                "Use tasteful, non-explicit, symbolic visuals appropriate for all audiences "
                "and all environments including workplaces and classrooms."
            )
            cfg['updated_at'] = time.strftime('%Y-%m-%d %H:%M')
            with open(config_path, 'w') as f:
                json.dump(cfg, f, indent=2)
            logger.info(f"  Updated '{cfg['name']}' ({name}) with content safety image_instructions")
        except Exception as e:
            logger.error(f"  Failed to update {name}: {e}")

    with open(flag_path, 'w') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S'))
    logger.info("=== PREGNANCY EXPLAINER SAFETY MIGRATION COMPLETE ===")

migrate_pregnancy_explainer_safety()


def migrate_pregnancy_explainer_female_only():
    """One-time migration: add female-only character constraint to Pregnancy Explainer."""
    flag_path = os.path.join(app.config['CHANNEL_FOLDER'], '_migration_pregnancy_female.done')
    if os.path.exists(flag_path):
        return

    logger.info("=== PREGNANCY EXPLAINER FEMALE-ONLY MIGRATION ===")
    channel_dir = app.config['CHANNEL_FOLDER']

    for name in os.listdir(channel_dir):
        if not name.startswith('ch_'):
            continue
        config_path = os.path.join(channel_dir, name, 'config.json')
        if not os.path.exists(config_path):
            continue
        try:
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            if cfg.get('name') != 'Pregnancy Explainer':
                continue

            cfg['image_instructions'] = (
                "Family-friendly medical education illustration. "
                "All characters must be female women — never depict male characters. "
                "All characters must be fully clothed and modest at all times. "
                "No nudity, no exposed bodies, no genitals, no revealing clothing, "
                "no graphic medical imagery, no anatomical diagrams, no surgical scenes. "
                "Use tasteful, non-explicit, symbolic visuals appropriate for all audiences "
                "and all environments including workplaces and classrooms."
            )
            cfg['updated_at'] = time.strftime('%Y-%m-%d %H:%M')
            with open(config_path, 'w') as f:
                json.dump(cfg, f, indent=2)
            logger.info(f"  Updated '{cfg['name']}' ({name}) with female-only image_instructions")
        except Exception as e:
            logger.error(f"  Failed to update {name}: {e}")

    with open(flag_path, 'w') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S'))
    logger.info("=== PREGNANCY EXPLAINER FEMALE-ONLY MIGRATION COMPLETE ===")

migrate_pregnancy_explainer_female_only()


def migrate_pregnancy_explainer_no_title_cards():
    """One-time migration: disable title cards & silence gaps, extend Veo animation to 180s."""
    flag_path = os.path.join(app.config['CHANNEL_FOLDER'], '_migration_pregnancy_no_titlecards.done')
    if os.path.exists(flag_path):
        return

    logger.info("=== PREGNANCY EXPLAINER: REMOVE TITLE CARDS + EXTEND VEO ===")
    channel_dir = app.config['CHANNEL_FOLDER']

    for name in os.listdir(channel_dir):
        if not name.startswith('ch_'):
            continue
        config_path = os.path.join(channel_dir, name, 'config.json')
        if not os.path.exists(config_path):
            continue
        try:
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            if cfg.get('name') != 'Pregnancy Explainer':
                continue

            fmt = cfg.setdefault('format', {})
            # Disable title cards and silence gaps
            fmt['topic_title_cards'] = False
            # Extend Veo animated intro to 180 seconds
            fmt['intro_duration'] = 180
            fmt['intro_animated'] = True
            cfg['updated_at'] = time.strftime('%Y-%m-%d %H:%M')
            with open(config_path, 'w') as f:
                json.dump(cfg, f, indent=2)
            logger.info(f"  Updated '{cfg['name']}' ({name}): title_cards=False, intro_duration=180, intro_animated=True")
        except Exception as e:
            logger.error(f"  Failed to update {name}: {e}")

    with open(flag_path, 'w') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S'))
    logger.info("=== PREGNANCY EXPLAINER: REMOVE TITLE CARDS + EXTEND VEO COMPLETE ===")

migrate_pregnancy_explainer_no_title_cards()


def migrate_space_theories_channel():
    """One-time migration: create 'Space Theories Explained' channel with alternating animation."""
    flag_path = os.path.join(app.config['CHANNEL_FOLDER'], '_migration_space_theories.done')
    if os.path.exists(flag_path):
        return

    logger.info("=== SPACE THEORIES EXPLAINED MIGRATION: Starting ===")

    # Create a fresh channel based on Pulse format
    channel_id = save_channel(
        name='Space Theories Explained',
        base_format='pulse',
        scene_instructions=(
            "Include the subject character in approximately 70% of all scenes. "
            "Leave some scenes as pure environment or space visuals without the subject.\n"
            "For EVERY scene with the subject, describe their specific movements, facial expressions, "
            "and hand/body gestures in vivid detail. Never leave the subject static or vaguely described."
        ),
        image_instructions=(
            "Graphic, exaggerated, dramatic visual style. Use bold compositions, extreme perspectives, "
            "vivid lighting, and theatrical staging. Scenes should feel cinematic and larger-than-life."
        ),
    )

    # Patch format config with Space Theories overrides
    channel_dir = os.path.join(app.config['CHANNEL_FOLDER'], channel_id)
    config_path = os.path.join(channel_dir, 'config.json')
    with open(config_path, 'r') as f:
        cfg = json.load(f)

    fmt = cfg['format']
    fmt['label'] = 'Space Theories'
    fmt['description'] = 'Animated space theories with exaggerated visuals'
    fmt['animation_pattern'] = 'alternating'
    fmt['intro_animated'] = False
    fmt['intro_scene_min_duration'] = 2
    fmt['intro_scene_max_duration'] = 8
    fmt['body_scene_min_duration'] = 2
    fmt['body_scene_max_duration'] = 8
    fmt['max_scene_duration'] = 8
    fmt['subject_mode'] = 'auto'
    fmt['subject_interval'] = 0
    cfg['updated_at'] = time.strftime('%Y-%m-%d %H:%M')

    with open(config_path, 'w') as f:
        json.dump(cfg, f, indent=2)

    logger.info(f"  Created 'Space Theories Explained' as {channel_id} with alternating animation, 2-8s scenes")

    with open(flag_path, 'w') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S'))
    logger.info("=== SPACE THEORIES EXPLAINED MIGRATION COMPLETE ===")

migrate_space_theories_channel()


def migrate_plants_explainer_channel():
    """One-time migration: create Plants Explainer channel with botanical format."""
    flag_path = os.path.join(app.config['CHANNEL_FOLDER'], '_migration_plants_explainer.done')
    if os.path.exists(flag_path):
        return

    logger.info("=== PLANTS EXPLAINER MIGRATION: Creating channel ===")

    # Check if channel already exists by name
    channel_dir = app.config['CHANNEL_FOLDER']
    for name in os.listdir(channel_dir):
        if not name.startswith('ch_'):
            continue
        config_path = os.path.join(channel_dir, name, 'config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    cfg = json.load(f)
                if cfg.get('name') == 'Plants Explainer':
                    logger.info("  'Plants Explainer' already exists — skipping creation")
                    with open(flag_path, 'w') as f:
                        f.write(time.strftime('%Y-%m-%d %H:%M:%S'))
                    return
            except Exception:
                pass

    channel_id = save_channel(
        name='Plants Explainer',
        base_format='botanical',
        tags='',
        tag_colors={},
        scene_instructions=(
            "This channel covers specific plants in educational segments. "
            "Each section of the transcript discusses a particular plant — identify it from context "
            "and ensure every scene in that section depicts ONLY that specific plant. "
            "Alternate between close-up botanical detail shots and wider scenes showing the plant "
            "in its natural habitat or garden setting."
        ),
        image_instructions=(
            "Lush, detailed botanical illustration style. Rich natural lighting, vivid greens and earth tones. "
            "Every plant must be accurately depicted with correct leaf shapes, flower structures, and growth habits."
        ),
    )

    # The botanical base format already has the right settings,
    # but let's confirm the key fields are set correctly
    ch_dir = os.path.join(app.config['CHANNEL_FOLDER'], channel_id)
    config_path = os.path.join(ch_dir, 'config.json')
    with open(config_path, 'r') as f:
        cfg = json.load(f)

    fmt = cfg['format']
    fmt['label'] = 'Botanical'
    fmt['description'] = 'Plant-focused educational with subject character'
    cfg['updated_at'] = time.strftime('%Y-%m-%d %H:%M')

    with open(config_path, 'w') as f:
        json.dump(cfg, f, indent=2)

    logger.info(f"  Created 'Plants Explainer' as {channel_id} with botanical format, 2-8s scenes, no Ken Burns zoom")

    with open(flag_path, 'w') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S'))
    logger.info("=== PLANTS EXPLAINER MIGRATION COMPLETE ===")

migrate_plants_explainer_channel()


def migrate_tv_show_pov_channel():
    """One-time migration: create TV Show POV channel with tv-show-pov format."""
    flag_path = os.path.join(app.config['CHANNEL_FOLDER'], '_migration_tv_show_pov.done')
    if os.path.exists(flag_path):
        return

    logger.info("=== TV SHOW POV MIGRATION: Creating channel ===")

    # Check if channel already exists by name
    channel_dir = app.config['CHANNEL_FOLDER']
    for name in os.listdir(channel_dir):
        if not name.startswith('ch_'):
            continue
        config_path = os.path.join(channel_dir, name, 'config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    cfg = json.load(f)
                if cfg.get('name') == 'TV Show POV':
                    logger.info("  'TV Show POV' already exists — skipping creation")
                    with open(flag_path, 'w') as f:
                        f.write(time.strftime('%Y-%m-%d %H:%M:%S'))
                    return
            except Exception:
                pass

    channel_id = save_channel(
        name='TV Show POV',
        base_format='tv-show-pov',
        tags='',
        tag_colors={},
        scene_instructions=(
            "This is a TV-show-style narration with a single persistent POV character "
            "who should appear in roughly 80% of scenes — they are the protagonist. "
            "Detect the time period and setting from the transcript (medieval, modern, sci-fi, etc.) "
            "and ensure ALL scenes reflect that era consistently. "
            "Vary camera angles: wide establishing shots with character visible, medium shots of character "
            "interacting with the environment, close-ups of character reactions, "
            "and occasionally first-person POV shots showing what the character sees through their eyes "
            "(e.g. looking down at their own hands, peering through a doorway, gazing across a landscape)."
        ),
        image_instructions=(
            "Cinematic TV-show quality. Dramatic lighting, rich color grading, depth of field. "
            "The POV character must be visually consistent across all frames — same face, same build, "
            "same core appearance. Attire should match the era and situation depicted in the transcript."
        ),
        character_instructions='',
    )

    # Patch format config
    ch_dir = os.path.join(app.config['CHANNEL_FOLDER'], channel_id)
    config_path = os.path.join(ch_dir, 'config.json')
    with open(config_path, 'r') as f:
        cfg = json.load(f)

    fmt = cfg['format']
    fmt['label'] = 'TV Show POV'
    fmt['description'] = 'PIL frame-by-frame with persistent POV character'
    cfg['updated_at'] = time.strftime('%Y-%m-%d %H:%M')

    with open(config_path, 'w') as f:
        json.dump(cfg, f, indent=2)

    logger.info(f"  Created 'TV Show POV' as {channel_id} with tv-show-pov format")

    with open(flag_path, 'w') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S'))
    logger.info("=== TV SHOW POV MIGRATION COMPLETE ===")

migrate_tv_show_pov_channel()


def migrate_fix_tv_show_pov_label():
    """One-time fix: rename 'Botanical – POV Character' label to 'TV Show POV'."""
    flag_path = os.path.join(app.config['CHANNEL_FOLDER'], '_migration_fix_tvshowpov_label.done')
    if os.path.exists(flag_path):
        return

    channel_dir = app.config['CHANNEL_FOLDER']
    for name in os.listdir(channel_dir):
        if not name.startswith('ch_'):
            continue
        config_path = os.path.join(channel_dir, name, 'config.json')
        if not os.path.exists(config_path):
            continue
        try:
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            fmt = cfg.get('format', {})
            if fmt.get('base') == 'tv-show-pov' and fmt.get('label') != 'TV Show POV':
                old_label = fmt.get('label', '?')
                fmt['label'] = 'TV Show POV'
                cfg['updated_at'] = time.strftime('%Y-%m-%d %H:%M')
                with open(config_path, 'w') as f:
                    json.dump(cfg, f, indent=2)
                logger.info(f"  Fixed TV Show POV label: '{old_label}' -> 'TV Show POV' in {name}")
        except Exception:
            pass

    with open(flag_path, 'w') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S'))
    logger.info("=== TV SHOW POV LABEL FIX COMPLETE ===")

migrate_fix_tv_show_pov_label()


def migrate_clear_new_channel_tags():
    """One-time fix: remove auto-assigned tags from botanical and tv-show-pov channels."""
    flag_path = os.path.join(app.config['CHANNEL_FOLDER'], '_migration_clear_new_channel_tags.done')
    if os.path.exists(flag_path):
        return

    channel_dir = app.config['CHANNEL_FOLDER']
    for name in os.listdir(channel_dir):
        if not name.startswith('ch_'):
            continue
        config_path = os.path.join(channel_dir, name, 'config.json')
        if not os.path.exists(config_path):
            continue
        try:
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            fmt_base = cfg.get('format', {}).get('base', '')
            if fmt_base in ('botanical', 'tv-show-pov') and cfg.get('tags'):
                old_tags = cfg['tags']
                cfg['tags'] = []
                cfg['tag_colors'] = {}
                cfg['updated_at'] = time.strftime('%Y-%m-%d %H:%M')
                with open(config_path, 'w') as f:
                    json.dump(cfg, f, indent=2)
                logger.info(f"  Cleared tags from {name}: {old_tags}")
        except Exception:
            pass

    with open(flag_path, 'w') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S'))
    logger.info("=== CLEAR NEW CHANNEL TAGS COMPLETE ===")

migrate_clear_new_channel_tags()


def migrate_fix_tv_show_pov_scene_instructions():
    """One-time fix: update TV Show POV scene_instructions from 'EVERY scene' to '80%'."""
    flag_path = os.path.join(app.config['CHANNEL_FOLDER'], '_migration_fix_tvshowpov_instructions.done')
    if os.path.exists(flag_path):
        return

    channel_dir = app.config['CHANNEL_FOLDER']
    for name in os.listdir(channel_dir):
        if not name.startswith('ch_'):
            continue
        config_path = os.path.join(channel_dir, name, 'config.json')
        if not os.path.exists(config_path):
            continue
        try:
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            if cfg.get('format', {}).get('base') != 'tv-show-pov':
                continue
            si = cfg.get('scene_instructions', '')
            if 'must appear in EVERY scene' in si:
                cfg['scene_instructions'] = si.replace(
                    'The character must appear in EVERY scene — they are the protagonist.',
                    'who should appear in roughly 80% of scenes — they are the protagonist.'
                )
                cfg['updated_at'] = time.strftime('%Y-%m-%d %H:%M')
                with open(config_path, 'w') as f:
                    json.dump(cfg, f, indent=2)
                logger.info(f"  Fixed TV Show POV scene_instructions in {name}: EVERY -> 80%")
        except Exception:
            pass

    with open(flag_path, 'w') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S'))
    logger.info("=== TV SHOW POV SCENE INSTRUCTIONS FIX COMPLETE ===")

migrate_fix_tv_show_pov_scene_instructions()


def migrate_fix_botanical_pov_character_base():
    """One-time fix: channels still carrying the old 'botanical-pov-character' base
    format need the base renamed to 'tv-show-pov', tags cleared, label/description
    updated, and scene_instructions corrected.  Earlier migrations missed these
    channels because they only matched base == 'tv-show-pov'."""
    flag_path = os.path.join(app.config['CHANNEL_FOLDER'],
                             '_migration_fix_botanical_pov_char_base.done')
    if os.path.exists(flag_path):
        return

    channel_dir = app.config['CHANNEL_FOLDER']
    for name in os.listdir(channel_dir):
        if not name.startswith('ch_'):
            continue
        config_path = os.path.join(channel_dir, name, 'config.json')
        if not os.path.exists(config_path):
            continue
        try:
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            fmt = cfg.get('format', {})
            if fmt.get('base') != 'botanical-pov-character':
                continue

            # --- Rename base format ---
            fmt['base'] = 'tv-show-pov'
            fmt['label'] = 'TV Show POV'
            fmt['description'] = 'PIL frame-by-frame with persistent POV character'
            # Copy over correct format settings from BASE_FORMATS
            canonical = BASE_FORMATS.get('tv-show-pov')
            if canonical:
                for key in ('subject_mode', 'ken_burns_effect', 'intro_scene_duration',
                            'min_scene_duration', 'max_scene_duration', 'animation_pattern'):
                    if key in canonical:
                        fmt[key] = canonical[key]

            # --- Clear tags ---
            cfg['tags'] = []
            cfg['tag_colors'] = {}

            # --- Fix scene instructions (EVERY -> 80%) ---
            si = cfg.get('scene_instructions', '')
            if 'must appear in EVERY scene' in si:
                cfg['scene_instructions'] = si.replace(
                    'The character must appear in EVERY scene — they are the protagonist.',
                    'who should appear in roughly 80% of scenes — they are the protagonist.'
                )

            cfg['updated_at'] = time.strftime('%Y-%m-%d %H:%M')
            with open(config_path, 'w') as f:
                json.dump(cfg, f, indent=2)
            logger.info(f"  Fixed botanical-pov-character -> tv-show-pov in {name} "
                        f"(base, label, tags, scene_instructions)")
        except Exception:
            pass

    with open(flag_path, 'w') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S'))
    logger.info("=== BOTANICAL-POV-CHARACTER BASE FIX COMPLETE ===")

migrate_fix_botanical_pov_character_base()


def migrate_disable_ken_burns_botanical_tvshow():
    """One-time migration: disable Ken Burns for botanical and tv-show-pov channels."""
    flag_path = os.path.join(app.config['CHANNEL_FOLDER'], '_migration_disable_kb_botanical_tvshow.done')
    if os.path.exists(flag_path):
        return

    logger.info("=== MIGRATION: Disabling Ken Burns for botanical/tv-show-pov channels ===")
    channel_dir = app.config['CHANNEL_FOLDER']
    patched = 0
    for name in os.listdir(channel_dir):
        if not name.startswith('ch_'):
            continue
        config_path = os.path.join(channel_dir, name, 'config.json')
        if not os.path.exists(config_path):
            continue
        try:
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            fmt = cfg.get('format', {})
            if fmt.get('base') in ('tv-show-pov',) and fmt.get('ken_burns_effect') != 'none':
                fmt['ken_burns_effect'] = 'none'
                cfg['updated_at'] = time.strftime('%Y-%m-%d %H:%M')
                with open(config_path, 'w') as f:
                    json.dump(cfg, f, indent=2)
                patched += 1
                logger.info(f"  Patched {name}: ken_burns_effect -> none")
        except Exception as e:
            logger.error(f"  Failed to patch {name}: {e}")

    with open(flag_path, 'w') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S'))
    logger.info(f"=== MIGRATION COMPLETE: {patched} channel(s) patched ===")

migrate_disable_ken_burns_botanical_tvshow()


def migrate_enable_subtle_zoom_botanical():
    """One-time migration: enable subtle zoom_in for botanical channels."""
    flag_path = os.path.join(app.config['CHANNEL_FOLDER'], '_migration_enable_zoom_botanical.done')
    if os.path.exists(flag_path):
        return

    logger.info("=== MIGRATION: Enabling subtle zoom for botanical channels ===")
    channel_dir = app.config['CHANNEL_FOLDER']
    patched = 0
    for name in os.listdir(channel_dir):
        if not name.startswith('ch_'):
            continue
        config_path = os.path.join(channel_dir, name, 'config.json')
        if not os.path.exists(config_path):
            continue
        try:
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            fmt = cfg.get('format', {})
            if fmt.get('base') == 'botanical':
                fmt['ken_burns_effect'] = 'zoom_in'
                fmt['ken_burns_scale'] = 1.03
                cfg['updated_at'] = time.strftime('%Y-%m-%d %H:%M')
                with open(config_path, 'w') as f:
                    json.dump(cfg, f, indent=2)
                patched += 1
                logger.info(f"  Patched {name}: ken_burns -> zoom_in, scale -> 1.03")
        except Exception as e:
            logger.error(f"  Failed to patch {name}: {e}")

    with open(flag_path, 'w') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S'))
    logger.info(f"=== MIGRATION COMPLETE: {patched} botanical channel(s) patched ===")

migrate_enable_subtle_zoom_botanical()


def migrate_disable_zoom_botanical():
    """One-time migration: disable Ken Burns zoom for botanical channels to fix assembly bugs."""
    flag_path = os.path.join(app.config['CHANNEL_FOLDER'], '_migration_disable_zoom_botanical.done')
    if os.path.exists(flag_path):
        return

    logger.info("=== MIGRATION: Disabling zoom for botanical channels ===")
    channel_dir = app.config['CHANNEL_FOLDER']
    patched = 0
    for name in os.listdir(channel_dir):
        if not name.startswith('ch_'):
            continue
        config_path = os.path.join(channel_dir, name, 'config.json')
        if not os.path.exists(config_path):
            continue
        try:
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            fmt = cfg.get('format', {})
            if fmt.get('base') == 'botanical':
                fmt['ken_burns_effect'] = 'none'
                cfg['updated_at'] = time.strftime('%Y-%m-%d %H:%M')
                with open(config_path, 'w') as f:
                    json.dump(cfg, f, indent=2)
                patched += 1
                logger.info(f"  Patched {name}: ken_burns -> none (zoom disabled)")
        except Exception as e:
            logger.error(f"  Failed to patch {name}: {e}")

    with open(flag_path, 'w') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S'))
    logger.info(f"=== MIGRATION COMPLETE: {patched} botanical channel(s) — zoom disabled ===")

migrate_disable_zoom_botanical()


# ===================== BACKWARD COMPAT: Preset wrappers =====================
def get_preset(preset_id):
    """Backward-compatible: loads a channel, falling back to old preset dir."""
    channel_id = f'ch_{preset_id}' if not preset_id.startswith('ch_') else preset_id
    result = get_channel(channel_id)
    if result:
        return result
    # Fallback: try old preset directory directly
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


def transcribe_audio_assemblyai(filepath, session_id):
    """Transcribe audio using AssemblyAI for more accurate timestamps."""
    import assemblyai as aai
    aai.settings.api_key = ASSEMBLYAI_API_KEY
    emit_progress(session_id, 'transcription', 2, 'Uploading to AssemblyAI...')

    config = aai.TranscriptionConfig(speech_models=["universal-3-pro"], language_detection=True, auto_chapters=True)
    transcriber = aai.Transcriber()
    emit_progress(session_id, 'transcription', 5, 'Submitting to AssemblyAI...')

    # Use submit() + manual polling with timeout instead of blocking transcribe()
    transcript = transcriber.submit(filepath, config=config)
    transcript_id = transcript.id
    logger.info(f"AssemblyAI job submitted: {transcript_id}")

    poll_headers = {"authorization": ASSEMBLYAI_API_KEY}
    start = time.time()
    TIMEOUT = 1800  # 30 minutes
    poll_data = None

    while True:
        elapsed = int(time.time() - start)
        if elapsed > TIMEOUT:
            logger.error(f"AssemblyAI timed out after {TIMEOUT}s (job {transcript_id})")
            raise Exception(f"AssemblyAI transcription timed out after {TIMEOUT}s")

        time.sleep(5)
        emit_progress(session_id, 'transcription', min(5 + elapsed // 30, 14),
                      f'Transcribing with AssemblyAI... ({elapsed}s)')

        try:
            resp = req.get(f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
                           headers=poll_headers, timeout=30)
            poll_data = resp.json()
        except Exception as e:
            logger.warning(f"AssemblyAI poll error: {e}")
            continue

        status = poll_data.get('status', '')
        if status == 'completed':
            logger.info(f"AssemblyAI completed in {elapsed}s")
            break
        elif status == 'error':
            error_msg = poll_data.get('error', 'Unknown error')
            logger.error(f"AssemblyAI error: {error_msg}")
            raise Exception(f"AssemblyAI transcription failed: {error_msg}")

    # Fetch sentences via API
    resp = req.get(f"https://api.assemblyai.com/v2/transcript/{transcript_id}/sentences",
                   headers=poll_headers, timeout=30)
    sentences_data = resp.json()

    segments = []
    for sentence in sentences_data.get('sentences', []):
        text = sentence.get('text', '').strip()
        if text:
            segments.append({
                'start': sentence['start'] / 1000.0,  # ms to seconds
                'end': sentence['end'] / 1000.0,
                'text': text
            })

    # Extract auto chapters from transcript data
    chapters = []
    for ch in poll_data.get('chapters', []) or []:
        chapters.append({
            'start': ch['start'] / 1000.0,
            'end': ch['end'] / 1000.0,
            'headline': ch.get('headline', ''),
            'summary': ch.get('summary', ''),
            'gist': ch.get('gist', '')
        })
    if chapters:
        logger.info(f"AssemblyAI chapters: {len(chapters)} topic boundaries detected")

    logger.info(f"AssemblyAI complete: {len(segments)} segments")
    if segments:
        logger.info(f"AssemblyAI range: first={segments[0]['start']:.1f}s, last ends={segments[-1]['end']:.1f}s")

    emit_progress(session_id, 'transcription', 15, f'Transcribed {len(segments)} segments')
    return {'full_text': poll_data.get('text', ''), 'segments': segments, 'chapters': chapters}


# ===================== SCENE DETECTION =====================
def get_format_scene_rules(format_config, audio_duration):
    """Build scene rules prompt dynamically from a channel's format config dict."""
    if isinstance(format_config, str):
        # Backward compat: if a string was passed, look up the base format
        format_config = copy.deepcopy(BASE_FORMATS.get(format_config, BASE_FORMATS['pulse']))

    label = format_config.get('label', format_config.get('base', 'Custom')).upper()
    desc = format_config.get('description', '')
    intro_dur = format_config.get('intro_duration', 30)
    intro_animated = format_config.get('intro_animated', False)
    intro_min = format_config.get('intro_scene_min_duration', 2)
    intro_max = format_config.get('intro_scene_max_duration', 8)
    body_min = format_config.get('body_scene_min_duration', 5)
    body_max = format_config.get('body_scene_max_duration', 15)
    body_animated = format_config.get('body_animated', False)
    periodic_interval = format_config.get('periodic_animation_interval', 0)
    periodic_window = format_config.get('periodic_animation_window', 30)
    max_scene = format_config.get('max_scene_duration', 15)

    # Deep format has split body durations for first/second half
    body_min_first = format_config.get('body_scene_min_duration_first_half')
    body_max_first = format_config.get('body_scene_max_duration_first_half')
    body_min_second = format_config.get('body_scene_min_duration_second_half')
    body_max_second = format_config.get('body_scene_max_duration_second_half')
    has_split_body = (body_min_first is not None and body_max_first is not None
                      and body_min_second is not None and body_max_second is not None)

    intro_video_str = "ALL intro scenes must have is_video: true" if intro_animated else "ALL intro scenes must have is_video: false (no animation)"
    body_video_str = "ALL body scenes must have is_video: true" if body_animated else "ALL body scenes must have is_video: false"

    rules = f"FORMAT: {label}"
    if desc:
        rules += f" ({desc})"
    rules += "\n\n"

    rules += f"INTRO (first ~{intro_dur} seconds — flexible, end at the nearest natural scene break):\n"
    rules += f"- {intro_video_str}\n"
    rules += f"- Scene durations: {intro_min}-{intro_max} seconds\n"
    if intro_animated:
        rules += f"- NEVER exceed {intro_max} seconds for an intro scene — these get animated, keep them SHORT\n"
    rules += "- Each intro scene needs its own unique visual matching the narration\n\n"

    if has_split_body:
        mid_point = audio_duration / 2 if audio_duration else 1800
        rules += f"FIRST HALF (intro to ~{mid_point:.0f}s):\n"
        rules += f"- Create a NEW scene every {body_min_first}-{body_max_first} seconds, PRIORITISE shorter scenes to maintain momentum\n"
        rules += f"- {body_video_str}\n\n"
        rules += f"SECOND HALF (~{mid_point:.0f}s to end):\n"
        rules += f"- Create a NEW scene every {body_min_second}-{body_max_second} seconds, more relaxed pacing\n"
        rules += f"- NEVER exceed {max_scene} seconds for any single scene\n"
        rules += f"- {body_video_str}\n\n"
    else:
        rules += "BODY (after intro):\n"
        rules += f"- {body_video_str}\n"
        rules += f"- Create a NEW scene every {body_min}-{body_max} seconds\n"
        rules += f"- NEVER exceed {max_scene} seconds for any single scene\n"
        if periodic_interval > 0:
            interval_min = periodic_interval // 60
            rules += (f"- EXCEPTION: Place ONE animated scene (is_video: true, 4-8 seconds) near every "
                      f"{interval_min}-minute mark (can be anywhere within ±{periodic_window} seconds of each mark)\n")
        rules += "\n"

    rules += (
        "COVERAGE:\n"
        "- Every segment MUST belong to exactly one scene — NO segments left out\n"
        "- Scene groupings MUST be consecutive — no skipping segments\n"
        "- The FIRST scene MUST include the first segment\n"
        "- The LAST scene MUST include the last segment\n"
    )
    return rules

def get_format_subject_rules(format_config, has_subject):
    """Build subject/character rules from a channel's format config dict."""
    if not has_subject:
        return ""

    if isinstance(format_config, str):
        format_config = copy.deepcopy(BASE_FORMATS.get(format_config, BASE_FORMATS['pulse']))

    subject_mode = format_config.get('subject_mode', 'auto')
    subject_interval = format_config.get('subject_interval', 300)

    if subject_mode == 'all':
        return (
            "\n\nMAIN CHARACTER (SUBJECT REFERENCE):\n"
            "A character reference image has been uploaded.\n"
            "- The character should appear in MOST scenes — aim for at least 70% of scenes\n"
            "- Set has_subject: true when the character can naturally fit into the scene\n"
            "- Set has_subject: false ONLY for pure establishing shots (wide landscapes, cityscapes, aerial views), "
            "object-only close-ups, or abstract concept visuals where a person would look forced or unnatural\n"
            "- When has_subject is true, describe the character's SPECIFIC emotion, body language, and action:\n"
            "  GOOD: 'looking frustrated while gripping a desk, furrowed brow, clenched jaw'\n"
            "  GOOD: 'pointing at a diagram with enthusiasm, wide smile'\n"
            "  BAD: 'the main character appears' (too vague)\n"
            "- The character should feel ALIVE — never stiff or static\n"
        )
    elif subject_mode == 'botanical':
        return (
            "\n\nMAIN CHARACTER (SUBJECT REFERENCE) — BOTANICAL FORMAT:\n"
            "A character reference image has been uploaded.\n"
            "- Aim for the character to appear in roughly HALF (~50%) of all scenes\n"
            "- Set has_subject: true when the character naturally fits — e.g. presenting a plant, "
            "holding a pot, kneeling in a garden, gesturing toward foliage\n"
            "- Set has_subject: false for PURE PLANT CLOSE-UPS — detailed shots of leaves, flowers, "
            "roots, bark, seed pods, growth stages, or garden/habitat landscapes without people\n"
            "- Alternate between subject scenes and plant-only scenes for visual variety\n"
            "- When has_subject is true, describe the character's BODY LANGUAGE and ACTION only:\n"
            "  GOOD: 'kneeling beside the plant, gently lifting a leaf to examine it'\n"
            "  GOOD: 'standing in a greenhouse, holding a watering can, reaching toward a hanging basket'\n"
            "  GOOD: 'crouching in a garden bed, carefully pruning stems with shears'\n"
            "  BAD: 'smiling warmly with a joyful expression' (NO facial emotion descriptions)\n"
            "  BAD: 'the main character appears' (too vague)\n"
            "- NEVER describe facial expressions, emotions, or face details — focus ONLY on "
            "body position, hand actions, posture, and interaction with plants/environment\n"
            "- The character should feel dynamic and purposeful — actively engaging with the plants\n"
        )
    elif subject_mode == 'pov':
        return (
            "\n\nMAIN CHARACTER (POV CHARACTER — CRITICAL):\n"
            "A character reference image has been uploaded. This character is the PROTAGONIST of the video.\n"
            "- The character should appear in roughly 80% of all scenes — set has_subject: true for ~80% of scenes\n"
            "- Set has_subject: false for ~20% of scenes — pure establishing shots, object close-ups, "
            "or environmental scenes that benefit from showing the world without the character\n"
            "- CHARACTER CONSISTENCY: The character's core appearance (face, body type, hair) must stay "
            "IDENTICAL across all scenes. You may make MINOR natural tweaks (e.g. sleeves rolled up, hair "
            "slightly windswept) but NEVER change defining features like hair color, build, or facial structure.\n"
            "- ERA-APPROPRIATE ATTIRE: Detect the time period from the transcript content.\n"
            "  If medieval/historical — dress the character in period-accurate clothing (tunics, armor, robes, etc.)\n"
            "  If modern/contemporary — use normal modern attire (casual, business, sportswear, etc.)\n"
            "  If futuristic/sci-fi — use appropriate futuristic clothing\n"
            "  The attire should match the SITUATION too (a character in a lab wears a lab coat, etc.)\n"
            "- When has_subject is true, describe the character's BODY LANGUAGE and ACTION only:\n"
            "  GOOD: 'standing in a dimly lit medieval tavern, leaning forward over a wooden table, gripping the edge'\n"
            "  GOOD: 'crouching behind a wall in modern tactical gear, peering around the corner with rifle raised'\n"
            "  BAD: 'smiling warmly with a joyful expression' (NO facial emotion descriptions)\n"
            "  BAD: 'the main character appears' (too vague)\n"
            "- NEVER describe facial expressions, emotions, or face details — focus ONLY on "
            "body position, hand actions, posture, and interaction with the environment\n"
            "- The character should feel like the LEAD of a TV show — always present, always in the action\n"
        )
    elif subject_mode == 'sparse':
        interval_min = max(1, subject_interval // 60)
        return (
            "\n\nMAIN CHARACTER (SUBJECT REFERENCE):\n"
            "A character reference image has been uploaded.\n"
            f"- Use the character SPARINGLY — roughly once every {interval_min} minutes\n"
            f"- Set has_subject: true only for scenes near {interval_min}-minute intervals\n"
            "- When used, the character can be an educator pointing at something, reacting, or presenting\n"
            "- All other scenes: has_subject: false\n"
            "- When has_subject is true, describe emotion, body language, and action richly\n"
        )
    else:  # 'auto' — let GPT decide
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

def detect_scene_changes(transcript_data, session_id, has_subject=False, format_config=None, audio_duration=0, scene_instructions='', chapters=None, character_instructions=''):
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

    if format_config is None:
        format_config = copy.deepcopy(BASE_FORMATS['pulse'])
    format_rules = get_format_scene_rules(format_config, audio_duration)
    subject_note = get_format_subject_rules(format_config, has_subject)

    # Number segments globally so Gemini can reference them
    for idx, seg in enumerate(segments):
        seg['seg_id'] = idx + 1  # 1-based

    for chunk_start in range(0, len(segments), CHUNK_SIZE):
        chunk_segments = segments[chunk_start:chunk_start + CHUNK_SIZE]
        chunk_num = chunk_start // CHUNK_SIZE + 1
        total_chunks = math.ceil(len(segments) / CHUNK_SIZE)
        emit_progress(session_id, 'scene_detection', int(16 + 8 * chunk_start / len(segments)),
                     f'Analyzing section {chunk_num}/{total_chunks}...')
        
        # Format: segment ID with timestamp and text
        segments_text = "\n".join([
            f"[Seg {s['seg_id']}] ({s['start']:.1f}s-{s['end']:.1f}s): {s['text']}" 
            for s in chunk_segments
        ])
        first_seg_id = chunk_segments[0]['seg_id']
        last_seg_id = chunk_segments[-1]['seg_id']

        # FIX 3: Calculate chunk time range so Gemini knows exactly what to cover
        chunk_time_start = chunk_segments[0]['start']
        chunk_time_end = chunk_segments[-1]['end']
        chunk_duration = chunk_time_end - chunk_time_start
        is_last_chunk = chunk_num == total_chunks

        system_prompt = (
            "You are a visual director creating scene breakdowns for an illustrated video.\n\n"
            "YOUR CORE TASK:\n"
            "Read through the numbered transcript segments. Group consecutive segments into scenes — "
            "every time the narrator moves to a new idea, concept, example, or subject, that's a new scene. "
            "Each scene gets a unique visual description.\n\n"
            "HOW TO CREATE SCENES:\n"
            "1. Read each numbered segment\n"
            "2. Group consecutive segments that cover the SAME visual idea into one scene\n"
            "3. When the narrator changes topic, start a NEW scene with the next segment\n"
            "4. Tell me which segments belong to each scene using segment_start and segment_end IDs\n"
            "5. The timestamps will be derived automatically from the segments — just group them correctly\n"
            "6. Each scene's visual_description is a UNIQUE IMAGE PROMPT based ONLY on what's said in those segments\n\n"
            "GROUPING RULES:\n"
            "- A scene can be a SINGLE segment or multiple consecutive segments — whatever makes visual sense\n"
            "- Start a NEW scene whenever the visual should change, even if it's just one segment\n"
            "- Group segments together ONLY when they describe the same visual idea\n"
            "- For scenes with is_video: true (animated), MAXIMUM 8 seconds — these are animated clips with a hard length limit\n"
            "- For scenes with is_video: false (static images), maximum 15 seconds\n"
            "- If segments span longer than the limit, split into separate scenes\n"
            f"- You MUST use ALL segments from {first_seg_id} to {last_seg_id} with no gaps or skips\n"
            "- Every segment must belong to exactly one scene\n\n"
            "SCENE VISUAL DESCRIPTION RULES:\n"
            "- Each visual_description is an IMAGE PROMPT — it must paint a specific, vivid picture\n"
            "- Every scene MUST look DIFFERENT — different setting, action, composition, mood, camera angle\n"
            "- Base each scene's visual ONLY on what the narrator says in those segments\n"
            "- Use metaphorical visuals for abstract concepts: money = piles of coins/bills, debt = heavy chains, "
            "growth = rising stairs, risk = stormy skies, success = golden light\n"
            "- Describe: camera angle, lighting, mood, character emotions/poses, environment details, props\n"
            "\nCINEMATIC SHOT VARIETY (CRITICAL — avoid repetitive framing):\n"
            "- VARY camera angles across scenes: low angle (looking up), high angle (looking down), "
            "eye level, bird's eye view, Dutch angle (tilted), over-the-shoulder\n"
            "- VARY shot types: wide establishing shot (showing full environment), medium shot (waist up), "
            "close-up (face/hands/details), extreme wide (tiny figure in vast landscape), "
            "tracking shot (following movement), point-of-view shot\n"
            "- VARY character positioning: character on left third, right third, center, "
            "seen from behind, silhouetted, reflected in surface, partially obscured by foreground elements\n"
            "- Include wider shots that showcase the TOPIC's environment — don't just show close-ups of the subject\n"
            "- Use depth: foreground elements framing the shot, middle ground action, background context\n"
            "- Examples of cinematic variety:\n"
            "  'wide aerial view of a bustling trading floor, the character small among rows of screens'\n"
            "  'low-angle shot looking up at the character standing confidently on a rooftop, city skyline behind'\n"
            "  'over-the-shoulder view as the character examines a glowing map spread across a table'\n"
            "  'extreme wide shot of a vast desert with the character as a tiny silhouette walking toward ruins'\n"
            "  'bird's eye view looking straight down at the character surrounded by scattered documents'\n"
            "- NEVER use the same shot type for more than 2 consecutive scenes\n"
            f"{subject_note}\n"
            "FORBIDDEN IN VISUAL DESCRIPTIONS:\n"
            "- NO text, words, numbers, labels, signs, letters, titles, captions, or writing anywhere\n"
            "- NO dollar amounts, percentages, statistics, charts, graphs, or data\n"
            "- NO art style words: 'cartoon', 'animated', 'illustrated', 'drawn', 'realistic', '3D'\n"
            "- The art style is controlled separately — only describe scene CONTENT\n\n"
            "Return valid JSON only, no markdown:\n"
            '{"scenes": [{"scene_number": 1, "segment_start": 1, "segment_end": 3, '
            '"narration_summary": "what narrator says", "visual_description": "unique image prompt for this scene only", '
            '"has_subject": true, "is_video": false}]}\n\n'
            f"{format_rules}"
        )
        # Topic title card detection + content safety (Pregnancy Explainer channel)
        if format_config.get('topic_title_cards', False):
            system_prompt += (
                "\n\nCONTENT SAFETY (pregnancy/medical education — MANDATORY):\n"
                "All visual descriptions MUST be appropriate for ALL audiences in ALL environments.\n"
                "- NEVER describe nudity, exposed bodies, bare skin, or intimate anatomy in any visual\n"
                "- NEVER describe graphic medical procedures, surgery, blood, or internal anatomy diagrams\n"
                "- NEVER describe genitals, breasts, or any revealing imagery even in medical context\n"
                "- ALL characters MUST be female — NEVER depict male characters in any visual\n"
                "- Characters MUST always be fully clothed in modest professional or casual attire\n"
                "- For pregnancy/birth topics: show hospital exterior, waiting room, character holding baby, nursery — NEVER the procedure\n"
                "- For body changes: show character in loose clothing reacting emotionally — NEVER exposed body\n"
                "- For medical topics: use symbolic/metaphorical visuals (stethoscope, medical chart from afar, doctor's office)\n"
                "- Think 'family-friendly illustration suitable for a workplace, classroom, or public screen'\n\n"
            )
            system_prompt += (
                "TOPIC DETECTION (CRITICAL):\n"
                "This audio covers distinct topics one at a time. Topics vary in length — some are short, some are long.\n"
                "Each topic begins with the narrator clearly stating the topic name (e.g., 'Back Pain', 'Morning Sickness').\n"
                "Do NOT assume any fixed duration per topic — only use the narrator's announcement of a new topic name as the boundary.\n"
                "You MUST detect each topic boundary and include a \"topic_title\" field in the FIRST scene of each new topic.\n\n"
                "Topic detection rules:\n"
                "- The segment_start of a topic_title scene MUST be the EXACT segment where "
                "the narrator FIRST speaks the new topic name — NOT any earlier segment\n"
                "- NEVER include the end of the previous topic's speech in the topic_title scene\n"
                "- If a segment contains both the tail of one topic and the start of the next, "
                "the topic_title scene must start at the NEXT segment after that one\n"
                "- It is better to start a topic_title scene one segment LATE than one segment EARLY\n"
                "- The topic_title should be a SHORT name (1-4 words) extracted from the narrator's announcement\n"
                "- Set topic_title ONLY on the FIRST scene of each new topic\n"
                "- All subsequent scenes within the same topic should NOT have topic_title\n"
                "- The very first topic in the audio MUST have a topic_title\n\n"
                "Updated JSON format with topic_title:\n"
                '{"scenes": [{"scene_number": 1, "segment_start": 1, "segment_end": 3, '
                '"narration_summary": "...", "visual_description": "...", '
                '"has_subject": true, "is_video": false, "topic_title": "Back Pain"}]}\n'
                "Only the first scene of each topic gets topic_title. All other scenes omit it or set it to empty string.\n"
            )
        # Botanical format: plant identification + subscribe CTA detection
        if format_config.get('base') == 'botanical':
            system_prompt += (
                "\n\nPLANT IDENTIFICATION (CRITICAL FOR BOTANICAL FORMAT):\n"
                "This transcript covers specific plants, one at a time or in sections.\n"
                "1. As you read through the transcript, IDENTIFY which plant is being discussed in each section.\n"
                "   Plants are typically announced at the start of each section or mentioned repeatedly.\n"
                "2. Every visual_description MUST name the SPECIFIC plant being discussed — NEVER use generic terms like 'a plant', 'the flower', 'vegetation'.\n"
                "   GOOD: 'A lush Monstera Deliciosa with large fenestrated leaves catching dappled sunlight'\n"
                "   GOOD: 'Close-up of a Snake Plant (Sansevieria) showing its distinctive upright sword-shaped leaves with yellow margins'\n"
                "   BAD: 'A green plant in a pot' (too generic — which plant?)\n"
                "   BAD: 'Beautiful flowers in a garden' (which flowers?)\n"
                "3. When the narrator transitions to a NEW plant, all scenes in that section must show ONLY that new plant.\n"
                "   Do NOT mix plants from different sections.\n"
                "4. Include botanical details visible in the scene: leaf shape, flower color, growth habit, "
                "stem texture, root structure — whatever the narrator is discussing.\n\n"
                "SUBSCRIBE CALL-TO-ACTION DETECTION:\n"
                "If the narrator mentions subscribing, liking, hitting the bell, or any call-to-action for the channel:\n"
                "- Mark that scene with: \"is_subscribe_cta\": true\n"
                "- The visual_description for CTA scenes will be overridden — just set it to 'subscribe call to action'\n"
                "- CTA scenes should ALWAYS have has_subject: true\n\n"
                "HYPER-REALISTIC DETAIL SHOTS:\n"
                "For each unique plant discussed in the transcript, mark EXACTLY 3 pure plant close-up scenes "
                "(has_subject: false) with \"hyper_realistic\": true in the JSON.\n"
                "These should be scenes showing striking botanical detail — leaf texture, flower structure, "
                "root systems, bark patterns, or seed pods.\n"
                "Spread them out across the plant's section — do NOT cluster them together.\n"
                "All other scenes remain without this field.\n"
            )

        if scene_instructions:
            system_prompt += f"\n\nCHANNEL-SPECIFIC SCENE INSTRUCTIONS:\n{scene_instructions}\n"

        if character_instructions:
            system_prompt += (
                f"\n\nCHARACTER-SPECIFIC INSTRUCTIONS:\n"
                f"The following describes the main character in detail. Use this to inform how you describe "
                f"them in visual_description fields. These details MUST persist across ALL scenes:\n"
                f"{character_instructions}\n"
            )

        user_msg = (
            f"Total audio duration: {audio_duration:.1f} seconds\n"
            f"This is section {chunk_num} of {total_chunks}.\n"
            f"Group segments {first_seg_id} through {last_seg_id} into scenes.\n"
        )
        if is_last_chunk:
            user_msg += f"IMPORTANT: This is the LAST section — you MUST include segment {last_seg_id} in your final scene.\n"

        # Add chapter hints from AssemblyAI if available
        if chapters:
            chunk_chapters = [ch for ch in chapters
                              if ch['end'] > chunk_time_start and ch['start'] < chunk_time_end]
            if chunk_chapters:
                user_msg += "\nTOPIC BOUNDARIES (from audio analysis — use these as hints for where to start new scenes):\n"
                for ch in chunk_chapters:
                    user_msg += f"  [{ch['start']:.1f}s-{ch['end']:.1f}s] {ch['headline']}\n"
                user_msg += "These are suggestions — you can split scenes more finely within a topic, but try to align scene breaks with these topic changes.\n"

        user_msg += f"\nTranscript:\n\n{segments_text}"

        # FIX 2: Higher max_tokens (32000) + FIX 5: Lower temperature (0.4) for consistency
        response = client.chat.completions.create(
            model=scene_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ],
            temperature=format_config.get('scene_detection_temperature', 0.4), max_tokens=32000
        )
        response_text = response.choices[0].message.content.strip()
        if response_text.startswith('```'):
            response_text = response_text.split('\n', 1)[1]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
        
        # Build seg_id -> segment lookup
        seg_lookup = {s['seg_id']: s for s in segments}
        
        try:
            chunk_scenes = json.loads(response_text)
            scenes_list = chunk_scenes.get('scenes', [])
            
            for scene in scenes_list:
                scene['scene_number'] = len(all_scenes) + 1
                
                # Resolve segment references to exact timestamps
                seg_start_id = scene.get('segment_start')
                seg_end_id = scene.get('segment_end')
                
                if seg_start_id and seg_end_id and seg_start_id in seg_lookup and seg_end_id in seg_lookup:
                    scene['start_time'] = seg_lookup[seg_start_id]['start']
                    scene['end_time'] = seg_lookup[seg_end_id]['end']
                else:
                    # Fallback: Gemini may have still returned start_time/end_time
                    if 'start_time' not in scene or 'end_time' not in scene:
                        logger.warning(f"Scene {scene.get('scene_number')}: invalid segment refs {seg_start_id}-{seg_end_id}, skipping")
                        continue
                
                all_scenes.append(scene)
            
            # FIX 4: Validation + retry — check if Gemini covered all segments
            if scenes_list:
                last_seg_end = scenes_list[-1].get('segment_end', 0)
                if last_seg_end < last_seg_id:
                    uncovered_segs = [s for s in chunk_segments if s['seg_id'] > last_seg_end]
                    if uncovered_segs and len(uncovered_segs) >= 3:
                        logger.warning(f"Chunk {chunk_num}: Gemini stopped at seg {last_seg_end}, {len(uncovered_segs)} segments uncovered. Retrying.")
                        retry_text = "\n".join([f"[Seg {s['seg_id']}] ({s['start']:.1f}s-{s['end']:.1f}s): {s['text']}" for s in uncovered_segs])
                        retry_first = uncovered_segs[0]['seg_id']
                        retry_last = uncovered_segs[-1]['seg_id']
                        retry_msg = (
                            f"You need to group segments {retry_first} through {retry_last} into scenes.\n"
                            f"You MUST include segment {retry_last} in your final scene.\n\n"
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
                                seg_s = scene.get('segment_start')
                                seg_e = scene.get('segment_end')
                                if seg_s and seg_e and seg_s in seg_lookup and seg_e in seg_lookup:
                                    scene['start_time'] = seg_lookup[seg_s]['start']
                                    scene['end_time'] = seg_lookup[seg_e]['end']
                                    all_scenes.append(scene)
                            logger.info(f"Retry added {len(retry_scenes.get('scenes', []))} scenes for uncovered segments")
                        except json.JSONDecodeError as e:
                            logger.error(f"Retry JSON parse error in chunk {chunk_num}: {e}")
            
            logger.info(f"Chunk {chunk_num}/{total_chunks}: {len(scenes_list)} scenes, "
                        f"{chunk_time_start:.1f}s-{chunk_time_end:.1f}s")
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error in chunk {chunk_num}: {e}")
            # Retry the entire chunk once on JSON parse failure
            logger.info(f"Retrying chunk {chunk_num} after JSON parse error...")
            try:
                retry_resp = client.chat.completions.create(
                    model=scene_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg}
                    ],
                    temperature=0.3, max_tokens=32000
                )
                retry_text = retry_resp.choices[0].message.content.strip()
                if retry_text.startswith('```'):
                    retry_text = retry_text.split('\n', 1)[1]
                    if retry_text.endswith('```'):
                        retry_text = retry_text[:-3]
                retry_chunk = json.loads(retry_text)
                for scene in retry_chunk.get('scenes', []):
                    scene['scene_number'] = len(all_scenes) + 1
                    seg_s = scene.get('segment_start')
                    seg_e = scene.get('segment_end')
                    if seg_s and seg_e and seg_s in seg_lookup and seg_e in seg_lookup:
                        scene['start_time'] = seg_lookup[seg_s]['start']
                        scene['end_time'] = seg_lookup[seg_e]['end']
                        all_scenes.append(scene)
                logger.info(f"Chunk {chunk_num} retry succeeded: {len(retry_chunk.get('scenes', []))} scenes")
            except Exception as retry_e:
                logger.error(f"Chunk {chunk_num} retry also failed: {retry_e}")
    
    # Log output
    if all_scenes:
        last_scene_end = max(s.get('end_time', 0) for s in all_scenes)
        logger.info(f"Gemini output: {len(all_scenes)} scenes, covering 0-{last_scene_end:.1f}s (audio={audio_duration:.1f}s)")
        if last_scene_end < audio_duration - 5:
            logger.warning(f"Gemini left {audio_duration - last_scene_end:.1f}s uncovered at end of audio!")
    
    # ===================== POST-PROCESSING =====================
    # Timestamps are now derived directly from Whisper segments, so they should be accurate.
    # We just need to: enforce continuity, split oversized scenes, and fill any gaps.
    
    if all_scenes and segments:
        logger.info(f"=== POST-PROCESSING: {len(all_scenes)} scenes from segment-anchored approach ===")

        # Sort scenes by start_time to ensure correct order
        all_scenes.sort(key=lambda s: s['start_time'])

        # ANTI-DRIFT: Snap every scene boundary to the nearest transcript segment boundary.
        # This prevents scenes from drifting away from what the narrator is actually saying.
        seg_starts = sorted(set(s['start'] for s in segments))
        seg_ends = sorted(set(s['end'] for s in segments))
        seg_boundaries = sorted(set(seg_starts + seg_ends))

        def snap_to_boundary(t, boundaries):
            """Snap a timestamp to the nearest segment boundary."""
            if not boundaries:
                return t
            idx = bisect.bisect_left(boundaries, t)
            candidates = []
            if idx > 0:
                candidates.append(boundaries[idx - 1])
            if idx < len(boundaries):
                candidates.append(boundaries[idx])
            return min(candidates, key=lambda b: abs(b - t))

        snap_fixes = 0
        for scene in all_scenes:
            snapped_start = snap_to_boundary(scene['start_time'], seg_boundaries)
            snapped_end = snap_to_boundary(scene['end_time'], seg_boundaries)
            if snapped_start != scene['start_time'] or snapped_end != scene['end_time']:
                snap_fixes += 1
            # Only snap if it doesn't create a zero/negative duration
            if snapped_end > snapped_start:
                scene['start_time'] = snapped_start
                scene['end_time'] = snapped_end
        if snap_fixes:
            logger.info(f"Snapped {snap_fixes} scene boundaries to transcript segments (anti-drift)")

        # Re-sort after snapping
        all_scenes.sort(key=lambda s: s['start_time'])

        # Enforce continuity: each scene starts where the previous ends (no gaps)
        # Scenes already have segment-derived timestamps, just close small gaps
        gap_fixes = 0
        for i in range(1, len(all_scenes)):
            prev_end = all_scenes[i-1]['end_time']
            curr_start = all_scenes[i]['start_time']
            gap = curr_start - prev_end
            if gap > 0.5:
                # There's a gap — extend previous scene to cover it
                all_scenes[i-1]['end_time'] = curr_start
                gap_fixes += 1
            elif gap < -0.5:
                # Overlap — trim current scene start
                all_scenes[i]['start_time'] = prev_end
                gap_fixes += 1
        if gap_fixes:
            logger.info(f"Fixed {gap_fixes} gaps/overlaps between scenes")
        
        # Log coverage stats
        gemini_total = sum(s['end_time'] - s['start_time'] for s in all_scenes)
        max_scene_dur = max(s['end_time'] - s['start_time'] for s in all_scenes)
        logger.info(f"Scene coverage: {gemini_total:.1f}s, Audio: {audio_duration:.1f}s, Max scene: {max_scene_dur:.1f}s")
        
        # Split scenes exceeding duration limits: 8s for animated, 15s for static
        # Determine which scenes will be animated based on format rules (not Gemini's is_video which gets overridden later)
        split_scenes = []
        for scene in all_scenes:
            dur = scene['end_time'] - scene['start_time']
            # Predict if this scene will be animated based on format config
            _intro_dur = format_config.get('intro_duration', 30)
            _intro_anim = format_config.get('intro_animated', False)
            will_be_animated = _intro_anim and scene['start_time'] < _intro_dur
            _intro_max = format_config.get('intro_scene_max_duration', 8)
            _max_scene = format_config.get('max_scene_duration', 15)
            max_dur = _intro_max if will_be_animated else _max_scene
            target_dur = max(4.0, _intro_max - 2) if will_be_animated else max(6.0, _max_scene - 5)
            if dur > max_dur:
                num_parts = math.ceil(dur / target_dur)
                part_dur = dur / num_parts
                base_desc = scene['visual_description']
                scene_segs = [s for s in segments if s['end'] > scene['start_time'] and s['start'] < scene['end_time']]
                
                for p in range(num_parts):
                    part_start = scene['start_time'] + p * part_dur
                    part_end = scene['start_time'] + (p + 1) * part_dur
                    
                    part_segs = [s for s in scene_segs if s['end'] > part_start and s['start'] < part_end]
                    part_text = ' '.join(s['text'].strip() for s in part_segs) if part_segs else ''
                    
                    if p == 0:
                        part_desc = base_desc
                    elif part_text:
                        part_desc = f"A new visual perspective illustrating: {part_text[:200]}"
                    else:
                        part_desc = f"{base_desc} — from a different angle and composition"
                    
                    part_scene = {
                        'scene_number': len(split_scenes) + 1,
                        'start_time': part_start,
                        'end_time': part_end,
                        'narration_summary': scene.get('narration_summary', ''),
                        'visual_description': part_desc,
                        'has_subject': scene.get('has_subject', True),
                        'is_video': scene.get('is_video', False)
                    }
                    # Preserve topic_title on the first part only
                    if p == 0 and scene.get('topic_title'):
                        part_scene['topic_title'] = scene['topic_title']
                    split_scenes.append(part_scene)
                logger.info(f"Split {dur:.1f}s scene into {num_parts} parts")
            else:
                scene['scene_number'] = len(split_scenes) + 1
                split_scenes.append(scene)
        
        if len(split_scenes) != len(all_scenes):
            logger.info(f"Post-processing: {len(all_scenes)} -> {len(split_scenes)} scenes after splitting")
        all_scenes = split_scenes
        
        # Ensure first scene starts at 0 and last reaches audio duration
        if all_scenes:
            all_scenes[0]['start_time'] = 0.0
            
            if all_scenes[-1]['end_time'] < audio_duration:
                gap = audio_duration - all_scenes[-1]['end_time']
                if gap <= 15.0:
                    all_scenes[-1]['end_time'] = audio_duration
                    logger.info(f"Extended last scene by {gap:.1f}s to reach audio end")
                else:
                    # Create scenes from remaining transcript
                    remaining_segs = [s for s in segments if s['start'] >= all_scenes[-1]['end_time'] - 1]
                    current_start = all_scenes[-1]['end_time']
                    current_texts = []
                    for seg in remaining_segs:
                        current_texts.append(seg['text'].strip())
                        if seg['end'] - current_start >= 10.0 or seg == remaining_segs[-1]:
                            narration = ' '.join(current_texts)
                            new_scene = {
                                'scene_number': len(all_scenes) + 1,
                                'start_time': current_start,
                                'end_time': min(seg['end'], audio_duration),
                                'narration_summary': narration[:100],
                                'visual_description': f"A vivid scene illustrating: {narration[:200]}",
                                'has_subject': True,
                                'is_video': False
                            }
                            all_scenes.append(new_scene)
                            current_start = seg['end']
                            current_texts = []
                            if current_start >= audio_duration:
                                break
                    if all_scenes[-1]['end_time'] < audio_duration:
                        all_scenes[-1]['end_time'] = audio_duration
                    logger.info(f"Created additional scenes to cover remaining audio")
        
        # Final stats
        total_dur = sum(s['end_time'] - s['start_time'] for s in all_scenes)
        avg_dur = total_dur / len(all_scenes) if all_scenes else 0
        logger.info(f"Final scenes: {len(all_scenes)}, total coverage: {total_dur:.1f}s, avg duration: {avg_dur:.1f}s")
        
        for s in all_scenes:
            dur = s['end_time'] - s['start_time']
            if dur > 20:
                logger.warning(f"Scene {s['scene_number']} still oversized: {dur:.1f}s ({s['start_time']:.1f}-{s['end_time']:.1f})")
    
    emit_progress(session_id, 'scene_detection', 25, f'Detected {len(all_scenes)} scenes')
    return all_scenes


# ===================== WHISK AUTH (POOL) =====================
from whisk_pool import WhiskPool
whisk_pool = WhiskPool()

def get_whisk_key():
    """Get next available Whisk key from pool. Returns dict with token, cookie, index."""
    key = whisk_pool.get_next()
    if not key:
        logger.error("No Whisk keys configured!")
        return None
    if key.get('wait_seconds', 0) > 0:
        wait = min(key['wait_seconds'], 30)
        logger.info(f"All keys cooling down, waiting {wait:.0f}s for key {key['index']+1}/{len(whisk_pool)}")
        time.sleep(wait)
    return key

def whisk_bearer_headers_for(key):
    return {
        "authorization": f"Bearer {key['token']}",
        "content-type": "application/json",
        "origin": "https://labs.google",
        "referer": "https://labs.google/",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
    }

def whisk_cookie_headers_for(key):
    return {
        "content-type": "application/json",
        "cookie": key['cookie'],
        "origin": "https://labs.google",
        "referer": "https://labs.google/fx/tools/whisk",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
    }

# Backwards-compatible wrappers (used by functions that don't yet pass key around)
def whisk_bearer_headers():
    key = get_whisk_key()
    return whisk_bearer_headers_for(key) if key else {}

def whisk_cookie_headers():
    key = get_whisk_key()
    return whisk_cookie_headers_for(key) if key else {}


# ===================== WHISK CAPTION & UPLOAD =====================
def caption_image_whisk(image_base64, media_category, workflow_id, session_ts, key=None):
    headers = whisk_cookie_headers_for(key) if key else whisk_cookie_headers()
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

def upload_image_to_whisk(image_base64, media_category, caption, workflow_id, session_ts, key=None):
    headers = whisk_cookie_headers_for(key) if key else whisk_cookie_headers()
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
    # Pin a Whisk key for this session's entire lifecycle to prevent
    # cross-contamination between concurrent generations
    pinned_key = whisk_pool.reserve_key(session_id)
    if not pinned_key:
        logger.error(f"[{session_id}] No Whisk keys available for upload")
        return None
    if pinned_key.get('wait_seconds', 0) > 0:
        wait = min(pinned_key['wait_seconds'], 30)
        logger.info(f"[{session_id}] Waiting {wait:.0f}s for reserved key {pinned_key['index']+1}")
        time.sleep(wait)

    workflow_id = str(uuid.uuid4())
    session_ts = f";{int(time.time() * 1000)}"
    result = {"workflow_id": workflow_id, "session_ts": session_ts}

    # Store pinned key index so generation functions use the same key
    result['_pinned_key_index'] = pinned_key['index']

    # Store preset config for potential re-upload on session expiry
    result['_preset_config'] = preset_config
    result['_reupload_lock'] = threading.Lock()
    result['_upload_time'] = time.time()

    # Pass style_text through to generation — sanitize once upfront to avoid
    # per-scene safety retries from copyrighted style descriptions
    style_text = preset_config.get('style_text', '')
    if style_text:
        style_text = sanitize_style_text(style_text)
    result['style_text'] = style_text

    if preset_config.get('style_base64'):
        emit_progress(session_id, 'generation', 26, 'Captioning style image...')
        auto_caption = caption_image_whisk(preset_config['style_base64'], "MEDIA_CATEGORY_STYLE", workflow_id, session_ts, key=pinned_key)
        style_caption = (
            "ART STYLE REFERENCE ONLY. Extract ONLY the visual rendering style from this image: "
            "line weight, outline thickness, color palette, shading technique, and rendering aesthetic. "
            "DO NOT reproduce, copy, or recreate any content, characters, objects, scenes, poses, "
            "or composition from this reference image. Every generated image must depict a completely "
            "new scene as described in the prompt — only the artistic rendering style should match."
        )
        if style_text:
            style_caption += f" Style notes: {style_text}"
        # Deliberately omit auto_caption — it describes the IMAGE CONTENT (characters,
        # objects, setting) which causes Whisk to reproduce the reference image itself.
        # We only want the style, not the content.
        result['style_caption'] = style_caption

        emit_progress(session_id, 'generation', 28, 'Uploading style reference...')
        style_id = upload_image_to_whisk(preset_config['style_base64'], "MEDIA_CATEGORY_STYLE", style_caption, workflow_id, session_ts, key=pinned_key)
        if style_id == "TOKEN_EXPIRED":
            return "TOKEN_EXPIRED"
        result['style_media_id'] = style_id
    elif style_text:
        # Text-only style — no image upload needed, style applied via userInstruction
        emit_progress(session_id, 'generation', 28, f'Using text style: {style_text[:50]}...')

    if preset_config.get('subject_base64'):
        emit_progress(session_id, 'generation', 29, 'Captioning subject...')
        auto_caption = caption_image_whisk(preset_config['subject_base64'], "MEDIA_CATEGORY_SUBJECT", workflow_id, session_ts, key=pinned_key)
        subject_caption = (
            "CHARACTER IDENTITY REFERENCE ONLY. Use this character's face, body type, hair, and clothing "
            "as identity reference. Draw the character with varied poses, expressions, and gestures to match "
            "each scene. The character should feel natural and dynamic, not stiff or static. "
            "DO NOT reproduce the pose, background, setting, or composition from this reference image — "
            "every scene must be a completely new illustration showing this character in the described scenario."
        )
        # Deliberately omit auto_caption — it describes the IMAGE CONTENT (pose, background,
        # setting) which causes Whisk to reproduce the reference image instead of creating
        # new scenes with the character.
        result['subject_caption'] = subject_caption

        emit_progress(session_id, 'generation', 29, 'Uploading subject character...')
        subject_id = upload_image_to_whisk(preset_config['subject_base64'], "MEDIA_CATEGORY_SUBJECT", subject_caption, workflow_id, session_ts, key=pinned_key)
        if subject_id == "TOKEN_EXPIRED":
            return "TOKEN_EXPIRED"
        result['subject_media_id'] = subject_id

    return result


def reupload_preset_if_needed(whisk_session, session_id):
    """Re-upload preset images when Whisk media IDs have expired (404).

    Uses a lock so only one thread re-uploads; others wait and then use the fresh IDs.
    Updates whisk_session dict in-place so all concurrent scenes benefit.
    """
    lock = whisk_session.get('_reupload_lock')
    if not lock:
        return
    with lock:
        # Check if another thread already re-uploaded recently (within last 60s)
        if time.time() - whisk_session.get('_upload_time', 0) < 60:
            logger.info(f"[{session_id}] Whisk session already refreshed by another thread, skipping re-upload")
            return

        preset_config = whisk_session.get('_preset_config')
        if not preset_config:
            logger.warning(f"[{session_id}] No _preset_config in whisk_session, cannot re-upload")
            return

        logger.info(f"[{session_id}] Re-uploading preset images to Whisk (media IDs expired)...")
        fresh = upload_preset_images_to_whisk(preset_config, session_id)
        if fresh == "TOKEN_EXPIRED":
            logger.error(f"[{session_id}] Token expired during re-upload")
            return

        # Update the shared whisk_session dict in-place with fresh IDs
        for key in ['style_media_id', 'subject_media_id', 'style_caption', 'subject_caption',
                    'workflow_id', 'session_ts', '_pinned_key_index']:
            if key in fresh:
                whisk_session[key] = fresh[key]
        whisk_session['_upload_time'] = time.time()
        logger.info(f"[{session_id}] Whisk session refreshed with new media IDs")


# ===================== PROMPT REPHRASING FOR FAILED SCENES =====================
def rephrase_prompt(original_prompt):
    """Use Gemini to rephrase a visual description that triggered safety filters."""
    try:
        api_key = GEMINI_API_KEY or OPENAI_API_KEY
        if GEMINI_API_KEY:
            client = openai.OpenAI(api_key=GEMINI_API_KEY, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
            model = "gemini-2.5-flash"
        else:
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            model = "gpt-4o-mini"
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content":
                 "You rewrite image generation prompts blocked by safety filters. "
                 "AGGRESSIVELY sanitize while keeping the core visual scene.\n"
                 "REMOVE: copyrighted/trademarked references (Family Guy, Simpsons, Disney, Marvel, etc), "
                 "aggressive actions (slamming, smashing, throwing, punching), "
                 "intense emotions (furious, enraged, devastated, anguished, hysterical), "
                 "confrontational language (screaming, glaring, threatening), medical/injury references, "
                 "dollar signs or specific monetary amounts.\n"
                 "REPLACE copyrighted styles with generic descriptions: 'Family Guy style' → 'bold-outline animated comedy cartoon', "
                 "'Simpsons style' → 'yellow-skinned cartoon', 'anime style' is fine as-is.\n"
                 "REPLACE WITH calm equivalents: 'slams laptop' → 'closes laptop', "
                 "'furiously types' → 'types quickly', 'screams in anger' → 'takes a deep breath'.\n"
                 "Keep setting, composition, and lighting. Return ONLY the rewritten prompt."},
                {"role": "user", "content": f"This image prompt was blocked by a safety filter. Rewrite it to be safe:\n\n{original_prompt}"}
            ],
            max_tokens=200,
            temperature=0.3
        )
        rephrased = response.choices[0].message.content.strip()
        logger.info(f"Rephrased prompt: '{original_prompt[:60]}...' → '{rephrased[:60]}...'")
        return rephrased
    except Exception as e:
        logger.warning(f"Prompt rephrase failed: {e}")
        return original_prompt


def rewrite_prompt_for_safety(original_prompt, style_text='', has_subject=False):
    """AI-powered full rewrite of a blocked prompt — completely reimagines the scene to pass safety filters
    while preserving the topic, style, and character presence."""
    try:
        if GEMINI_API_KEY:
            client = openai.OpenAI(api_key=GEMINI_API_KEY, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
            model = "gemini-2.5-flash"
        else:
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            model = "gpt-4o-mini"

        style_directive = ""
        if style_text:
            style_directive = (
                f"\n\nMANDATORY ART STYLE: The rewritten prompt MUST begin with this style directive: "
                f"'{style_text} style.'"
            )

        subject_directive = ""
        if has_subject:
            subject_directive = (
                "\n\nCHARACTER REQUIREMENT: The scene MUST include the main character/subject. "
                "Describe their specific body movements, facial expression, and hand gestures "
                "in the rewritten prompt. The character should feel alive and expressive."
            )

        system_msg = (
            "You completely rewrite image generation prompts that were blocked by safety filters "
            "after 3 attempts. The previous rephrasing was too mild — you must FUNDAMENTALLY "
            "reimagine the scene to be safe for ALL audiences including children.\n\n"
            "RULES:\n"
            "- Keep the same general TOPIC and SETTING (e.g. space, planets, stars)\n"
            "- REPLACE all violent, destructive, death, war, weapon, explosion, injury, or graphic content "
            "with safe dramatic alternatives:\n"
            "  'star exploding and destroying planets' → 'star glowing brilliantly with swirling cosmic energy'\n"
            "  'civilization collapsing and people dying' → 'ancient ruins with dramatic storm clouds gathering'\n"
            "  'asteroid impact wiping out life' → 'massive asteroid streaking across a dramatic sky'\n"
            "  'black hole consuming everything' → 'black hole with a glowing accretion disk bending light'\n"
            "  'alien invasion destroying cities' → 'alien spacecraft hovering above a city skyline at sunset'\n"
            "- Use DRAMATIC, CINEMATIC framing — the scene should still feel intense and visually stunning\n"
            "- NO death, dying, killing, blood, corpses, skeletons, suffering, agony, screaming, or panic\n"
            "- NO weapons, missiles, bombs, guns, lasers firing at people, or military combat\n"
            "- Prefer awe, wonder, mystery, and scale over fear, destruction, and violence\n"
            "- Return ONLY the rewritten prompt, nothing else"
            f"{style_directive}"
            f"{subject_directive}"
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"This prompt was blocked 3 times by safety filters. "
                 f"Completely rewrite it to be safe:\n\n{original_prompt}"}
            ],
            max_tokens=300,
            temperature=0.4
        )
        rewritten = response.choices[0].message.content.strip()
        logger.info(f"Safety rewrite: '{original_prompt[:60]}...' → '{rewritten[:80]}...'")
        return rewritten
    except Exception as e:
        logger.warning(f"Safety rewrite failed: {e}")
        # Last-resort fallback: generic safe prompt with style
        if style_text:
            return f"{style_text} style. A dramatic panoramic view of deep space with glowing nebulae and distant stars."
        return "A dramatic panoramic view of deep space with glowing nebulae and distant stars."


def sanitize_style_text(style_text):
    """One-time sanitization of user style text to remove copyrighted references
    before any scenes are generated.  This avoids per-scene safety retries."""
    if not style_text or len(style_text) < 10:
        return style_text
    try:
        if GEMINI_API_KEY:
            client = openai.OpenAI(api_key=GEMINI_API_KEY, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
            model = "gemini-2.5-flash"
        else:
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            model = "gpt-4o-mini"

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content":
                 "You rewrite art-style descriptions to remove any copyrighted or trademarked references "
                 "while preserving the EXACT visual characteristics.\n"
                 "REPLACE named styles with their visual traits:\n"
                 "  'Family Guy style' → 'bold thick black outlines, flat solid colors, rounded heads, simplified bodies'\n"
                 "  'Simpsons style' → 'yellow skin tones, overbite features, bold outlines, flat shading'\n"
                 "  'Disney style' → 'soft rounded features, expressive eyes, clean line art, warm palette'\n"
                 "  'South Park style' → 'paper cut-out look, simple geometric shapes, flat colors'\n"
                 "If the description ALREADY uses only generic visual terms (no brand/show names), "
                 "return it unchanged.\n"
                 "Keep ALL visual details: line weight, color palette, shading style, proportions, etc.\n"
                 "Return ONLY the rewritten style description, nothing else."},
                {"role": "user", "content": style_text}
            ],
            max_tokens=200,
            temperature=0.2
        )
        sanitized = response.choices[0].message.content.strip()
        if sanitized and sanitized != style_text:
            logger.info(f"Sanitized style text: '{style_text[:80]}...' → '{sanitized[:80]}...'")
        return sanitized or style_text
    except Exception as e:
        logger.warning(f"Style text sanitization failed: {e}")
        return style_text


# ===================== WHISK IMAGE GENERATION =====================
def generate_image_whisk(prompt, output_path, session_id, scene_num, whisk_session=None, scene_has_subject=False):
    # Route to recipe if this scene will actually have media inputs.
    # A session may have subject_media_id but this scene may not use it
    # (scene_has_subject=False); if there's also no style image, recipe
    # would be called with 0 inputs → Whisk returns 400 INVALID_ARGUMENT.
    scene_will_use_subject = scene_has_subject and whisk_session and whisk_session.get('subject_media_id')
    scene_will_use_style = whisk_session and whisk_session.get('style_media_id')
    if whisk_session and (scene_will_use_style or scene_will_use_subject):
        current_prompt = prompt
        session_refreshed = False
        for attempt in range(3):
            result = generate_image_with_recipe(current_prompt, output_path, session_id, scene_num, whisk_session, scene_has_subject, safety_retry=attempt)
            if result in ("TOKEN_EXPIRED", "QUOTA_EXHAUSTED"):
                return result
            if result == "SESSION_EXPIRED" and not session_refreshed:
                # Media IDs expired — re-upload preset images and retry with same prompt
                logger.info(f"Session expired at scene {scene_num}, triggering re-upload...")
                reupload_preset_if_needed(whisk_session, session_id)
                session_refreshed = True
                result = generate_image_with_recipe(current_prompt, output_path, session_id, scene_num, whisk_session, scene_has_subject, safety_retry=attempt)
                if result in ("TOKEN_EXPIRED", "QUOTA_EXHAUSTED"):
                    return result
                if result is not None and result != "SESSION_EXPIRED":
                    return result
                # If still failing after re-upload, fall through to rephrase logic
            if result is not None and result != "SESSION_EXPIRED":
                return result
            if attempt < 2:
                logger.warning(f"Retry {attempt+2}/3 for scene {scene_num} — rephrasing prompt + simplifying captions")
                # Include style text in the rephrase so copyrighted names get
                # converted to generic visual equivalents (e.g. "Family Guy Style" →
                # "bold-outline animated comedy cartoon") rather than dropped entirely.
                style_text = whisk_session.get('style_text', '') if whisk_session else ''
                has_style_image = whisk_session.get('style_media_id') is not None if whisk_session else False
                if style_text and not has_style_image:
                    current_prompt = rephrase_prompt(f"{style_text} style. {prompt}")
                else:
                    current_prompt = rephrase_prompt(prompt)
                time.sleep(3)
        # All 3 attempts failed — dynamically rewrite the prompt for safety
        style_text = whisk_session.get('style_text', '') if whisk_session else ''
        safe_prompt = rewrite_prompt_for_safety(prompt, style_text=style_text, has_subject=scene_has_subject)
        logger.info(f"All 3 attempts failed for scene {scene_num}, trying safety-rewritten prompt: {safe_prompt[:100]}")
        emit_progress(session_id, 'generation', -1, f'Scene {scene_num} blocked 3x — trying safe rewrite...', log_type='warn')
        # First try with subject, then without — the subject reference may be
        # triggering the safety filter when combined with certain scene content
        result = generate_image_with_recipe(safe_prompt, output_path, session_id, scene_num, whisk_session, scene_has_subject, safety_retry=0)
        if result is not None and result not in ("TOKEN_EXPIRED", "QUOTA_EXHAUSTED", "SESSION_EXPIRED"):
            return result
        # If the safety rewrite failed WITH subject, try once WITHOUT subject —
        # the subject reference image itself may conflict with the scene content
        if scene_has_subject:
            logger.info(f"Safety rewrite with subject failed for scene {scene_num} — retrying WITHOUT subject")
            emit_progress(session_id, 'generation', -1, f'Scene {scene_num}: retrying without character...', log_type='warn')
            result = generate_image_with_recipe(safe_prompt, output_path, session_id, scene_num, whisk_session, False, safety_retry=0)
            if result is not None and result not in ("TOKEN_EXPIRED", "QUOTA_EXHAUSTED", "SESSION_EXPIRED"):
                return result
        logger.warning(f"Safety rewrite also failed for scene {scene_num}")
        logger.error(f"All attempts failed for scene {scene_num}, using placeholder")
        create_placeholder_image(prompt, output_path)
        return None

    # Text-only style or no style — use basic generateImage with style in prompt
    style_text = whisk_session.get('style_text', '') if whisk_session else ''
    if style_text:
        full_prompt = f"MANDATORY ART STYLE — every element must be rendered in this style: {style_text}. DO NOT use photorealistic or realistic rendering. Scene: {prompt}"
    else:
        full_prompt = prompt

    # Use reserved key if available, otherwise round-robin
    key = whisk_pool.get_reserved_key(session_id)
    if not key:
        create_placeholder_image(prompt, output_path)
        return None
    if key.get('wait_seconds', 0) > 0:
        time.sleep(min(key['wait_seconds'], 30))
    headers = whisk_bearer_headers_for(key)
    json_data = {
        "clientContext": {"workflowId": str(uuid.uuid4()), "tool": "BACKBONE", "sessionId": f";{int(time.time()*1000)}"},
        "imageModelSettings": {"imageModel": "IMAGEN_3_5", "aspectRatio": "IMAGE_ASPECT_RATIO_LANDSCAPE"},
        "mediaCategory": "MEDIA_CATEGORY_BOARD", "prompt": full_prompt, "seed": 0
    }
    try:
        response = req.post("https://aisandbox-pa.googleapis.com/v1/whisk:generateImage", json=json_data, headers=headers, timeout=120)
    except (ConnectionError, Exception) as e:
        logger.warning(f"Whisk generateImage connection error for scene {scene_num}: {e}")
        create_placeholder_image(prompt, output_path)
        return None
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


def generate_image_with_recipe(prompt, output_path, session_id, scene_num, whisk_session, scene_has_subject=False, safety_retry=0):
    # Use pinned key from whisk_session to stay on the same Google account
    # where subject/style media IDs were uploaded
    pinned_index = whisk_session.get('_pinned_key_index')
    if pinned_index is not None:
        key = whisk_pool.get_reserved_key(session_id)
        if key and key.get('wait_seconds', 0) > 0:
            time.sleep(min(key['wait_seconds'], 30))
    else:
        key = get_whisk_key()
    if not key:
        return None
    headers = {
        "authorization": f"Bearer {key['token']}",
        "content-type": "text/plain;charset=UTF-8",
        "origin": "https://labs.google",
        "referer": "https://labs.google/",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
    }

    workflow_id = whisk_session.get('workflow_id', str(uuid.uuid4()))
    session_ts = whisk_session.get('session_ts', f";{int(time.time() * 1000)}")

    # Get captions — progressively simplified on safety retries to avoid filter triggers
    subject_caption = whisk_session.get('subject_caption', 'Character identity reference — adapt pose and expression to scene, DO NOT reproduce the reference image')
    style_caption = whisk_session.get('style_caption', 'Art style reference only — match rendering style, DO NOT reproduce image content')

    if safety_retry >= 1:
        # Strip auto-generated caption text that may contribute to safety filter triggers
        for marker in [" Art style features:", " Character identity details:"]:
            subject_caption = subject_caption.split(marker)[0]
            style_caption = style_caption.split(marker)[0]
        logger.info(f"Safety retry {safety_retry}: stripped auto-captions for scene {scene_num}")
    if safety_retry >= 2:
        # Minimal captions on final attempt
        subject_caption = "Character reference"
        style_caption = "Art style reference"
        logger.info(f"Safety retry {safety_retry}: using minimal captions for scene {scene_num}")

    recipe_inputs = []
    if scene_has_subject and whisk_session.get('subject_media_id'):
        recipe_inputs.append({
            "caption": subject_caption,
            "mediaInput": {
                "mediaCategory": "MEDIA_CATEGORY_SUBJECT",
                "mediaGenerationId": whisk_session['subject_media_id']
            }
        })
    if whisk_session.get('style_media_id'):
        recipe_inputs.append({
            "caption": style_caption,
            "mediaInput": {
                "mediaCategory": "MEDIA_CATEGORY_STYLE",
                "mediaGenerationId": whisk_session['style_media_id']
            }
        })

    # Safety net: if no media inputs, runImageRecipe will fail with 400.
    # Return None so the caller falls through to text-only generateImage.
    if not recipe_inputs:
        logger.warning(f"No recipe inputs for scene {scene_num} (subject={scene_has_subject}, "
                        f"subject_media_id={bool(whisk_session.get('subject_media_id'))}, "
                        f"style_media_id={bool(whisk_session.get('style_media_id'))}) — skipping recipe")
        return None

    # Build style instruction — keep it clean and simple like browser UI
    style_text = whisk_session.get('style_text', '')
    has_style_image = whisk_session.get('style_media_id') is not None

    # The style and subject are already communicated via recipeMediaInputs captions.
    # userInstruction should just be the scene description, clean and direct.
    has_subject = scene_has_subject and whisk_session.get('subject_media_id')

    if style_text and not has_style_image:
        # Text-only style — include style hint since there's no style image
        if safety_retry >= 1:
            # On retry, the rephrased prompt already contains a sanitized style
            # description (e.g. "Family Guy Style" → "bold-outline animated comedy cartoon")
            # so don't re-add the raw copyrighted style text
            styled_prompt = prompt
        else:
            styled_prompt = f"MANDATORY ART STYLE — every element must be rendered in this style: {style_text}. DO NOT use photorealistic or realistic rendering. {prompt}"
        if has_subject:
            styled_prompt = f"KEEP the reference character's face and identity — place them in a completely new pose, expression, and setting. DO NOT copy the reference image's background or composition. {styled_prompt}"
    elif has_style_image:
        # Style image is the reference — reinforce style_text alongside image
        style_reinforcement = f" MANDATORY ART STYLE: {style_text}. DO NOT use photorealistic or realistic rendering." if style_text else ""
        subject_reinforcement = " KEEP the reference character's face and identity — vary pose, expression, and setting." if has_subject else ""
        styled_prompt = f"Generate a brand new unique scene (DO NOT replicate the style reference image).{style_reinforcement}{subject_reinforcement} {prompt}"
    else:
        styled_prompt = prompt

    json_data = {
        "clientContext": {"workflowId": workflow_id, "tool": "BACKBONE", "sessionId": session_ts},
        "imageModelSettings": {"imageModel": "GEM_PIX", "aspectRatio": "IMAGE_ASPECT_RATIO_LANDSCAPE"},
        "recipeMediaInputs": recipe_inputs,
        "seed": random.randint(100000, 999999),
        "userInstruction": styled_prompt
    }

    logger.info(f"Whisk runImageRecipe for scene {scene_num} (subject={scene_has_subject}, inputs={len(recipe_inputs)}, key={key['index']+1}/{len(whisk_pool)})")
    try:
        response = req.post("https://aisandbox-pa.googleapis.com/v1/whisk:runImageRecipe",
                            data=json.dumps(json_data), headers=headers, timeout=120)
    except (ConnectionError, Exception) as e:
        logger.warning(f"Whisk connection error for scene {scene_num}: {e}")
        return None
    logger.info(f"Whisk recipe response for scene {scene_num}: {response.status_code}")

    # Token expired — handle based on whether key is pinned
    if response.status_code == 401:
        logger.warning(f"Token expired at scene {scene_num} (key {key['index']+1})")
        whisk_pool.mark_expired(key['index'])
        if pinned_index is not None:
            # Pinned key: can't switch keys (media IDs are tied to this account)
            # Wait for cooldown then retry with same key
            emit_progress(session_id, 'generation', -1, f'Key {key["index"]+1} expired — retrying after cooldown...', log_type='error')
            time.sleep(10)
            key = whisk_pool.get_reserved_key(session_id)
            if key and key.get('wait_seconds', 0) > 0:
                time.sleep(min(key['wait_seconds'], 30))
            if key:
                headers["authorization"] = f"Bearer {key['token']}"
                try:
                    response = req.post("https://aisandbox-pa.googleapis.com/v1/whisk:runImageRecipe",
                                        data=json.dumps(json_data), headers=headers, timeout=120)
                except (ConnectionError, Exception) as e:
                    logger.warning(f"Whisk retry connection error for scene {scene_num}: {e}")
                    return None
                if response.status_code == 401:
                    return "TOKEN_EXPIRED"
            else:
                return "TOKEN_EXPIRED"
        else:
            # Unpinned: old behavior — try next key from pool
            next_key = get_whisk_key()
            if next_key:
                headers["authorization"] = f"Bearer {next_key['token']}"
                key = next_key
                emit_progress(session_id, 'generation', -1, f'Token expired — switching to key {key["index"]+1}...', log_type='error')
                try:
                    response = req.post("https://aisandbox-pa.googleapis.com/v1/whisk:runImageRecipe",
                                        data=json.dumps(json_data), headers=headers, timeout=120)
                except (ConnectionError, Exception) as e:
                    logger.warning(f"Whisk retry connection error for scene {scene_num}: {e}")
                    return None
                logger.info(f"Whisk retry with key {key['index']+1} for scene {scene_num}: {response.status_code}")
                if response.status_code == 401:
                    whisk_pool.mark_expired(key['index'])
                    return "TOKEN_EXPIRED"
            else:
                emit_progress(session_id, 'generation', -1, f'Token expired — update in Railway. Retrying in 60s...', log_type='error')
                time.sleep(60)
                whisk_pool.reload_from_env()
                return "TOKEN_EXPIRED"
    if response.status_code == 429:
        is_quota = 'QUOTA_REACHED' in (response.text or '')
        if is_quota:
            logger.warning(f"QUOTA EXHAUSTED at scene {scene_num} (key {key['index']+1}) — 0 credits remaining")
            whisk_pool.mark_quota_exhausted(key['index'])
            if whisk_pool.all_quota_exhausted():
                logger.error(f"ALL {len(whisk_pool)} Whisk accounts have 0 credits — stopping generation")
                emit_progress(session_id, 'error', 0, 'All Whisk accounts have 0 credits — generation stopped.', log_type='error')
                return "QUOTA_EXHAUSTED"
            emit_progress(session_id, 'generation', -1, f'Key {key["index"]+1} has 0 credits — other keys still available', log_type='warn')
        else:
            logger.warning(f"Rate limited at scene {scene_num} (key {key['index']+1})")
            whisk_pool.mark_cooldown(key['index'], seconds=60)
    if response.status_code == 404:
        logger.warning(f"Whisk session expired for scene {scene_num} (404 NOT_FOUND — media IDs stale)")
        return "SESSION_EXPIRED"
    if response.status_code == 400:
        logger.warning(f"Whisk UNSAFE_GENERATION for scene {scene_num}: {response.text[:300]}")
        logger.warning(f"  userInstruction was: {styled_prompt[:200]}")
        logger.warning(f"  captions: subject={subject_caption[:80]}, style={style_caption[:80]}")
        return None  # Triggers rephrase+retry, does NOT stop the generation
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
        whisk_pool.clear_quota(key['index'])  # Successful — clear any quota flag
        return {"media_id": media_id, "prompt": img_prompt, "encoded_image": encoded_image, "workflow_id": workflow_id}

    logger.warning(f"No image in recipe response for scene {scene_num}")
    return None


# ===================== ANIMATION =====================
def animate_image_whisk(image_info, script, output_path, session_id, scene_num, motion_override=None):
    # Try reserved key first, but if it's cooling down (quota exhausted / rate limited),
    # immediately switch to a different key — animation uses rawBytes so any key works
    key = whisk_pool.get_reserved_key(session_id)
    if not key:
        return False
    original_key_index = key['index']
    if key.get('wait_seconds', 0) > 0:
        # Only switch to keys not reserved by other sessions to avoid
        # cascading rate-limits across concurrent generations
        alt_key = whisk_pool.get_unreserved_key(session_id)
        if alt_key and alt_key['index'] != key['index']:
            logger.info(f"Animate scene {scene_num}: reserved key {key['index']+1} cooling down, using unreserved key {alt_key['index']+1} instead")
            key = alt_key
        else:
            time.sleep(min(key['wait_seconds'], 30))
    headers = whisk_bearer_headers_for(key)
    session_ts = f";{int(time.time() * 1000)}"
    raw_bytes = image_info.get("encoded_image", "")
    if raw_bytes and "," in raw_bytes[:100]:
        raw_bytes = raw_bytes.split(",", 1)[1]
    # Only include media_id if on the original account that uploaded it —
    # other accounts would get 404 since media IDs are account-specific
    media_id = image_info.get("media_id", "") if key['index'] == original_key_index else ""
    # Build motion instructions so Veo produces actual animation, not static output
    if motion_override:
        motion_instructions = motion_override
    else:
        motion_instructions = (
            "Animate this scene with smooth, cinematic motion. "
            "Add subtle camera movement (slow pan, gentle zoom, or drift). "
            "Animate all elements naturally: flowing hair, swaying foliage, drifting clouds, "
            "flickering lights, moving water, floating particles. "
            "Characters should have lifelike micro-movements: breathing, blinking, slight head turns, "
            "hand gestures, shifting weight. "
            "The scene must feel alive and dynamic — never static or frozen."
        )
    animate_data = {
        "clientContext": {"sessionId": session_ts, "tool": "BACKBONE", "workflowId": image_info.get("workflow_id", str(uuid.uuid4()))},
        "loopVideo": False, "modelKey": "", "modelNameType": "VEO_3_1_I2V_12STEP",
        "promptImageInput": {"mediaGenerationId": media_id,
                             "prompt": f"ORIGINAL IMAGE DESCRIPTION:\n{image_info.get('prompt', script)}", "rawBytes": raw_bytes},
        "userInstructions": motion_instructions
    }
    logger.info(f"Whisk Animate starting for scene {scene_num} (media_id={'yes' if media_id else 'skipped(diff key)'}, key={key['index']+1}/{len(whisk_pool)}, has_raw_bytes={bool(raw_bytes)})")
    response = None
    for attempt in range(5):
        try:
            response = req.post("https://aisandbox-pa.googleapis.com/v1/whisk:generateVideo", json=animate_data, headers=headers, timeout=60)
        except (ConnectionError, Exception) as e:
            logger.warning(f"Animate connection error scene {scene_num} (attempt {attempt+1}): {e}")
            time.sleep(10 * (attempt + 1))
            continue
        logger.info(f"Animate response scene {scene_num}: status={response.status_code} (attempt {attempt+1})")
        if response.status_code == 401:
            whisk_pool.mark_expired(key['index'])
            # Animation doesn't need account consistency — try another unreserved key
            next_key = whisk_pool.get_unreserved_key(session_id)
            if next_key and next_key['index'] != key['index']:
                logger.info(f"Animate scene {scene_num}: key {key['index']+1} expired, switching to unreserved key {next_key['index']+1}")
                emit_progress(session_id, 'generation', -1, f'Animation key expired — switching to key {next_key["index"]+1}...', log_type='error')
                key = next_key
                headers = whisk_bearer_headers_for(key)
                # Clear media ID — it belongs to the old account
                animate_data["promptImageInput"]["mediaGenerationId"] = ""
                continue
            return "TOKEN_EXPIRED"
        if response.status_code == 404:
            # Media ID not found on this account — clear it and retry with rawBytes only
            if animate_data["promptImageInput"].get("mediaGenerationId"):
                logger.info(f"Animate scene {scene_num}: 404 — clearing media ID and retrying with rawBytes only")
                animate_data["promptImageInput"]["mediaGenerationId"] = ""
                continue
            logger.warning(f"Animate scene {scene_num} HTTP 404 even without media ID — body: {response.text[:300]}")
            break
        if response.status_code == 429:
            is_quota = 'QUOTA_REACHED' in (response.text or '')
            if is_quota:
                logger.warning(f"QUOTA EXHAUSTED during animation scene {scene_num} (key {key['index']+1}) — 0 credits")
                whisk_pool.mark_quota_exhausted(key['index'])
                if whisk_pool.all_quota_exhausted():
                    logger.error(f"ALL {len(whisk_pool)} Whisk accounts have 0 credits — stopping")
                    emit_progress(session_id, 'error', 0, 'All Whisk accounts have 0 credits — generation stopped.', log_type='error')
                    return "QUOTA_EXHAUSTED"
            else:
                whisk_pool.mark_cooldown(key['index'], seconds=120)
            # Animation sends rawBytes so it doesn't need account consistency — but
            # only switch to keys NOT reserved by other sessions to avoid cascading
            # rate-limits across concurrent generations.
            next_key = whisk_pool.get_unreserved_key(session_id)
            if next_key and next_key['index'] != key['index']:
                logger.info(f"Animate scene {scene_num}: key {key['index']+1} {'quota exhausted' if is_quota else 'rate-limited'}, switching to unreserved key {next_key['index']+1}")
                emit_progress(session_id, 'generation', -1, f'Animation key {key["index"]+1} {"has 0 credits" if is_quota else "rate-limited"} — switching to key {next_key["index"]+1}...', log_type='warn')
                key = next_key
                headers = whisk_bearer_headers_for(key)
                # Clear media ID — it belongs to the old account
                animate_data["promptImageInput"]["mediaGenerationId"] = ""
                time.sleep(5)
            else:
                # No unreserved keys available — wait on own reserved key's cooldown
                # instead of stealing another session's key
                key = whisk_pool.get_reserved_key(session_id)
                if key:
                    wait = key.get('wait_seconds', 0)
                    if wait > 0:
                        logger.info(f"Animate scene {scene_num}: no unreserved keys, waiting {min(wait, 30):.0f}s on own key {key['index']+1}")
                        time.sleep(min(wait, 30))
                    headers = whisk_bearer_headers_for(key)
                time.sleep(10 * (attempt + 1))
            continue
        if response.status_code != 200:
            logger.warning(f"Animate scene {scene_num} HTTP error: {response.status_code} — body: {response.text[:300]}")
        break
    if response is None or response.status_code != 200:
        logger.error(f"Animate scene {scene_num} FAILED after all attempts")
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
        try:
            poll_resp = req.post("https://aisandbox-pa.googleapis.com/v1:runVideoFxSingleClipsStatusCheck",
                                 json={"operations": [{"operation": {"name": operation_name}}]}, headers=headers, timeout=30)
        except (ConnectionError, Exception) as e:
            logger.warning(f"Poll {i+1} scene {scene_num}: connection error: {e}")
            continue
        if poll_resp.status_code == 401:
            whisk_pool.mark_expired(key['index'])
            next_key = get_whisk_key()
            if next_key and next_key['index'] != key['index']:
                logger.info(f"Poll scene {scene_num}: key {key['index']+1} expired, switching to key {next_key['index']+1}")
                key = next_key
                headers = whisk_bearer_headers_for(key)
                continue
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


def create_topic_title_image(topic_name, output_path):
    """Create a 1920x1080 title card with soft pastel blush pink background and white text."""
    from PIL import ImageDraw, ImageFont
    img = Image.new('RGB', (1920, 1080), color=(232, 216, 214))
    draw = ImageDraw.Draw(img)
    text = topic_name.strip()
    # Try large font first, shrink if too wide
    font_size = 72
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    if text_w > 1600:
        font_size = 56
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except Exception:
            pass
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (1920 - text_w) // 2
    y = (1080 - text_h) // 2
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    img.save(output_path, 'PNG')
    logger.info(f"Created title card image: '{topic_name}' -> {output_path}")


def insert_topic_silences(audio_path, topic_boundaries, work_dir, prepend_silence=0.0):
    """Insert 2s of silence at each topic boundary (except the first topic).

    If prepend_silence > 0, also prepend that many seconds of silence at the
    very start of the audio (before the first topic).

    Returns (new_audio_path, insert_points) where insert_points are the original
    timestamps where silence was inserted (boundaries[1:]).
    """
    if not topic_boundaries:
        return audio_path, []

    boundaries = sorted(topic_boundaries)
    # Only insert silence BETWEEN topics, not before the first one
    insert_points = boundaries[1:]

    # If no inter-topic silences and no prepend, nothing to do
    if not insert_points and prepend_silence <= 0:
        return audio_path, []

    # Get audio sample rate
    try:
        probe_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'stream=sample_rate',
                     '-of', 'default=noprint_wrappers=1:nokey=1', audio_path]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
        sample_rate = probe_result.stdout.strip().split('\n')[0] or '44100'
    except Exception:
        sample_rate = '44100'

    # Build ffmpeg filter_complex: split audio at boundaries, interleave with silence
    filter_parts = []
    concat_labels = []

    # Prepend silence at the start if requested (for first topic audio delay)
    if prepend_silence > 0:
        filter_parts.append(f"aevalsrc=0:d={prepend_silence:.1f}:s={sample_rate}:c=stereo[presil]")
        concat_labels.append("[presil]")

    num_segments = len(insert_points) + 1

    for i in range(num_segments):
        seg_start = 0.0 if i == 0 else insert_points[i - 1]
        seg_end = insert_points[i] if i < len(insert_points) else None

        # Audio segment
        if seg_end is not None:
            filter_parts.append(f"[0:a]atrim=start={seg_start:.3f}:end={seg_end:.3f},asetpts=PTS-STARTPTS[seg{i}]")
        else:
            filter_parts.append(f"[0:a]atrim=start={seg_start:.3f},asetpts=PTS-STARTPTS[seg{i}]")
        concat_labels.append(f"[seg{i}]")

        # 2s silence after each segment (except last)
        if i < len(insert_points):
            filter_parts.append(f"aevalsrc=0:d=2:s={sample_rate}:c=stereo[sil{i}]")
            concat_labels.append(f"[sil{i}]")

    total_streams = len(concat_labels)
    filter_parts.append(f"{''.join(concat_labels)}concat=n={total_streams}:v=0:a=1[out]")
    filter_complex = ';'.join(filter_parts)

    output_path = os.path.join(work_dir, '_audio_with_topic_gaps.mp3')
    cmd = ['ffmpeg', '-y', '-i', audio_path,
           '-filter_complex', filter_complex,
           '-map', '[out]', '-c:a', 'libmp3lame', '-b:a', '192k',
           output_path]

    prepend_msg = f" (+ {prepend_silence:.1f}s prepend)" if prepend_silence > 0 else ""
    logger.info(f"Inserting {len(insert_points)} x 2s silence gaps into audio{prepend_msg} at: {[f'{t:.1f}s' for t in insert_points]}")
    subprocess.run(cmd, check=True, capture_output=True, timeout=300)
    logger.info(f"Audio with topic gaps saved to {output_path}")
    return output_path, insert_points


def _create_video_from_image_pil_blocking(image_path, video_path, duration, effect='none', scene_index=0, scale=1.10):
    """Create a video from a still image using PIL frame-by-frame rendering for smooth Ken Burns.

    Renders each frame with Pillow for sub-pixel smooth motion, then encodes with ffmpeg.
    Falls back to ffmpeg-only approach on failure.
    NOTE: This is CPU-bound. Call via create_video_from_image_pil() which runs it in a thread.
    """
    from PIL import Image as PILImage
    FPS = 25
    OUTPUT_W, OUTPUT_H = 1920, 1080
    frames = max(int(duration * FPS), FPS)

    if effect == 'rotate_pulse':
        effects_cycle = ['pan_right', 'zoom_in', 'zoom_out']
        actual_effect = effects_cycle[scene_index % 3]
    elif effect in ('zoom_in', 'zoom_out', 'pan_right'):
        actual_effect = effect
    else:
        actual_effect = 'none'

    try:
        img = PILImage.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if actual_effect != 'none':
            SCALE = scale
            scaled_w = int(OUTPUT_W * SCALE)
            scaled_h = int(OUTPUT_H * SCALE)
            img = img.resize((scaled_w, scaled_h), PILImage.LANCZOS)

            # Write individual frames to a temp dir, then encode with ffmpeg
            frame_dir = video_path.replace('.mp4', '_frames')
            os.makedirs(frame_dir, exist_ok=True)

            for f_idx in range(frames):
                t = f_idx / max(frames - 1, 1)  # 0.0 → 1.0

                if actual_effect == 'zoom_in':
                    crop_w = scaled_w - int((scaled_w - OUTPUT_W) * t)
                    crop_h = scaled_h - int((scaled_h - OUTPUT_H) * t)
                    x = (scaled_w - crop_w) // 2
                    y = (scaled_h - crop_h) // 2
                elif actual_effect == 'zoom_out':
                    crop_w = OUTPUT_W + int((scaled_w - OUTPUT_W) * t)
                    crop_h = OUTPUT_H + int((scaled_h - OUTPUT_H) * t)
                    x = (scaled_w - crop_w) // 2
                    y = (scaled_h - crop_h) // 2
                elif actual_effect == 'pan_right':
                    crop_w = OUTPUT_W
                    crop_h = OUTPUT_H
                    x = int((scaled_w - OUTPUT_W) * t)
                    y = (scaled_h - OUTPUT_H) // 2

                frame = img.crop((x, y, x + crop_w, y + crop_h))
                frame = frame.resize((OUTPUT_W, OUTPUT_H), PILImage.LANCZOS)
                frame.save(os.path.join(frame_dir, f'frame_{f_idx:05d}.png'), 'PNG')

            # Encode frames to video with ffmpeg
            cmd = ['ffmpeg', '-y', '-framerate', str(FPS),
                   '-i', os.path.join(frame_dir, 'frame_%05d.png'),
                   '-c:v', 'libx264', '-preset', 'ultrafast', '-b:v', '5M',
                   '-pix_fmt', 'yuv420p', '-r', str(FPS), video_path]
            subprocess.run(cmd, check=True, capture_output=True, timeout=300)

            # Clean up frame directory
            try:
                shutil.rmtree(frame_dir)
            except Exception:
                pass
        else:
            # No effect — static image, just use ffmpeg directly
            img_resized = img.resize((OUTPUT_W, OUTPUT_H), PILImage.LANCZOS)
            static_path = image_path.replace('.png', '_static.png')
            img_resized.save(static_path, 'PNG')
            cmd = ['ffmpeg', '-y', '-loop', '1', '-i', static_path,
                   '-c:v', 'libx264', '-preset', 'ultrafast', '-b:v', '5M', '-t', str(duration),
                   '-pix_fmt', 'yuv420p',
                   '-vf', f'scale={OUTPUT_W}:{OUTPUT_H}',
                   '-r', str(FPS), video_path]
            subprocess.run(cmd, check=True, capture_output=True, timeout=180)
            try:
                os.remove(static_path)
            except Exception:
                pass

    except Exception as e:
        logger.error(f"PIL video from image failed: {e}")
        # Fallback to simple ffmpeg static
        try:
            cmd = ['ffmpeg', '-y', '-loop', '1', '-i', image_path,
                   '-c:v', 'libx264', '-preset', 'ultrafast', '-b:v', '5M', '-t', str(duration),
                   '-pix_fmt', 'yuv420p',
                   '-vf', 'scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080',
                   '-r', '25', video_path]
            subprocess.run(cmd, check=True, capture_output=True, timeout=180)
        except Exception as e2:
            logger.error(f"Static fallback also failed: {e2}")


def create_video_from_image_pil(image_path, video_path, duration, effect='none', scene_index=0, scale=1.10):
    """Create video from image with optional Ken Burns effect.

    Runs directly in the greenlet context (not threadpool) so that gevent's
    monkey-patched subprocess can use child watchers on the default event loop.
    """
    _create_video_from_image_pil_blocking(image_path, video_path, duration, effect, scene_index, scale=scale)


def create_video_from_image(image_path, video_path, duration, effect='none', scene_index=0, scale=1.10):
    """Create a video from a still image with optional smooth Ken Burns effect.

    Routes to PIL frame-by-frame renderer for smooth sub-pixel motion,
    or ffmpeg-only for static (no effect) scenes.
    """
    create_video_from_image_pil(image_path, video_path, duration, effect=effect, scene_index=scene_index, scale=scale)


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

    # Validate concat video duration vs audio — pad with freeze frame if too short
    try:
        probe = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                                '-of', 'default=noprint_wrappers=1:nokey=1', temp_video],
                               capture_output=True, text=True, timeout=30)
        concat_dur = float(probe.stdout.strip())
        gap = audio_duration - concat_dur if audio_duration else 0
        logger.info(f"Concat video duration: {concat_dur:.1f}s (audio={audio_duration:.1f}s, diff={gap:.1f}s)")
        if gap > 2:
            logger.warning(f"VIDEO SHORTER THAN AUDIO by {gap:.1f}s — padding with freeze frame of last scene")
            padded_video = os.path.join(work_dir, 'concat_padded.mp4')
            # Use tpad filter to freeze the last frame until audio_duration
            pad_cmd = ['ffmpeg', '-y', '-i', temp_video,
                       '-vf', f'tpad=stop_mode=clone:stop_duration={gap + 1}',
                       '-c:v', 'libx264', '-preset', 'ultrafast', '-b:v', '5M',
                       '-pix_fmt', 'yuv420p', '-r', '25',
                       padded_video]
            subprocess.run(pad_cmd, check=True, capture_output=True, timeout=300)
            temp_video = padded_video
            logger.info(f"Padded video to cover audio duration")
    except Exception as e:
        logger.warning(f"Duration validation/padding failed: {e}")

    # Add audio — re-encode video to fix any timestamp issues from concat
    emit_progress(session_id, 'compositing', 94, 'Syncing with audio...')
    emit_progress(session_id, 'compositing', 97, 'Adding audio track...')
    if audio_duration:
        cmd = ['ffmpeg', '-y', '-i', temp_video, '-i', audio_path,
               '-c:v', 'libx264', '-preset', 'ultrafast', '-b:v', '5M',
               '-pix_fmt', 'yuv420p', '-r', '25', '-vsync', 'cfr',
               '-c:a', 'aac', '-b:a', '192k',
               '-map', '0:v:0', '-map', '1:a:0', '-t', str(audio_duration),
               output_path]
    else:
        cmd = ['ffmpeg', '-y', '-i', temp_video, '-i', audio_path,
               '-c:v', 'libx264', '-preset', 'ultrafast', '-b:v', '5M',
               '-pix_fmt', 'yuv420p', '-r', '25', '-vsync', 'cfr',
               '-c:a', 'aac', '-b:a', '192k',
               '-map', '0:v:0', '-map', '1:a:0', output_path]
    subprocess.run(cmd, check=True, capture_output=True, timeout=1800)
    emit_progress(session_id, 'compositing', 100, 'Final video complete!')
    return output_path


# ===================== MAIN PIPELINE =====================
def process_voiceover(filepath, session_id, channel_id=None, project_title='', device_id='unknown', character_instructions=''):
    # Track this session as active
    with _active_sessions_lock:
        active_sessions[session_id] = {
            'channel_id': channel_id,
            'project_title': project_title,
            'started_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
    try:
        start_time_ts = time.time()
        work_dir = os.path.join(app.config['OUTPUT_FOLDER'], session_id)
        os.makedirs(work_dir, exist_ok=True)

        # Load channel config (backward compat: also works with old preset IDs)
        channel_config = None
        format_config = None
        has_subject = False
        if channel_id:
            channel_config = get_channel(channel_id) if channel_id.startswith('ch_') else get_preset(channel_id)
            if channel_config:
                has_subject = channel_config.get('has_subject', False)
                format_config = channel_config.get('format')
        if not format_config:
            format_config = copy.deepcopy(BASE_FORMATS['pulse'])
        # For backward compat, keep preset_config alias for upload_preset_images_to_whisk
        preset_config = channel_config

        audio_duration = get_audio_duration(filepath)
        emit_progress(session_id, 'init', 1, f'Audio: {audio_duration/60:.1f} min')

        check_cancelled(session_id)

        if ASSEMBLYAI_API_KEY:
            try:
                transcript_data = transcribe_audio_assemblyai(filepath, session_id)
            except Exception as e:
                logger.warning(f"AssemblyAI failed ({e}), falling back to Whisper")
                emit_progress(session_id, 'transcription', 5, 'AssemblyAI failed — using Whisper...')
                transcript_data = transcribe_audio(filepath, session_id)
        else:
            transcript_data = transcribe_audio(filepath, session_id)
        check_cancelled(session_id)

        # Resolve character instructions: per-video override > channel default
        resolved_char_instructions = character_instructions or (
            channel_config.get('character_instructions', '') if channel_config else ''
        )
        scenes = detect_scene_changes(
            transcript_data, session_id, has_subject,
            format_config=format_config, audio_duration=audio_duration,
            scene_instructions=channel_config.get('scene_instructions', '') if channel_config else '',
            chapters=transcript_data.get('chapters'),
            character_instructions=resolved_char_instructions
        )

        with open(os.path.join(work_dir, 'scenes.json'), 'w') as f:
            json.dump({'transcript': transcript_data, 'scenes': scenes, 'audio_duration': audio_duration}, f, indent=2)

        # Merge short scenes
        MIN_DUR = 5.0
        merged = []
        for scene in scenes:
            if merged and (merged[-1]['end_time'] - merged[-1]['start_time']) < MIN_DUR and not scene.get('topic_title'):
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

        # ===================== TOPIC TITLE CARDS (Pregnancy Explainer) =====================
        topic_title_cards_enabled = format_config.get('topic_title_cards', False)
        if topic_title_cards_enabled:
            # Step A: Collect topic boundaries from scene detection results
            topic_boundaries = []
            topic_titles_map = {}  # original_start_time -> topic_title
            for scene in scenes:
                if scene.get('topic_title'):
                    topic_boundaries.append(scene['start_time'])
                    topic_titles_map[scene['start_time']] = scene['topic_title']

            logger.info(f"Topic title cards: detected {len(topic_boundaries)} topic boundaries")
            for t, name in sorted(topic_titles_map.items()):
                logger.info(f"  {t:.1f}s: {name}")

            if topic_boundaries:
                SILENCE_DUR = 2.0   # seconds of silence between topics
                TITLE_CARD_DUR = 3.0  # seconds the title card stays on screen
                TITLE_OVERLAP = 2.0  # seconds title card overlaps into topic audio
                FIRST_TOPIC_DELAY = 1.0  # seconds of silence before first topic audio

                title_card_prompt = (
                    "A static title card with text on a soft pastel background. "
                    "Very subtle, gentle floating movement — the text softly hovers in place "
                    "with a dreamlike quality. Keep text perfectly readable and centered."
                )

                # Step B: Insert 1s silence at start + 2s silence between topics
                insert_points = []
                gap_count = max(0, len(topic_boundaries) - 1)
                emit_progress(session_id, 'generation', 27, f'Inserting {FIRST_TOPIC_DELAY:.0f}s start delay + {gap_count} topic gaps into audio...')
                modified_audio_path, insert_points = insert_topic_silences(
                    filepath, topic_boundaries, work_dir, prepend_silence=FIRST_TOPIC_DELAY)
                filepath = modified_audio_path
                audio_duration = get_audio_duration(filepath)
                logger.info(f"Audio duration after topic gaps: {audio_duration:.1f}s (+{FIRST_TOPIC_DELAY + len(insert_points) * SILENCE_DUR:.0f}s)")

                # Step C: Mark non-first topic scenes BEFORE shifting
                non_first_topics = set()
                first_topic_scene_id = None
                for b in topic_boundaries:
                    for scene in scenes:
                        if scene.get('topic_title') and abs(scene['start_time'] - b) < 1.0:
                            if b == topic_boundaries[0]:
                                first_topic_scene_id = id(scene)
                            else:
                                non_first_topics.add(id(scene))
                            break

                # Shift scene timestamps — 1s for initial delay + 2s per silence insertion point
                for scene in scenes:
                    shift = FIRST_TOPIC_DELAY + sum(SILENCE_DUR for ip in insert_points if ip <= scene['start_time'])
                    scene['start_time'] += shift
                    scene['end_time'] += shift

                # Step D: Build title card image helper
                def make_title_card_info(topic_title):
                    safe_name = topic_title.replace(' ', '_').replace('/', '_')
                    card_img_path = os.path.join(work_dir, f'title_card_{safe_name}.png')
                    create_topic_title_image(topic_title, card_img_path)
                    with open(card_img_path, 'rb') as f:
                        card_b64 = base64.b64encode(f.read()).decode()
                    return card_img_path, card_b64

                # Step E: Insert title card scenes
                new_scenes = []

                for scene in scenes:
                    if first_topic_scene_id is not None and id(scene) == first_topic_scene_id:
                        # FIRST TOPIC: title card at [0, 3.0], 1s audio delay then narrator speaks
                        card_img, card_b64 = make_title_card_info(scene['topic_title'])
                        title_card_scene = {
                            'scene_number': 0,
                            'start_time': 0.0,
                            'end_time': TITLE_CARD_DUR,
                            'narration_summary': f'Topic: {scene["topic_title"]}',
                            'visual_description': title_card_prompt,
                            'has_subject': False,
                            'is_video': True,
                            'is_title_card': True,
                            'topic_title': scene['topic_title'],
                            'title_card_image': card_img,
                            'title_card_image_b64': f'data:image/png;base64,{card_b64}',
                        }
                        new_scenes.append(title_card_scene)
                        # Trim the first content scene to start after title card
                        scene['start_time'] = max(scene['start_time'], TITLE_CARD_DUR)
                        new_scenes.append(scene)

                    elif id(scene) in non_first_topics:
                        # NON-FIRST TOPICS: 2s silence gap precedes this scene
                        # Title card starts halfway through silence, extends into topic audio
                        # Silence gap: [scene.start_time - 2.0, scene.start_time]
                        silence_start = scene['start_time'] - SILENCE_DUR
                        card_start = silence_start + (SILENCE_DUR - TITLE_CARD_DUR + TITLE_OVERLAP)  # halfway into silence
                        card_end = card_start + TITLE_CARD_DUR  # 3.0s on screen

                        # Extend previous scene to cover the first part of the silence
                        if new_scenes:
                            new_scenes[-1]['end_time'] = card_start

                        card_img, card_b64 = make_title_card_info(scene['topic_title'])
                        title_card_scene = {
                            'scene_number': 0,
                            'start_time': card_start,
                            'end_time': card_end,
                            'narration_summary': f'Topic: {scene["topic_title"]}',
                            'visual_description': title_card_prompt,
                            'has_subject': False,
                            'is_video': True,
                            'is_title_card': True,
                            'topic_title': scene['topic_title'],
                            'title_card_image': card_img,
                            'title_card_image_b64': f'data:image/png;base64,{card_b64}',
                        }
                        new_scenes.append(title_card_scene)
                        # Trim topic's first scene to start after title card ends
                        scene['start_time'] = card_end
                        new_scenes.append(scene)
                    else:
                        new_scenes.append(scene)

                scenes = new_scenes

                # Re-sort by start_time and renumber
                scenes.sort(key=lambda s: s['start_time'])
                for i, scene in enumerate(scenes):
                    scene['scene_number'] = i + 1
                total = len(scenes)

                title_count = sum(1 for s in scenes if s.get('is_title_card'))
                logger.info(f"Topic title cards: inserted {title_count} title cards (incl. first topic), total scenes: {total}")
            else:
                logger.warning(f"Topic title cards enabled but no topics detected — skipping")

        # Animation flags — driven by format_config
        format_label = format_config.get('base', 'pulse')
        logger.info(f"Animation selection for format={format_label}, {len(scenes)} scenes:")
        logger.info(f"  First scene: {scenes[0]['start_time']:.1f}-{scenes[0]['end_time']:.1f}s ({scenes[0]['end_time']-scenes[0]['start_time']:.1f}s)")
        if len(scenes) > 1:
            logger.info(f"  Last scene: {scenes[-1]['start_time']:.1f}-{scenes[-1]['end_time']:.1f}s ({scenes[-1]['end_time']-scenes[-1]['start_time']:.1f}s)")
        # Log any scene > 30s as a critical warning — these indicate timestamp drift
        for i, scene in enumerate(scenes):
            dur = scene['end_time'] - scene['start_time']
            if dur > 30:
                logger.error(f"  CRITICAL: Scene {i} is {dur:.1f}s — timestamps are still wrong!")
        for i, scene in enumerate(scenes[:20]):  # Log first 20 scenes' timing
            logger.info(f"  Scene {i}: start={scene['start_time']:.1f}s, end={scene['end_time']:.1f}s, dur={scene['end_time']-scene['start_time']:.1f}s")

        intro_end = format_config.get('intro_duration', 30)
        intro_animated = format_config.get('intro_animated', False)
        body_animated = format_config.get('body_animated', False)
        periodic_interval = format_config.get('periodic_animation_interval', 0)
        periodic_window = format_config.get('periodic_animation_window', 30)
        subject_mode = format_config.get('subject_mode', 'auto')

        # Find last intro scene index
        last_intro_idx = -1
        for i, scene in enumerate(scenes):
            if scene['start_time'] < intro_end:
                last_intro_idx = i

        animation_pattern = format_config.get('animation_pattern', '')
        animated_count = 0
        for i, scene in enumerate(scenes):
            # Title card scenes already have is_video set — don't override
            if scene.get('is_title_card'):
                if scene.get('is_video'):
                    animated_count += 1
                continue
            if animation_pattern == 'alternating':
                # Alternate: animate even-indexed scenes (0, 2, 4, ...)
                scene['is_video'] = (i % 2 == 0)
                if scene['is_video']:
                    animated_count += 1
            elif i <= last_intro_idx:
                # Intro zone — animate if intro_animated is True
                scene['is_video'] = intro_animated
                if intro_animated:
                    animated_count += 1
            else:
                # Body zone
                if body_animated:
                    scene['is_video'] = True
                    animated_count += 1
                elif periodic_interval > 0:
                    mid = (scene['start_time'] + scene['end_time']) / 2
                    marks = range(periodic_interval, int(audio_duration), periodic_interval)
                    near_mark = any(abs(mid - m) < periodic_window for m in marks)
                    scene['is_video'] = near_mark and scene.get('is_video', False)
                    if scene['is_video']:
                        animated_count += 1
                else:
                    scene['is_video'] = False

        logger.info(f"Animation: {animated_count} scenes marked for animation (intro_end={intro_end}s, intro_animated={intro_animated})")
        if animated_count > 0:
            first_anim = next((s for s in scenes if s.get('is_video')), None)
            last_anim = next((s for s in reversed(scenes) if s.get('is_video')), None)
            if first_anim and last_anim:
                logger.info(f"Animation range: first ends at {first_anim['end_time']:.1f}s, last ends at {last_anim['end_time']:.1f}s")

        # Botanical: force first content scene to hyper-realistic animated flower
        if format_config.get('base') == 'botanical':
            first_content = next((s for s in scenes if not s.get('is_title_card')), None)
            if first_content:
                first_content['hyper_realistic'] = True
                first_content['is_video'] = True
                first_content['has_subject'] = False
                first_content['botanical_hero_flower'] = True
                # Replace description with a simple flower close-up referencing the
                # plant from the original description so it stays on-topic
                orig = first_content.get('visual_description', '')
                first_content['visual_description'] = (
                    f"Extreme close-up of a single beautiful flower in its natural habitat, "
                    f"dew-covered petals, soft golden-hour sunlight filtering through leaves. "
                    f"Context: {orig}"
                )
                logger.info(f"Botanical: scene {first_content.get('scene_number')} → hyper-realistic hero flower + Veo")
        # For subject_mode 'all', Gemini already has strong guidance to include the
        # character in ~80% of scenes.  We no longer force has_subject=True on every
        # scene because scenes that don't naturally feature a person (landscapes,
        # object close-ups, abstract visuals) trigger safety filters when the subject
        # reference image is included, leading to placeholders.

        # Upload preset images
        whisk_session = None
        if preset_config:
            emit_progress(session_id, 'generation', 28, 'Uploading style/subject to Whisk...')
            whisk_session = upload_preset_images_to_whisk(preset_config, session_id)
            if whisk_session == "TOKEN_EXPIRED":
                whisk_pool.release_key(session_id)
                emit_progress(session_id, 'error', 0, 'Token expired — update in Railway settings.')
                return None
            if whisk_session == "QUOTA_EXHAUSTED":
                whisk_pool.release_key(session_id)
                emit_progress(session_id, 'error', 0, 'All Whisk accounts have 0 credits — generation stopped.')
                return None

        # Generate visuals
        scene_videos = []
        # Determine Ken Burns effect from format config
        def get_scene_effect(fmt_config, scene_is_video):
            if scene_is_video:
                return 'none'  # Animated scenes don't need Ken Burns
            return fmt_config.get('ken_burns_effect', 'none')
        
        for i, scene in enumerate(scenes):
            scene_num = scene['scene_number']
            start, end = scene['start_time'], scene['end_time']
            duration = end - start
            is_video = scene.get('is_video', False)
            scene_has_subject = scene.get('has_subject', False) and has_subject
            scene_effect = get_scene_effect(format_config, is_video)

            logger.info(f"Scene {scene_num}/{total}: {start:.1f}-{end:.1f}s, video={is_video}, subject={scene_has_subject}, effect={scene_effect}")

        # ===================== PHASE 1: BATCH IMAGE GENERATION =====================
        IMAGE_BATCH_SIZE = 5  # Reduced from 10 to avoid Whisk rate limits
        image_results = [None] * len(scenes)
        images_generated = 0

        # Channel-specific image instructions (prepended to every scene prompt)
        image_instructions = channel_config.get('image_instructions', '') if channel_config else ''

        def generate_single_image(idx):
            scene = scenes[idx]
            scene_num = scene['scene_number']

            # Title card scenes: image already generated by PIL, return as image_info for Veo
            if scene.get('is_title_card'):
                card_img = scene.get('title_card_image')
                card_b64 = scene.get('title_card_image_b64', '')
                logger.info(f"Scene {scene_num}: title card '{scene.get('topic_title')}' — PIL image (skipping Whisk)")
                return idx, {
                    'encoded_image': card_b64,
                    'media_id': '',
                    'prompt': scene['visual_description'],
                    'workflow_id': str(uuid.uuid4()),
                }

            scene_has_subject = scene.get('has_subject', False) and has_subject
            img_path = os.path.join(work_dir, f'scene_{scene_num:04d}.png')
            prompt = scene['visual_description']

            # Subscribe CTA scenes: override prompt with subject pointing at subscribe button
            if scene.get('is_subscribe_cta'):
                prompt = (
                    "A simple, clean image with a plain white background. "
                    "The main character is standing to one side, enthusiastically pointing "
                    "with one hand toward a large, floating YouTube subscribe button. "
                    "The subscribe button is red with white text reading 'SUBSCRIBE', "
                    "hovering in the air next to the character. "
                    "The character's other hand gives a thumbs up. "
                    "Clean minimal composition, bright and inviting."
                )
                scene_has_subject = True  # Always include subject for CTA
                logger.info(f"Scene {scene_num}: Subscribe CTA — using override prompt")

            # Content safety: sanitize unsafe terms for pregnancy/medical channels
            if topic_title_cards_enabled:
                import re
                _unsafe = [
                    (r'\b(naked|nude|nudity|topless|bare.?breasted?)\b', 'modestly dressed'),
                    (r'\b(genital[s]?|vagina[l]?|penis|vulva|cervix|uterus)\b', 'medical symbol'),
                    (r'\b(exposed\s+(body|skin|belly|torso|chest|breast))', 'clothed figure'),
                    (r'\b(anatomical\s+diagram|cross.?section|internal\s+anatomy|dissection)\b', 'educational illustration'),
                    (r'\b(graphic\s+(birth|delivery|surgery|procedure))\b', 'hospital room scene'),
                    (r'\b(blood|bleeding|bloody|surgical\s+incision)\b', 'medical care'),
                    (r'\b(breast.?feeding|nursing\s+bare)\b', 'mother caring for baby'),
                    (r'\b(man|men|male|boy|father|dad|husband|grandfather)\b', 'woman'),
                ]
                for pattern, replacement in _unsafe:
                    prompt = re.sub(pattern, replacement, prompt, flags=re.IGNORECASE)
            # Hyper-realistic override for botanical scenes (replaces illustration style)
            if scene.get('hyper_realistic') and format_config.get('base') == 'botanical':
                prompt = f"Hyper-realistic macro photography, extreme botanical detail, shallow depth of field, natural lighting. {prompt}"
            elif image_instructions:
                prompt = f"{image_instructions}. {prompt}"
            if scene_has_subject and resolved_char_instructions and format_config.get('base') == 'tv-show-pov':
                prompt = f"CHARACTER CONTEXT (the character's identity comes from the uploaded reference image — use these notes for additional context only, do NOT replace the character): {resolved_char_instructions}. {prompt}"
            result = generate_image_whisk(prompt, img_path, session_id, scene_num, whisk_session, scene_has_subject)
            add_credits(1, f'Image generation — scene {scene_num}', session_id)
            return idx, result

        logger.info(f"=== PHASE 1: Batch image generation for {total} scenes (batches of {IMAGE_BATCH_SIZE}) ===")
        emit_progress(session_id, 'generation', 30, f'Generating images (0/{total})...')

        for batch_start in range(0, len(scenes), IMAGE_BATCH_SIZE):
            check_cancelled(session_id)
            batch_end = min(batch_start + IMAGE_BATCH_SIZE, len(scenes))
            batch_indices = list(range(batch_start, batch_end))

            pool = GeventPool(size=IMAGE_BATCH_SIZE)
            batch_results = pool.map(generate_single_image, batch_indices)

            fatal_error = None
            for idx, result in batch_results:
                image_results[idx] = result
                images_generated += 1
                if result == "TOKEN_EXPIRED":
                    fatal_error = 'Token expired — update in Railway settings.'
                elif result == "QUOTA_EXHAUSTED":
                    fatal_error = 'All Whisk accounts have 0 credits — generation stopped.'

            if fatal_error:
                whisk_pool.release_key(session_id)
                emit_progress(session_id, 'error', 0, fatal_error)
                return None

            progress = 30 + (25 * images_generated / total)
            emit_progress(session_id, 'generation', int(progress), f'Generating images ({images_generated}/{total})...')
            logger.info(f"Image batch {batch_start//IMAGE_BATCH_SIZE + 1}: {images_generated}/{total} complete")

            # Brief pause between batches to avoid throttling
            time.sleep(1)

        logger.info(f"=== PHASE 1 COMPLETE: {images_generated}/{total} images generated ===")
        
        # ===================== PHASE 1B: RETRY FAILED SCENES =====================
        # Find scenes that got placeholder images (result was None)
        failed_indices = [i for i, r in enumerate(image_results) if r is None and not scenes[i].get('is_title_card')]
        if failed_indices:
            logger.info(f"=== PHASE 1B: Retrying {len(failed_indices)} failed scenes (1 at a time) ===")
            emit_progress(session_id, 'generation', 56, f'Retrying 0/{len(failed_indices)} images...', log_type='warn')
            time.sleep(5)  # Wait before retrying

            retried = 0
            for retry_num, idx in enumerate(failed_indices, 1):
                emit_progress(session_id, 'generation', 56, f'Retrying {retry_num}/{len(failed_indices)} images...', log_type='warn')
                scene = scenes[idx]
                scene_num = scene['scene_number']
                scene_has_subject = scene.get('has_subject', False) and has_subject
                img_path = os.path.join(work_dir, f'scene_{scene_num:04d}.png')

                logger.info(f"Retrying scene {scene_num} with rephrased prompt...")
                rephrased = rephrase_prompt(scene['visual_description'])
                if scene.get('hyper_realistic') and format_config.get('base') == 'botanical':
                    rephrased = f"Hyper-realistic macro photography, extreme botanical detail, shallow depth of field, natural lighting. {rephrased}"
                elif image_instructions:
                    rephrased = f"{image_instructions}. {rephrased}"
                if scene_has_subject and resolved_char_instructions and format_config.get('base') == 'tv-show-pov':
                    rephrased = f"CHARACTER CONTEXT (the character's identity comes from the uploaded reference image — use these notes for additional context only, do NOT replace the character): {resolved_char_instructions}. {rephrased}"
                result = generate_image_whisk(rephrased, img_path, session_id, scene_num, whisk_session, scene_has_subject)
                
                if result == "TOKEN_EXPIRED":
                    whisk_pool.release_key(session_id)
                    emit_progress(session_id, 'error', 0, 'Token expired — update in Railway settings.')
                    return None
                if result == "QUOTA_EXHAUSTED":
                    whisk_pool.release_key(session_id)
                    emit_progress(session_id, 'error', 0, 'All Whisk accounts have 0 credits — generation stopped.')
                    return None

                if result is not None:
                    image_results[idx] = result
                    retried += 1
                    logger.info(f"Retry SUCCESS for scene {scene_num}")
                else:
                    logger.warning(f"Retry FAILED for scene {scene_num} — placeholder will be used")
                
                time.sleep(2)  # Gentle pace for retries
            
            still_failed = sum(1 for r in image_results if r is None)
            logger.info(f"=== PHASE 1B COMPLETE: {retried}/{len(failed_indices)} recovered, {still_failed} still using placeholders ===")

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

            # Botanical hero flower: use soft, minimal motion for natural footage
            scene_motion = None
            if scene.get('botanical_hero_flower'):
                scene_motion = (
                    "Gentle breeze softly moving petals, very subtle natural sway. "
                    "Slow, steady camera. Natural outdoor ambient motion only."
                )

            for anim_attempt in range(max_anim_retries):
                try:
                    animated = animate_image_whisk(image_info, scene['visual_description'], video_path, session_id, scene_num, motion_override=scene_motion)
                except Exception as e:
                    logger.error(f"Animation error scene {scene_num} (attempt {anim_attempt+1}/{max_anim_retries}): {e}")
                    animated = False
                if animated in ("TOKEN_EXPIRED", "QUOTA_EXHAUSTED"):
                    return idx, animated, None
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
                    regen_prompt = scene['visual_description']
                    if scene.get('hyper_realistic') and format_config.get('base') == 'botanical':
                        regen_prompt = f"Hyper-realistic macro photography, extreme botanical detail, shallow depth of field, natural lighting. {regen_prompt}"
                    elif image_instructions:
                        regen_prompt = f"{image_instructions}. {regen_prompt}"
                    if scene_has_subject and resolved_char_instructions and format_config.get('base') == 'tv-show-pov':
                        regen_prompt = f"CHARACTER CONTEXT (the character's identity comes from the uploaded reference image — use these notes for additional context only, do NOT replace the character): {resolved_char_instructions}. {regen_prompt}"
                    new_image = generate_image_whisk(regen_prompt, img_path, session_id, scene_num, whisk_session, scene_has_subject)
                    if new_image in ("TOKEN_EXPIRED", "QUOTA_EXHAUSTED"):
                        return idx, new_image, None
                    if new_image and isinstance(new_image, dict):
                        image_results[idx] = new_image

            return idx, animated, video_path if animated else None

        if total_anims > 0:
            check_cancelled(session_id)
            logger.info(f"=== PHASE 2: Batch animation for {total_anims} scenes (batches of {ANIM_BATCH_SIZE}) ===")
            emit_progress(session_id, 'generation', 60, f'Animating scenes (0/{total_anims})...')

            for batch_start in range(0, total_anims, ANIM_BATCH_SIZE):
                check_cancelled(session_id)
                batch_end = min(batch_start + ANIM_BATCH_SIZE, total_anims)
                batch_indices = [anim_scenes[j] for j in range(batch_start, batch_end)]

                pool = GeventPool(size=ANIM_BATCH_SIZE)
                batch_results = pool.map(animate_single_scene, batch_indices)

                fatal_error = None
                for idx, animated, video_path in batch_results:
                    anim_results[idx] = (animated, video_path)
                    anims_completed += 1
                    if animated == "TOKEN_EXPIRED":
                        fatal_error = 'Token expired — update in Railway settings.'
                    elif animated == "QUOTA_EXHAUSTED":
                        fatal_error = 'All Whisk accounts have 0 credits — generation stopped.'

                if fatal_error:
                    whisk_pool.release_key(session_id)
                    emit_progress(session_id, 'error', 0, fatal_error)
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
            check_cancelled(session_id)
            scene = scenes[i]
            scene_num = scene['scene_number']
            start, end = scene['start_time'], scene['end_time']
            duration = end - start
            is_video = scene.get('is_video', False)
            scene_effect = get_scene_effect(format_config, is_video)
            img_path = os.path.join(work_dir, f'scene_{scene_num:04d}.png')

            # Title card scenes: use pre-generated PIL image (animated by Veo if available)
            if scene.get('is_title_card'):
                title_img = scene.get('title_card_image', img_path)
                # Check if Veo animation succeeded for this title card
                if i in anim_results:
                    animated, video_path = anim_results[i]
                    if animated and video_path:
                        # Trim Veo animation to title card duration
                        trimmed = os.path.join(work_dir, f'scene_{scene_num:04d}_titlecard.mp4')
                        try:
                            cmd = ['ffmpeg', '-y', '-i', video_path, '-t', str(duration),
                                   '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', '-r', '25',
                                   '-vf', 'scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080',
                                   trimmed]
                            subprocess.run(cmd, check=True, capture_output=True, timeout=60)
                            return i, trimmed
                        except Exception:
                            pass  # Fall through to static
                # Fallback: static title card image
                vid = os.path.join(work_dir, f'scene_{scene_num:04d}_titlecard_static.mp4')
                create_video_from_image(title_img, vid, duration, effect='none', scene_index=i)
                return i, vid

            if i in anim_results:
                animated, video_path = anim_results[i]
                if animated and video_path:
                    add_credits(2, f'Animation (Veo) — scene {scene_num}', session_id)
                    trimmed = os.path.join(work_dir, f'scene_{scene_num:04d}_trimmed.mp4')
                    try:
                        # Probe actual Veo clip duration
                        probe_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                                     '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
                        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
                        veo_duration = float(probe_result.stdout.strip()) if probe_result.stdout.strip() else 0
                        logger.info(f"Veo clip scene {scene_num}: raw={veo_duration:.1f}s, needed={duration:.1f}s")
                        
                        if veo_duration >= duration - 0.1:
                            # Veo clip is long enough — just trim to exact duration
                            cmd = ['ffmpeg', '-y', '-i', video_path, '-t', str(duration),
                                   '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', '-r', '25',
                                   '-vf', 'scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080',
                                   trimmed]
                        else:
                            # Veo clip is shorter than scene — slow it down to fill the duration
                            speed_factor = veo_duration / duration if veo_duration > 0 else 1.0
                            logger.info(f"Veo clip scene {scene_num}: stretching {veo_duration:.1f}s → {duration:.1f}s (speed={speed_factor:.2f}x)")
                            cmd = ['ffmpeg', '-y', '-i', video_path, '-t', str(duration),
                                   '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', '-r', '25',
                                   '-vf', f'setpts={1/speed_factor:.4f}*PTS,scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080',
                                   trimmed]
                        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
                        return i, trimmed
                    except:
                        return i, video_path
                else:
                    logger.error(f"=== ANIMATION DIAGNOSTIC DUMP for scene {scene_num} ===")
                    logger.error(f"  Scene: {scene_num}/{total}, start={start:.1f}s, end={end:.1f}s, duration={duration:.1f}s")
                    logger.error(f"  is_video={is_video}, format={format_label}")
                    logger.error(f"  image_info type={type(image_results[i]).__name__}, media_id={image_results[i].get('media_id', 'none') if isinstance(image_results[i], dict) else 'N/A'}")
                    logger.error(f"  All animation attempts FAILED — falling back to still image")
                    logger.error(f"=== END DIAGNOSTIC ===")
                    vid = os.path.join(work_dir, f'scene_{scene_num:04d}_video.mp4')
                    create_video_from_image(img_path, vid, duration, effect=scene_effect, scene_index=i, scale=format_config.get('ken_burns_scale', 1.10))
                    return i, vid
            elif is_video and (not image_results[i] or not isinstance(image_results[i], dict)):
                logger.warning(f"Scene {scene_num} marked is_video=True but image_info invalid — falling back to still")
                vid = os.path.join(work_dir, f'scene_{scene_num:04d}_video.mp4')
                create_video_from_image(img_path, vid, duration, effect=scene_effect, scene_index=i, scale=format_config.get('ken_burns_scale', 1.10))
                return i, vid
            else:
                vid = os.path.join(work_dir, f'scene_{scene_num:04d}_video.mp4')
                create_video_from_image(img_path, vid, duration, effect=scene_effect, scene_index=i, scale=format_config.get('ken_burns_scale', 1.10))
                return i, vid

        check_cancelled(session_id)
        logger.info(f"=== PHASE 3: Parallel clip assembly for {total} scenes (batches of {CLIP_BATCH_SIZE}) ===")
        emit_progress(session_id, 'generation', 85, f'Assembling clips (0/{total})...')

        for batch_start in range(0, len(scenes), CLIP_BATCH_SIZE):
            check_cancelled(session_id)
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
        
        # Log clip durations to catch outliers
        for i, scene in enumerate(scenes):
            dur = scene['end_time'] - scene['start_time']
            if dur > 20:
                logger.warning(f"LONG CLIP: scene {scene['scene_number']} = {dur:.1f}s ({scene['start_time']:.1f}-{scene['end_time']:.1f})")
        
        # Verify no None clips
        missing = [i for i, v in enumerate(scene_videos) if v is None]
        if missing:
            logger.error(f"Missing clips for scenes: {missing}")
            # Fill with placeholder
            for i in missing:
                scene = scenes[i]
                vid = os.path.join(work_dir, f'scene_{scene["scene_number"]:04d}_placeholder.mp4')
                create_video_from_image(
                    os.path.join(work_dir, f'scene_{scene["scene_number"]:04d}.png'),
                    vid, scene['end_time'] - scene['start_time'],
                    effect='none', scene_index=i
                )
                scene_videos[i] = vid

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
            'channel_id': channel_id,
            'format_config': format_config,
            'has_subject': has_subject,
            'project_title': project_title,
            'character_instructions': resolved_char_instructions,
            # Backward compat fields for older regeneration code
            'preset_id': channel_id,
            'video_format': format_config.get('base', 'pulse')
        }
        with open(os.path.join(work_dir, 'generation_state.json'), 'w') as f:
            json.dump(generation_state, f, indent=2)

        with _active_sessions_lock:
            active_sessions.pop(session_id, None)
        whisk_pool.release_key(session_id)
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
            'device_id': device_id,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'filename': os.path.basename(filepath),
            'audio_duration': round(audio_duration, 1),
            'scene_count': total,
            'channel_id': channel_id or 'none',
            'channel_name': channel_config.get('name', 'Unknown') if channel_config else 'None',
            'format_label': format_config.get('label', format_config.get('base', 'pulse')),
            'credits_used': session_credits,
            'processing_time': processing_time,
            'status': 'complete',
            'video_url': f'/download/{session_id}/{output_filename}',
            'project_title': project_title,
            # Backward compat
            'preset_id': channel_id or 'none',
            'preset_name': channel_config.get('name', 'Unknown') if channel_config else 'None',
            'animate_intro': format_config.get('base', 'pulse')
        })
        
        return output_path
    except GenerationCancelled:
        with _active_sessions_lock:
            active_sessions.pop(session_id, None)
        whisk_pool.release_key(session_id)
        clear_cancel(session_id)
        logger.info(f"[{session_id}] Generation cancelled by user")
        emit_progress(session_id, 'cancelled', 0, 'Generation cancelled')
        log_generation({
            'session_id': session_id,
            'device_id': device_id,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'filename': os.path.basename(filepath),
            'channel_id': channel_id or 'none',
            'channel_name': channel_config.get('name', 'Unknown') if channel_config else 'None',
            'status': 'cancelled',
            'error': 'Cancelled by user',
            'project_title': project_title,
            'preset_id': channel_id or 'none'
        })
    except Exception as e:
        with _active_sessions_lock:
            active_sessions.pop(session_id, None)
        whisk_pool.release_key(session_id)
        clear_cancel(session_id)
        logger.error(f"Pipeline error: {e}", exc_info=True)
        emit_progress(session_id, 'error', 0, f'Error: {str(e)}')

        # Log failed generation
        log_generation({
            'session_id': session_id,
            'device_id': device_id,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'filename': os.path.basename(filepath),
            'channel_id': channel_id or 'none',
            'channel_name': channel_config.get('name', 'Unknown') if channel_config else 'None',
            'status': 'error',
            'error': str(e),
            'project_title': project_title,
            # Backward compat
            'preset_id': channel_id or 'none'
        })
        raise


# ===================== ATOMIC FILE OPERATIONS =====================
def _atomic_json_write(filepath, data):
    """Write JSON atomically using temp file + rename to prevent corruption."""
    dir_name = os.path.dirname(filepath)
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, filepath)
    except Exception as e:
        logger.error(f"Atomic write failed for {filepath}: {e}")
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

_history_lock = threading.Lock()
_credits_lock = threading.Lock()


# ===================== GENERATION HISTORY =====================
HISTORY_FILE = os.path.join(PERSISTENT_DIR, 'generation_history.json')

def log_generation(entry):
    with _history_lock:
        history = load_history()
        history.insert(0, entry)  # newest first
        _atomic_json_write(HISTORY_FILE, history)

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
            # Auto-prune entries older than 7 days
            cutoff = time.time() - 7 * 24 * 60 * 60
            pruned = []
            for entry in history:
                try:
                    ts = time.mktime(time.strptime(entry.get('timestamp', ''), '%Y-%m-%d %H:%M:%S'))
                    if ts >= cutoff:
                        pruned.append(entry)
                except (ValueError, TypeError):
                    pruned.append(entry)  # keep entries with unparseable timestamps
            if len(pruned) < len(history):
                _atomic_json_write(HISTORY_FILE, pruned)
            return pruned
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


@app.route('/admin/history/clear', methods=['POST'])
def admin_history_clear():
    auth = request.cookies.get('admin_auth')
    if auth != ADMIN_PASSWORD:
        return jsonify({'error': 'Unauthorized'}), 401
    with _history_lock:
        _atomic_json_write(HISTORY_FILE, [])
    return jsonify({'ok': True, 'message': 'History cleared'})


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
    _atomic_json_write(CREDITS_FILE, data)

def add_credits(amount, description, session_id=''):
    with _credits_lock:
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


@app.route('/admin/whisk-pool')
def admin_whisk_pool():
    auth = request.cookies.get('admin_auth')
    if auth != ADMIN_PASSWORD:
        return jsonify({'error': 'Unauthorized'}), 401
    return jsonify(whisk_pool.status())

@app.route('/admin/whisk-pool/reload', methods=['POST'])
def admin_whisk_pool_reload():
    auth = request.cookies.get('admin_auth')
    if auth != ADMIN_PASSWORD:
        return jsonify({'error': 'Unauthorized'}), 401
    whisk_pool.reload_from_env()
    return jsonify({'ok': True, 'pool': whisk_pool.status()})


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
    channel_id = request.form.get('channel_id', '') or request.form.get('preset_id', '')
    project_title = request.form.get('project_title', '').strip()
    character_instructions = request.form.get('character_instructions', '').strip()
    session_id = str(uuid.uuid4())[:12]
    # Build a device fingerprint from IP + User-Agent
    raw_ip = request.headers.get('X-Forwarded-For', request.remote_addr or 'unknown')
    client_ip = raw_ip.split(',')[0].strip()
    user_agent = request.headers.get('User-Agent', 'unknown')
    device_id = hashlib.sha256(f'{client_ip}|{user_agent}'.encode()).hexdigest()[:12]
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'{session_id}_{filename}')
    file.save(filepath)
    socketio.start_background_task(process_voiceover, filepath, session_id, channel_id, project_title, device_id, character_instructions)
    return jsonify({'session_id': session_id, 'message': 'Processing started', 'filename': filename})

@app.route('/api/cancel-generation', methods=['POST'])
def cancel_generation():
    """Cancel an in-progress generation by session ID."""
    data = request.get_json(silent=True) or {}
    session_id = data.get('session_id', '')
    if not session_id:
        return jsonify({'error': 'Missing session_id'}), 400
    with _active_sessions_lock:
        if session_id not in active_sessions:
            return jsonify({'error': 'Session not found or already finished'}), 404
    request_cancel(session_id)
    logger.info(f"[{session_id}] Cancellation requested by user")
    return jsonify({'ok': True, 'message': 'Cancellation requested'})

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
    # Backward compat: support both channel_id and preset_id
    channel_id = state.get('channel_id') or state.get('preset_id')
    format_config = state.get('format_config')
    if not format_config:
        video_format = state.get('video_format', 'pulse')
        format_config = copy.deepcopy(BASE_FORMATS.get(video_format, BASE_FORMATS['pulse']))
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
    channel_config = get_channel(channel_id) if channel_id else (get_preset(channel_id) if channel_id else None)
    whisk_session = None
    if channel_config:
        whisk_session = upload_preset_images_to_whisk(channel_config, session_id)
        if whisk_session == "TOKEN_EXPIRED":
            whisk_pool.release_key(session_id)
            return jsonify({'error': 'Whisk token expired — update in Railway settings'}), 401
        if whisk_session == "QUOTA_EXHAUSTED":
            whisk_pool.release_key(session_id)
            return jsonify({'error': 'All Whisk accounts have 0 credits'}), 429

    scene_has_subject = scene.get('has_subject', False) and has_subject
    img_path = os.path.join(work_dir, f'scene_{scene_num:04d}.png')

    # Regenerate image
    logger.info(f"Regenerating scene {scene_num} for session {session_id}")
    character_instructions = state.get('character_instructions', '')
    video_format = state.get('video_format', 'pulse')
    image_instructions = format_config.get('image_instructions', '')
    if scene.get('hyper_realistic') and format_config.get('base') == 'botanical':
        prompt = f"Hyper-realistic macro photography, extreme botanical detail, shallow depth of field, natural lighting. {prompt}"
    elif image_instructions:
        prompt = f"{image_instructions}. {prompt}"
    if scene_has_subject and character_instructions and video_format == 'tv-show-pov':
        prompt = f"CHARACTER CONTEXT (the character's identity comes from the uploaded reference image — use these notes for additional context only, do NOT replace the character): {character_instructions}. {prompt}"
    socketio.emit('regen_progress', {'session_id': session_id, 'progress': 15, 'message': 'Generating image...'})
    result = generate_image_whisk(prompt, img_path, session_id, scene_num, whisk_session, scene_has_subject)

    if result == "TOKEN_EXPIRED":
        whisk_pool.release_key(session_id)
        return jsonify({'error': 'Whisk token expired'}), 401
    if result == "QUOTA_EXHAUSTED":
        whisk_pool.release_key(session_id)
        return jsonify({'error': 'All Whisk accounts have 0 credits'}), 429
    if not result:
        whisk_pool.release_key(session_id)
        return jsonify({'error': 'Image generation failed'}), 500

    socketio.emit('regen_progress', {'session_id': session_id, 'progress': 70, 'message': 'Rebuilding clip...'})
    # Rebuild clip for this scene
    is_video = scene.get('is_video', False)
    duration = scene['end_time'] - scene['start_time']
    scene_effect = 'none'
    if not is_video:
        scene_effect = format_config.get('ken_burns_effect', 'none')
    
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
            create_video_from_image(img_path, vid_path, duration, effect=scene_effect, scene_index=scene_idx, scale=format_config.get('ken_burns_scale', 1.10))
    else:
        create_video_from_image(img_path, vid_path, duration, effect=scene_effect, scene_index=scene_idx, scale=format_config.get('ken_burns_scale', 1.10))

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
    socketio.emit('regen_progress', {'session_id': session_id, 'progress': 85, 'message': 'Recomposing video...'})
    logger.info(f"Recomposing video after scene {scene_num} regeneration")
    output_path = os.path.join(work_dir, output_filename)
    compose_final_video(scene_videos, audio_path, output_path, session_id, audio_duration=audio_duration)

    socketio.emit('regen_progress', {'session_id': session_id, 'progress': 100, 'message': 'Complete'})
    logger.info(f"Scene {scene_num} regenerated and video recomposed")
    whisk_pool.release_key(session_id)
    return jsonify({
        'success': True,
        'scene_number': scene_num,
        'video_url': f'/download/{session_id}/{output_filename}',
        'image_url': f'/scene-image/{session_id}/{scene_num}',
        'scenes': scenes
    })

@app.route('/api/regenerate-batch', methods=['POST'])
def regenerate_batch():
    """Regenerate multiple scenes at once, recompose video only once at the end."""
    data = request.json
    session_id = data.get('session_id')
    scene_numbers = data.get('scene_numbers', [])
    
    if not session_id or not scene_numbers:
        return jsonify({'error': 'Missing session_id or scene_numbers'}), 400
    
    work_dir = os.path.join(app.config['OUTPUT_FOLDER'], session_id)
    state_path = os.path.join(work_dir, 'generation_state.json')
    
    if not os.path.exists(state_path):
        return jsonify({'error': 'Generation state not found'}), 404
    
    with open(state_path) as f:
        state = json.load(f)
    
    scenes = state['scenes']
    scene_videos = state['scene_videos']
    audio_path = state['audio_path']
    audio_duration = state['audio_duration']
    output_filename = state['output_filename']
    # Backward compat: support both channel_id and preset_id
    channel_id = state.get('channel_id') or state.get('preset_id')
    format_config = state.get('format_config')
    if not format_config:
        video_format = state.get('video_format', 'pulse')
        format_config = copy.deepcopy(BASE_FORMATS.get(video_format, BASE_FORMATS['pulse']))
    has_subject = state.get('has_subject', False)

    # Setup Whisk session once for the whole batch
    channel_config = get_channel(channel_id) if channel_id else (get_preset(channel_id) if channel_id else None)
    whisk_session = None
    if channel_config:
        whisk_session = upload_preset_images_to_whisk(channel_config, session_id)
        if whisk_session == "TOKEN_EXPIRED":
            whisk_pool.release_key(session_id)
            return jsonify({'error': 'Whisk token expired'}), 401
        if whisk_session == "QUOTA_EXHAUSTED":
            whisk_pool.release_key(session_id)
            return jsonify({'error': 'All Whisk accounts have 0 credits'}), 429

    results = []
    total = len(scene_numbers)

    # Phase 1: Regenerate all images
    for si, scene_num in enumerate(scene_numbers):
        pct = 5 + int(si / total * 80)
        socketio.emit('regen_progress', {'session_id': session_id, 'progress': pct, 'message': f'Generating scene {si + 1} of {total}...'})

        scene_idx = None
        scene = None
        for i, s in enumerate(scenes):
            if s['scene_number'] == scene_num:
                scene_idx = i
                scene = s
                break

        if scene is None:
            results.append({'scene_number': scene_num, 'success': False, 'error': 'Not found'})
            continue

        prompt = scene['visual_description']
        scene_has_subject = scene.get('has_subject', False) and has_subject
        img_path = os.path.join(work_dir, f'scene_{scene_num:04d}.png')

        character_instructions = state.get('character_instructions', '')
        batch_video_format = state.get('video_format', 'pulse')
        batch_image_instructions = format_config.get('image_instructions', '')
        if scene.get('hyper_realistic') and format_config.get('base') == 'botanical':
            prompt = f"Hyper-realistic macro photography, extreme botanical detail, shallow depth of field, natural lighting. {prompt}"
        elif batch_image_instructions:
            prompt = f"{batch_image_instructions}. {prompt}"
        if scene_has_subject and character_instructions and batch_video_format == 'tv-show-pov':
            prompt = f"CHARACTER CONTEXT (the character's identity comes from the uploaded reference image — use these notes for additional context only, do NOT replace the character): {character_instructions}. {prompt}"

        logger.info(f"Batch regen: generating image for scene {scene_num}")
        result = generate_image_whisk(prompt, img_path, session_id, scene_num, whisk_session, scene_has_subject)

        if result == "TOKEN_EXPIRED":
            whisk_pool.release_key(session_id)
            return jsonify({'error': 'Whisk token expired', 'completed': results}), 401
        if result == "QUOTA_EXHAUSTED":
            whisk_pool.release_key(session_id)
            return jsonify({'error': 'All Whisk accounts have 0 credits', 'completed': results}), 429
        if not result:
            results.append({'scene_number': scene_num, 'success': False, 'error': 'Image gen failed'})
            continue

        # Rebuild clip
        is_video = scene.get('is_video', False)
        duration = scene['end_time'] - scene['start_time']
        scene_effect = 'none'
        if not is_video:
            scene_effect = format_config.get('ken_burns_effect', 'none')
        
        vid_path = os.path.join(work_dir, f'scene_{scene_num:04d}_video.mp4')
        
        if is_video and isinstance(result, dict):
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
                create_video_from_image(img_path, vid_path, duration, effect=scene_effect, scene_index=scene_idx, scale=format_config.get('ken_burns_scale', 1.10))
        else:
            create_video_from_image(img_path, vid_path, duration, effect=scene_effect, scene_index=scene_idx, scale=format_config.get('ken_burns_scale', 1.10))

        scene_videos[scene_idx] = vid_path
        results.append({'scene_number': scene_num, 'success': True})
        logger.info(f"Batch regen: scene {scene_num} done ({len(results)}/{len(scene_numbers)})")
    
    # Phase 2: Save state and recompose video ONCE
    state['scene_videos'] = scene_videos
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)
    
    socketio.emit('regen_progress', {'session_id': session_id, 'progress': 88, 'message': 'Recomposing video...'})
    logger.info(f"Batch regen: recomposing video after {len(scene_numbers)} scene regenerations")
    output_path = os.path.join(work_dir, output_filename)
    compose_final_video(scene_videos, audio_path, output_path, session_id, audio_duration=audio_duration)

    successful = sum(1 for r in results if r.get('success'))
    socketio.emit('regen_progress', {'session_id': session_id, 'progress': 100, 'message': 'Complete'})
    logger.info(f"Batch regen complete: {successful}/{len(scene_numbers)} scenes regenerated")
    whisk_pool.release_key(session_id)
    return jsonify({
        'success': True,
        'results': results,
        'video_url': f'/download/{session_id}/{output_filename}',
        'scenes': scenes
    })

# ===================== CHANNEL API ROUTES =====================
@app.route('/api/channels', methods=['GET'])
def list_channels():
    return jsonify(get_all_channels())

@app.route('/api/channels', methods=['POST'])
def create_channel_route():
    name = request.form.get('name', 'Untitled')
    base_format = request.form.get('base_format', 'pulse')
    style_text = request.form.get('style_text', '').strip()
    scene_instructions = request.form.get('scene_instructions', '').strip()
    image_instructions = request.form.get('image_instructions', '').strip()
    tags = request.form.get('tags', '').strip()
    tag_colors_raw = request.form.get('tag_colors', '{}')
    try:
        tag_colors = json.loads(tag_colors_raw)
    except:
        tag_colors = {}
    style_b64 = None
    if 'style' in request.files:
        sf = request.files['style']
        if sf.filename and allowed_image(sf.filename):
            style_b64 = base64.b64encode(sf.read()).decode('utf-8')
    subject_b64 = None
    if 'subject' in request.files:
        sf = request.files['subject']
        if sf.filename and allowed_image(sf.filename):
            subject_b64 = base64.b64encode(sf.read()).decode('utf-8')
    logo_b64 = None
    if 'logo' in request.files:
        lf = request.files['logo']
        if lf.filename and allowed_image(lf.filename):
            logo_b64 = base64.b64encode(lf.read()).decode('utf-8')
    channel_id = save_channel(name, base_format=base_format, style_data=style_b64,
                              subject_data=subject_b64, logo_data=logo_b64,
                              style_text=style_text, tags=tags, tag_colors=tag_colors,
                              scene_instructions=scene_instructions, image_instructions=image_instructions)
    return jsonify({'id': channel_id, 'message': 'Channel created'})

@app.route('/api/channels/<channel_id>', methods=['GET'])
def get_single_channel(channel_id):
    channel_dir = os.path.join(app.config['CHANNEL_FOLDER'], channel_id)
    config_path = os.path.join(channel_dir, 'config.json')
    if not os.path.exists(config_path):
        return jsonify({'error': 'Not found'}), 404
    with open(config_path) as f:
        config = json.load(f)
    config['id'] = channel_id
    config['has_logo'] = os.path.exists(os.path.join(channel_dir, 'logo.png'))
    return jsonify(config)

@app.route('/api/channels/<channel_id>', methods=['PUT'])
def update_channel_route(channel_id):
    fields = {}
    name = request.form.get('name')
    if name is not None:
        fields['name'] = name
    style_text = request.form.get('style_text')
    if style_text is not None:
        fields['style_text'] = style_text
    scene_instructions = request.form.get('scene_instructions')
    if scene_instructions is not None:
        fields['scene_instructions'] = scene_instructions
    image_instructions = request.form.get('image_instructions')
    if image_instructions is not None:
        fields['image_instructions'] = image_instructions
    character_instructions = request.form.get('character_instructions')
    if character_instructions is not None:
        fields['character_instructions'] = character_instructions
    tags = request.form.get('tags')
    if tags is not None:
        fields['tags'] = [t.strip() for t in tags.split(',') if t.strip()] if tags else []
    tag_colors = request.form.get('tag_colors')
    if tag_colors:
        try:
            fields['tag_colors'] = json.loads(tag_colors)
        except:
            pass
    if 'style' in request.files:
        sf = request.files['style']
        if sf.filename and allowed_image(sf.filename):
            fields['style_data'] = base64.b64encode(sf.read()).decode('utf-8')
    if 'subject' in request.files:
        sf = request.files['subject']
        if sf.filename and allowed_image(sf.filename):
            fields['subject_data'] = base64.b64encode(sf.read()).decode('utf-8')
    if 'logo' in request.files:
        lf = request.files['logo']
        if lf.filename and allowed_image(lf.filename):
            fields['logo_data'] = base64.b64encode(lf.read()).decode('utf-8')
    if request.form.get('remove_subject') == 'true':
        fields['remove_subject'] = True
    if request.form.get('remove_logo') == 'true':
        fields['remove_logo'] = True
    if request.form.get('remove_style') == 'true':
        fields['remove_style'] = True
    animation_pattern = request.form.get('animation_pattern')
    if animation_pattern is not None:
        fields['animation_pattern'] = animation_pattern
    if update_channel(channel_id, **fields):
        return jsonify({'ok': True, 'id': channel_id})
    return jsonify({'error': 'Channel not found'}), 404

@app.route('/api/channels/<channel_id>', methods=['DELETE'])
def delete_channel_route(channel_id):
    return jsonify({'message': 'Deleted'}) if delete_channel(channel_id) else (jsonify({'error': 'Not found'}), 404)

@app.route('/api/channels/<channel_id>/logo.png')
def channel_logo_image(channel_id):
    path = os.path.join(app.config['CHANNEL_FOLDER'], channel_id, 'logo.png')
    return send_file(path, mimetype='image/png') if os.path.exists(path) else ('', 404)

@app.route('/api/channels/<channel_id>/style.png')
def channel_style_image(channel_id):
    path = os.path.join(app.config['CHANNEL_FOLDER'], channel_id, 'style.png')
    return send_file(path, mimetype='image/png') if os.path.exists(path) else ('', 404)

@app.route('/api/channels/<channel_id>/subject.png')
def channel_subject_image(channel_id):
    path = os.path.join(app.config['CHANNEL_FOLDER'], channel_id, 'subject.png')
    return send_file(path, mimetype='image/png') if os.path.exists(path) else ('', 404)

@app.route('/api/channels/reorder', methods=['POST'])
def reorder_channels():
    order = request.json.get('order', [])
    if not isinstance(order, list):
        return jsonify({'error': 'Invalid order'}), 400
    save_channel_registry(order)
    return jsonify({'ok': True})

@app.route('/api/channels/export')
def export_channels():
    channels = get_all_channels()
    export_data = []
    for ch in channels:
        channel_dir = os.path.join(app.config['CHANNEL_FOLDER'], ch['id'])
        entry = {
            'name': ch.get('name'), 'tags': ch.get('tags', []),
            'tag_colors': ch.get('tag_colors', {}), 'style_text': ch.get('style_text', ''),
            'scene_instructions': ch.get('scene_instructions', ''),
            'image_instructions': ch.get('image_instructions', ''),
            'format': ch.get('format', {}),
            'has_subject': ch.get('has_subject', False),
        }
        for img_name, key in [('style.png', 'style_b64'), ('subject.png', 'subject_b64'), ('logo.png', 'logo_b64')]:
            img_path = os.path.join(channel_dir, img_name)
            if os.path.exists(img_path):
                with open(img_path, 'rb') as f:
                    entry[key] = base64.b64encode(f.read()).decode('utf-8')
        export_data.append(entry)
    response = app.response_class(
        response=json.dumps(export_data, indent=2), mimetype='application/json',
        headers={'Content-Disposition': 'attachment; filename=channels-export.json'})
    return response

@app.route('/api/channels/import', methods=['POST'])
def import_channels():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    try:
        data = json.loads(file.read().decode('utf-8'))
    except Exception as e:
        return jsonify({'error': f'Invalid JSON: {str(e)}'}), 400
    if not isinstance(data, list):
        return jsonify({'error': 'Expected a list of channels'}), 400
    count = 0
    for entry in data:
        name = entry.get('name', 'Imported')
        fmt = entry.get('format', {})
        base_format = fmt.get('base', 'pulse') if fmt else 'pulse'
        style_b64 = entry.get('style_b64')
        subject_b64 = entry.get('subject_b64')
        logo_b64 = entry.get('logo_b64')
        style_text = entry.get('style_text', '')
        tags = entry.get('tags', [])
        tag_colors = entry.get('tag_colors', {})
        scene_instructions = entry.get('scene_instructions', '')
        image_instructions = entry.get('image_instructions', '')
        cid = save_channel(name, base_format=base_format, style_data=style_b64,
                           subject_data=subject_b64, logo_data=logo_b64,
                           style_text=style_text, tags=tags if isinstance(tags, list) else '',
                           tag_colors=tag_colors, scene_instructions=scene_instructions,
                           image_instructions=image_instructions)
        # Apply full format config if provided (overrides base defaults)
        if fmt and cid:
            ch_dir = os.path.join(app.config['CHANNEL_FOLDER'], cid, 'config.json')
            with open(ch_dir) as f:
                cfg = json.load(f)
            cfg['format'] = fmt
            with open(ch_dir, 'w') as f:
                json.dump(cfg, f, indent=2)
        count += 1
    return jsonify({'ok': True, 'count': count})

# ===================== SESSION STATE & RECENT GENERATIONS =====================
@app.route('/api/active-generations')
def get_active_generations():
    """Return count of currently processing generations."""
    with _active_sessions_lock:
        count = len(active_sessions)
    return jsonify({'count': count})

@app.route('/api/recent-generations')
def recent_generations():
    """Public endpoint: last 10 generations for the home screen."""
    history = load_history()
    recent = history[:10]
    safe_fields = ['session_id', 'timestamp', 'audio_duration', 'scene_count',
                   'channel_name', 'preset_name', 'format_label', 'animate_intro',
                   'status', 'video_url', 'project_title']
    return jsonify([{k: h.get(k) for k in safe_fields if k in h} for h in recent])

@app.route('/api/session/<session_id>/state')
def get_session_state(session_id):
    """Load generation state for reopening a previous generation."""
    work_dir = os.path.join(app.config['OUTPUT_FOLDER'], session_id)
    state_path = os.path.join(work_dir, 'generation_state.json')
    if not os.path.exists(state_path):
        return jsonify({'error': 'Session not found'}), 404
    with open(state_path) as f:
        state = json.load(f)
    return jsonify({
        'session_id': session_id,
        'scenes': state.get('scenes', []),
        'audio_duration': state.get('audio_duration', 0),
        'output_filename': state.get('output_filename', ''),
        'channel_id': state.get('channel_id', state.get('preset_id', '')),
        'channel_name': state.get('channel_name', ''),
        'video_url': f'/download/{session_id}/{state.get("output_filename", "")}' if state.get('output_filename') else None,
    })

# ===================== BACKWARD COMPAT: Preset API aliases =====================
@app.route('/api/presets', methods=['GET'])
def list_presets_compat():
    return list_channels()

@app.route('/api/presets/<preset_id>/style.png')
def preset_style_compat(preset_id):
    channel_id = f'ch_{preset_id}' if not preset_id.startswith('ch_') else preset_id
    return channel_style_image(channel_id)

@app.route('/api/presets/<preset_id>/subject.png')
def preset_subject_compat(preset_id):
    channel_id = f'ch_{preset_id}' if not preset_id.startswith('ch_') else preset_id
    return channel_subject_image(channel_id)

@app.route('/version')
def version():
    return jsonify({"version": "v52", "features": ["channels", "wizard", "subject", "long_form",
        "style_enforcement", "retry_logic", "resolution_1080p", "dark_mode", "audio_fix",
        "subject_detection", "scene_renumber", "style_text", "admin_dashboard"]})

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'openai': bool(OPENAI_API_KEY), 'gemini': bool(GEMINI_API_KEY),
                    'assemblyai': bool(ASSEMBLYAI_API_KEY),
                    'whisk_keys': len(whisk_pool), 'whisk_pool': whisk_pool.status()})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)
