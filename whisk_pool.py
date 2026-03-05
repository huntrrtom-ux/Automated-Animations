"""
V56: Whisk Pool - Round-robin multi-key support with cooldown failover

Drop-in replacement for single-key Whisk auth. Works with 1 key (identical
behaviour to current code) or N keys (round-robin with 429 cooldown).

RAILWAY ENV VARS:
  Single key (current - still works):
    WHISK_API_KEY=ya29.xxx
    WHISK_COOKIE=__Secure-xxx...

  Multiple keys (new):
    WHISK_KEYS=token1|cookie1,token2|cookie2,token3|cookie3

  Or numbered:
    WHISK_KEY_1=ya29.xxx
    WHISK_COOKIE_1=__Secure-xxx...
    WHISK_KEY_2=ya29.yyy
    WHISK_COOKIE_2=__Secure-yyy...
"""

import os
import time
import threading
import logging

logger = logging.getLogger(__name__)


class WhiskPool:
    """Thread-safe round-robin pool of Whisk API keys with cooldown on 429s."""

    def __init__(self):
        self.keys = []  # list of {'token': str, 'cookie': str, 'cooldown_until': float}
        self._index = 0
        self._lock = threading.Lock()
        self._reservations = {}       # session_id -> key_index
        self._reservation_counts = {} # key_index -> count of active reservations
        self._quota_exhausted = set() # key indices with 0 credits remaining
        self._load_keys()
        logger.info(f"WhiskPool initialized with {len(self.keys)} key(s)")

    def _load_keys(self):
        """Load keys from environment variables. Supports 3 formats."""

        # Format 1: WHISK_KEYS=token1|cookie1,token2|cookie2,...
        keys_str = os.environ.get('WHISK_KEYS', '')
        if keys_str:
            for pair in keys_str.split(','):
                pair = pair.strip()
                if '|' in pair:
                    token, cookie = pair.split('|', 1)
                    self.keys.append({
                        'token': token.strip(),
                        'cookie': cookie.strip(),
                        'cooldown_until': 0
                    })
            if self.keys:
                return

        # Always check for original WHISK_API_KEY / WHISK_COOKIE first
        orig_token = os.environ.get('WHISK_API_KEY') or os.environ.get('WHISK_API_TOKEN') or ''
        orig_cookie = os.environ.get('WHISK_COOKIE', '')
        if orig_token or orig_cookie:
            self.keys.append({
                'token': orig_token,
                'cookie': orig_cookie,
                'cooldown_until': 0
            })

        # Then add any numbered keys (WHISK_KEY_1, WHISK_COOKIE_1, etc.)
        for i in range(1, 11):  # support up to 10 extra keys
            token = os.environ.get(f'WHISK_KEY_{i}', '')
            cookie = os.environ.get(f'WHISK_COOKIE_{i}', '')
            if token or cookie:
                self.keys.append({
                    'token': token,
                    'cookie': cookie,
                    'cooldown_until': 0
                })

    def get_next(self):
        """Get the next available key (skipping cooled-down ones).
        Returns dict with 'token', 'cookie', 'index' or None if all cooling down."""
        with self._lock:
            if not self.keys:
                return None

            now = time.time()
            n = len(self.keys)

            # Try each key starting from current index
            for attempt in range(n):
                idx = (self._index + attempt) % n
                key = self.keys[idx]
                if key['cooldown_until'] <= now:
                    self._index = (idx + 1) % n  # advance for next call
                    return {
                        'token': key['token'],
                        'cookie': key['cookie'],
                        'index': idx
                    }

            # All keys cooling down - return the one with shortest remaining cooldown
            soonest = min(self.keys, key=lambda k: k['cooldown_until'])
            idx = self.keys.index(soonest)
            wait = soonest['cooldown_until'] - now
            logger.warning(f"All {n} Whisk keys cooling down. Shortest wait: {wait:.0f}s (key {idx+1})")
            return {
                'token': soonest['token'],
                'cookie': soonest['cookie'],
                'index': idx,
                'wait_seconds': wait
            }

    def mark_cooldown(self, index, seconds=60):
        """Mark a key as rate-limited for N seconds."""
        with self._lock:
            if 0 <= index < len(self.keys):
                self.keys[index]['cooldown_until'] = time.time() + seconds
                logger.info(f"Whisk key {index+1}/{len(self.keys)} cooling down for {seconds}s")

    def mark_expired(self, index):
        """Mark a key as expired (long cooldown until env vars refreshed)."""
        with self._lock:
            if 0 <= index < len(self.keys):
                self.keys[index]['cooldown_until'] = time.time() + 300  # 5 min
                logger.warning(f"Whisk key {index+1}/{len(self.keys)} marked expired (5min cooldown)")

    def mark_quota_exhausted(self, index):
        """Mark a key as having 0 credits (quota reached). Long cooldown until daily reset."""
        with self._lock:
            if 0 <= index < len(self.keys):
                self._quota_exhausted.add(index)
                self.keys[index]['cooldown_until'] = time.time() + 3600  # 1 hour
                exhausted = len(self._quota_exhausted)
                total = len(self.keys)
                logger.warning(f"Whisk key {index+1}/{total} has 0 credits (quota exhausted) — {exhausted}/{total} keys exhausted")

    def all_quota_exhausted(self):
        """Check if ALL keys have exhausted their quota."""
        with self._lock:
            if not self.keys:
                return True
            return len(self._quota_exhausted) >= len(self.keys)

    def clear_quota(self, index):
        """Clear quota-exhausted status for a key (e.g. after a successful request)."""
        with self._lock:
            self._quota_exhausted.discard(index)

    def refresh_key(self, index, token=None, cookie=None):
        """Update a key's credentials (e.g. after token refresh)."""
        with self._lock:
            if 0 <= index < len(self.keys):
                if token:
                    self.keys[index]['token'] = token
                if cookie:
                    self.keys[index]['cookie'] = cookie
                self.keys[index]['cooldown_until'] = 0

    def reload_from_env(self):
        """Reload all keys from environment (picks up Railway var changes)."""
        with self._lock:
            old_count = len(self.keys)
            self.keys = []
            self._index = 0
            self._quota_exhausted.clear()
        self._load_keys()
        logger.info(f"WhiskPool reloaded: {old_count} -> {len(self.keys)} keys")

    def reserve_key(self, session_id):
        """Reserve a key for a generation session. Picks the key with fewest
        active reservations for load balancing. Returns key dict or None."""
        with self._lock:
            if not self.keys:
                return None

            # If session already has a reservation, return that key
            if session_id in self._reservations:
                idx = self._reservations[session_id]
                key = self.keys[idx]
                return {
                    'token': key['token'],
                    'cookie': key['cookie'],
                    'index': idx
                }

            now = time.time()

            # Find available key with fewest reservations
            best_idx = None
            best_count = float('inf')
            for idx, key in enumerate(self.keys):
                if key['cooldown_until'] > now:
                    continue
                count = self._reservation_counts.get(idx, 0)
                if count < best_count:
                    best_count = count
                    best_idx = idx

            # If all cooling down, pick the one with fewest reservations
            if best_idx is None:
                best_idx = min(range(len(self.keys)),
                               key=lambda i: (self._reservation_counts.get(i, 0),
                                              self.keys[i]['cooldown_until']))

            self._reservations[session_id] = best_idx
            self._reservation_counts[best_idx] = self._reservation_counts.get(best_idx, 0) + 1
            logger.info(f"Reserved key {best_idx+1}/{len(self.keys)} for session {session_id} "
                        f"(active reservations on key: {self._reservation_counts[best_idx]})")

            key = self.keys[best_idx]
            return {
                'token': key['token'],
                'cookie': key['cookie'],
                'index': best_idx
            }

    def get_reserved_key(self, session_id):
        """Get the reserved key for a session. Falls back to get_next() if
        no reservation exists (backward compat)."""
        with self._lock:
            if session_id in self._reservations:
                idx = self._reservations[session_id]
                key = self.keys[idx]
                result = {
                    'token': key['token'],
                    'cookie': key['cookie'],
                    'index': idx
                }
                now = time.time()
                if key['cooldown_until'] > now:
                    result['wait_seconds'] = key['cooldown_until'] - now
                return result
        # No reservation — fallback to round-robin
        return self.get_next()

    def get_unreserved_key(self, session_id):
        """Get a non-cooling-down key that isn't reserved by another session.
        Returns key dict or None if no such key exists.  Used by animation
        retries so they don't steal keys from concurrent generations."""
        with self._lock:
            if not self.keys:
                return None

            now = time.time()
            own_idx = self._reservations.get(session_id)

            # Collect indices reserved by OTHER sessions
            other_reserved = {idx for sid, idx in self._reservations.items() if sid != session_id}

            for idx, key in enumerate(self.keys):
                if key['cooldown_until'] > now:
                    continue
                if idx in other_reserved:
                    continue
                return {
                    'token': key['token'],
                    'cookie': key['cookie'],
                    'index': idx
                }

            # No unreserved key available — caller should wait on its own
            return None

    def release_key(self, session_id):
        """Release a session's key reservation."""
        with self._lock:
            if session_id in self._reservations:
                idx = self._reservations.pop(session_id)
                count = self._reservation_counts.get(idx, 1)
                if count <= 1:
                    self._reservation_counts.pop(idx, None)
                else:
                    self._reservation_counts[idx] = count - 1
                logger.info(f"Released key {idx+1}/{len(self.keys)} for session {session_id}")

    def __len__(self):
        return len(self.keys)

    def status(self):
        """Return pool status for admin/logging."""
        now = time.time()
        return {
            'total_keys': len(self.keys),
            'available': sum(1 for k in self.keys if k['cooldown_until'] <= now),
            'cooling_down': sum(1 for k in self.keys if k['cooldown_until'] > now),
            'active_reservations': dict(self._reservation_counts),
            'reserved_sessions': list(self._reservations.keys()),
            'quota_exhausted': len(self._quota_exhausted),
            'all_quota_exhausted': len(self._quota_exhausted) >= len(self.keys) if self.keys else True,
            'keys': [
                {
                    'index': i,
                    'has_token': bool(k['token']),
                    'has_cookie': bool(k['cookie']),
                    'available': k['cooldown_until'] <= now,
                    'cooldown_remaining': max(0, k['cooldown_until'] - now),
                    'reservations': self._reservation_counts.get(i, 0),
                    'quota_exhausted': i in self._quota_exhausted
                }
                for i, k in enumerate(self.keys)
            ]
        }


# =====================================================================
# INTEGRATION CODE - Replace existing auth functions in app.py
# =====================================================================

# Initialize pool (do this once at module level, replacing current auth)
# whisk_pool = WhiskPool()

# Then replace calls throughout app.py as shown below:

INTEGRATION_NOTES = """
=== CHANGES NEEDED IN app.py ===

1. IMPORTS: Add at top of app.py:
   from whisk_pool import WhiskPool

2. INITIALIZATION: Replace the WHISK AUTH section (~line 718):

   OLD:
   # ===================== WHISK AUTH =====================
   def get_whisk_token():
       return os.environ.get('WHISK_API_KEY') or os.environ.get('WHISK_API_TOKEN') or ''

   def whisk_bearer_headers():
       return {
           "authorization": f"Bearer {get_whisk_token()}",
           ...
       }

   def whisk_cookie_headers():
       return {
           "cookie": get_whisk_cookie(),
           ...
       }

   NEW:
   # ===================== WHISK AUTH =====================
   whisk_pool = WhiskPool()

   def get_whisk_key():
       '''Get next available Whisk key from pool.'''
       key = whisk_pool.get_next()
       if not key:
           logger.error("No Whisk keys configured!")
           return None
       if key.get('wait_seconds', 0) > 0:
           wait = key['wait_seconds']
           logger.info(f"All keys cooling down, waiting {wait:.0f}s for key {key['index']+1}")
           time.sleep(min(wait, 30))  # wait up to 30s
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

3. IMAGE GENERATION (generate_image_with_recipe, ~line 836):

   OLD:
   token = get_whisk_token()
   headers = {
       "authorization": f"Bearer {token}",
       ...
   }
   response = req.post("https://aisandbox-pa.googleapis.com/v1/whisk:runImageRecipe",
                        data=json.dumps(json_data), headers=headers, timeout=120)

   NEW:
   key = get_whisk_key()
   if not key:
       create_placeholder_image(prompt, output_path)
       return None
   headers = {
       "authorization": f"Bearer {key['token']}",
       "content-type": "text/plain;charset=UTF-8",
       "origin": "https://labs.google",
       "referer": "https://labs.google/",
       "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
   }
   logger.info(f"Whisk runImageRecipe for scene {scene_num} (subject={scene_has_subject}, inputs={len(recipe_inputs)}, key={key['index']+1}/{len(whisk_pool)})")
   response = req.post("https://aisandbox-pa.googleapis.com/v1/whisk:runImageRecipe",
                        data=json.dumps(json_data), headers=headers, timeout=120)

4. HANDLE 429 IN RETRY LOOP (~line 814 in the 3-attempt loop):

   After detecting a 429 or "PUBLIC_ERROR_USER_QUOTA_REACHED":
       whisk_pool.mark_cooldown(key['index'], seconds=60)
       key = get_whisk_key()  # gets next available key
       # rebuild headers with new key
       headers["authorization"] = f"Bearer {key['token']}"

5. HANDLE 401 TOKEN EXPIRED:

   After detecting 401:
       whisk_pool.mark_expired(key['index'])
       key = get_whisk_key()  # gets next available key

6. UPLOAD/CAPTION FUNCTIONS (caption_image_whisk, upload_image_to_whisk):
   These use cookie headers. Same pattern:

   OLD:
   headers = whisk_cookie_headers()

   NEW:
   key = get_whisk_key()
   headers = whisk_cookie_headers_for(key)

   Note: For upload_preset_images_to_whisk, get ONE key at the start and
   use it for all uploads in that session (keeps media IDs on same account).

7. ADMIN API (optional): Add a /admin/whisk-pool endpoint:

   @app.route('/admin/whisk-pool')
   def admin_whisk_pool():
       auth = request.cookies.get('admin_auth')
       if auth != ADMIN_PASSWORD:
           return jsonify({'error': 'Unauthorized'}), 401
       return jsonify(whisk_pool.status())

=== RAILWAY ENV VARS ===

Option A (simplest for 1-3 keys):
   WHISK_KEY_1=ya29.xxx_first_token
   WHISK_COOKIE_1=__Secure-xxx_first_cookie
   WHISK_KEY_2=ya29.yyy_second_token
   WHISK_COOKIE_2=__Secure-yyy_second_cookie

Option B (compact):
   WHISK_KEYS=ya29.xxx|__Secure-xxx,ya29.yyy|__Secure-yyy

Option C (backwards compatible - single key, no changes needed):
   WHISK_API_KEY=ya29.xxx
   WHISK_COOKIE=__Secure-xxx
"""

if __name__ == '__main__':
    # Quick test
    os.environ['WHISK_KEYS'] = 'token1|cookie1,token2|cookie2,token3|cookie3'
    pool = WhiskPool()
    print(f"Pool size: {len(pool)}")
    print(f"Status: {pool.status()}")

    # Simulate round-robin
    for i in range(6):
        key = pool.get_next()
        print(f"Request {i+1}: key {key['index']+1} (token={key['token']})")

    # Simulate cooldown
    print("\n--- Marking key 1 as rate-limited ---")
    pool.mark_cooldown(0, seconds=30)
    for i in range(4):
        key = pool.get_next()
        print(f"Request: key {key['index']+1} (token={key['token']})")

    print(f"\nStatus: {pool.status()}")
