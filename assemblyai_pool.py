"""
AssemblyAI Pool - Round-robin multi-key support for concurrent transcriptions.

Distributes transcription jobs across multiple API keys so concurrent
generations don't bottleneck on a single account's rate/concurrency limits.

RAILWAY ENV VARS:
  Single key (current - still works):
    ASSEMBLYAI_API_KEY=abc123

  Multiple keys (numbered):
    ASSEMBLYAI_API_KEY=abc123
    ASSEMBLYAI_API_KEY_2=def456
    ASSEMBLYAI_API_KEY_3=ghi789

  Or comma-separated:
    ASSEMBLYAI_KEYS=abc123,def456,ghi789
"""

import os
import time
import threading
import logging

logger = logging.getLogger(__name__)


class AssemblyAIPool:
    """Thread-safe round-robin pool of AssemblyAI API keys."""

    def __init__(self):
        self.keys = []  # list of {'key': str, 'cooldown_until': float}
        self._index = 0
        self._lock = threading.Lock()
        self._reservations = {}       # session_id -> key_index
        self._reservation_counts = {} # key_index -> count of active reservations
        self._load_keys()
        logger.info(f"AssemblyAIPool initialized with {len(self.keys)} key(s)")

    def _load_keys(self):
        """Load keys from environment variables."""

        # Format 1: ASSEMBLYAI_KEYS=key1,key2,key3
        keys_str = os.environ.get('ASSEMBLYAI_KEYS', '')
        if keys_str:
            for k in keys_str.split(','):
                k = k.strip()
                if k:
                    self.keys.append({'key': k, 'cooldown_until': 0})
            if self.keys:
                return

        # Format 2: ASSEMBLYAI_API_KEY (original) + ASSEMBLYAI_API_KEY_2, _3, etc.
        orig = os.environ.get('ASSEMBLYAI_API_KEY', '')
        if orig:
            self.keys.append({'key': orig, 'cooldown_until': 0})

        for i in range(2, 11):  # support up to 10 keys
            k = os.environ.get(f'ASSEMBLYAI_API_KEY_{i}', '')
            if k:
                self.keys.append({'key': k, 'cooldown_until': 0})

    def reserve_key(self, session_id):
        """Reserve a key for a transcription session. Picks the key with fewest
        active reservations for load balancing. Returns key string or None."""
        with self._lock:
            if not self.keys:
                return None

            # If session already has a reservation, return that key
            if session_id in self._reservations:
                idx = self._reservations[session_id]
                return self.keys[idx]['key']

            now = time.time()

            # Find available key with fewest reservations
            best_idx = None
            best_count = float('inf')
            for idx, entry in enumerate(self.keys):
                if entry['cooldown_until'] > now:
                    continue
                count = self._reservation_counts.get(idx, 0)
                if count < best_count:
                    best_count = count
                    best_idx = idx

            # If all cooling down, pick fewest reservations
            if best_idx is None:
                best_idx = min(range(len(self.keys)),
                               key=lambda i: (self._reservation_counts.get(i, 0),
                                              self.keys[i]['cooldown_until']))

            self._reservations[session_id] = best_idx
            self._reservation_counts[best_idx] = self._reservation_counts.get(best_idx, 0) + 1
            logger.info(f"AssemblyAI: reserved key {best_idx+1}/{len(self.keys)} for session {session_id} "
                        f"(active: {self._reservation_counts[best_idx]})")
            return self.keys[best_idx]['key']

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
                logger.info(f"AssemblyAI: released key {idx+1}/{len(self.keys)} for session {session_id}")

    def mark_cooldown(self, session_id, seconds=120):
        """Mark the key used by a session as rate-limited."""
        with self._lock:
            idx = self._reservations.get(session_id)
            if idx is not None and 0 <= idx < len(self.keys):
                self.keys[idx]['cooldown_until'] = time.time() + seconds
                logger.warning(f"AssemblyAI key {idx+1}/{len(self.keys)} cooling down for {seconds}s")

    def has_keys(self):
        """Check if any keys are configured."""
        return len(self.keys) > 0

    def __len__(self):
        return len(self.keys)

    def status(self):
        """Return pool status for health/admin endpoints."""
        now = time.time()
        return {
            'total_keys': len(self.keys),
            'available': sum(1 for k in self.keys if k['cooldown_until'] <= now),
            'cooling_down': sum(1 for k in self.keys if k['cooldown_until'] > now),
            'active_reservations': dict(self._reservation_counts),
            'reserved_sessions': list(self._reservations.keys()),
        }
