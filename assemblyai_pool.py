"""
AssemblyAI Pool - Queued multi-key support for concurrent transcriptions.

Each key handles at most ONE transcription at a time. When all keys are busy,
new requests queue and wait until a key becomes free — preventing AssemblyAI
from throttling/stalling when multiple jobs hit the same account concurrently.

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
from collections import deque

logger = logging.getLogger(__name__)


class AssemblyAIPool:
    """Thread-safe pool of AssemblyAI API keys with 1-job-per-key queuing."""

    def __init__(self):
        self.keys = []  # list of {'key': str, 'cooldown_until': float}
        self._index = 0
        self._lock = threading.Lock()
        self._reservations = {}       # session_id -> key_index
        self._reservation_counts = {} # key_index -> count of active reservations
        self._key_available = threading.Condition(self._lock)
        self._queue = deque()         # ordered queue of waiting session_ids
        self._queue_set = set()       # fast lookup for queue membership
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

    def _find_free_key(self):
        """Find a key with 0 active reservations (not on cooldown). Returns index or None.
        Must be called with self._lock held."""
        now = time.time()
        best_idx = None
        for idx, entry in enumerate(self.keys):
            count = self._reservation_counts.get(idx, 0)
            if count == 0 and entry['cooldown_until'] <= now:
                best_idx = idx
                break  # first free key wins
        return best_idx

    def reserve_key(self, session_id, emit_fn=None):
        """Reserve a key for a transcription session. Only 1 job per key.
        If all keys are busy, blocks until one is free (FIFO queue).

        emit_fn: optional callable(position, total_queued) for queue status updates.
        Returns key string or None if no keys configured."""
        with self._lock:
            if not self.keys:
                return None

            # If session already has a reservation, return that key
            if session_id in self._reservations:
                idx = self._reservations[session_id]
                return self.keys[idx]['key']

            # Try to grab a free key immediately
            free_idx = self._find_free_key()
            if free_idx is not None:
                self._reservations[session_id] = free_idx
                self._reservation_counts[free_idx] = 1
                logger.info(f"AssemblyAI: reserved key {free_idx+1}/{len(self.keys)} for session {session_id} (immediate)")
                return self.keys[free_idx]['key']

            # All keys busy — join the queue
            self._queue.append(session_id)
            self._queue_set.add(session_id)
            queue_pos = len(self._queue)
            logger.info(f"AssemblyAI: session {session_id} queued at position {queue_pos} (all {len(self.keys)} keys busy)")

            if emit_fn:
                try:
                    emit_fn(queue_pos, len(self._queue))
                except Exception:
                    pass

            # Wait until this session is at the front of the queue AND a key is free
            while True:
                self._key_available.wait(timeout=10)

                # Check if session was removed from queue externally (e.g., cancel)
                if session_id not in self._queue_set:
                    # Session was cancelled while queued
                    return None

                # Only the front of the queue gets to try
                if self._queue and self._queue[0] == session_id:
                    free_idx = self._find_free_key()
                    if free_idx is not None:
                        self._queue.popleft()
                        self._queue_set.discard(session_id)
                        self._reservations[session_id] = free_idx
                        self._reservation_counts[free_idx] = 1
                        logger.info(f"AssemblyAI: reserved key {free_idx+1}/{len(self.keys)} for session {session_id} (from queue)")
                        return self.keys[free_idx]['key']

                # Update queue position for status display
                if emit_fn:
                    try:
                        pos = list(self._queue).index(session_id) + 1
                        emit_fn(pos, len(self._queue))
                    except (ValueError, Exception):
                        pass

    def release_key(self, session_id):
        """Release a session's key reservation and wake queued waiters."""
        with self._lock:
            # Remove from queue if still waiting (e.g., cancelled before reservation)
            if session_id in self._queue_set:
                self._queue_set.discard(session_id)
                try:
                    self._queue.remove(session_id)
                except ValueError:
                    pass
                logger.info(f"AssemblyAI: removed session {session_id} from queue")

            if session_id in self._reservations:
                idx = self._reservations.pop(session_id)
                self._reservation_counts.pop(idx, None)
                logger.info(f"AssemblyAI: released key {idx+1}/{len(self.keys)} for session {session_id}")
                # Wake up queued waiters so the next in line can grab this key
                self._key_available.notify_all()

    def cancel_queued(self, session_id):
        """Remove a session from the queue without releasing a key."""
        with self._lock:
            if session_id in self._queue_set:
                self._queue_set.discard(session_id)
                try:
                    self._queue.remove(session_id)
                except ValueError:
                    pass
                logger.info(f"AssemblyAI: cancelled queued session {session_id}")
                # Wake waiters so they can re-check their position
                self._key_available.notify_all()

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

    def queue_length(self):
        """Return the number of sessions waiting in queue."""
        with self._lock:
            return len(self._queue)

    def status(self):
        """Return pool status for health/admin endpoints."""
        now = time.time()
        with self._lock:
            return {
                'total_keys': len(self.keys),
                'available': sum(1 for idx in range(len(self.keys))
                                 if self._reservation_counts.get(idx, 0) == 0
                                 and self.keys[idx]['cooldown_until'] <= now),
                'cooling_down': sum(1 for k in self.keys if k['cooldown_until'] > now),
                'active_reservations': dict(self._reservation_counts),
                'reserved_sessions': list(self._reservations.keys()),
                'queued_sessions': list(self._queue),
                'queue_length': len(self._queue),
            }
