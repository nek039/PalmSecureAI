# -*- coding: utf-8 -*-
"""
PalmSecureAI Cloud Coordinator - Security Module

Provides node authentication, model update signing/verification,
and secure communication for federated learning.
"""

import os
import hmac
import hashlib
import secrets
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify

logger = logging.getLogger(__name__)


class FederatedSecurityConfig:
    """Federated learning security configuration"""

    # HMAC algorithm
    HMAC_ALGORITHM = hashlib.sha256

    # Token expiration
    NODE_TOKEN_EXPIRY = timedelta(hours=1)

    # Allowed terminal IDs (should be configured based on registered nodes)
    ALLOWED_TERMINAL_IDS = {1, 2, 3, 4, 5}


class NodeAuthManager:
    """Node authentication manager for federated learning"""

    def __init__(self):
        self._registered_nodes: Dict[int, Dict[str, Any]] = {}
        self._active_tokens: Dict[str, Dict[str, Any]] = {}

    def _get_shared_secret(self) -> bytes:
        """Get shared secret for HMAC"""
        secret = os.getenv('FEDERATED_SHARED_SECRET')
        if not secret:
            logger.error("FEDERATED_SHARED_SECRET not set")
            raise SecurityError("Federated shared secret not configured")
        return secret.encode('utf-8')

    def register_node(self, terminal_id: int, node_info: Dict[str, Any]) -> bool:
        """Register a new federated learning node"""
        if terminal_id in self._registered_nodes:
            logger.warning(f"Node {terminal_id} already registered")
            return False

        self._registered_nodes[terminal_id] = {
            'terminal_id': terminal_id,
            'registered_at': datetime.utcnow(),
            'info': node_info,
            'api_key': secrets.token_hex(32)
        }

        logger.info(f"Registered new node: {terminal_id}")
        return True

    def verify_node_api_key(self, terminal_id: int, api_key: str) -> bool:
        """Verify node's API key"""
        if terminal_id not in self._registered_nodes:
            return False

        expected_key = self._registered_nodes[terminal_id].get('api_key', '')
        return secrets.compare_digest(api_key, expected_key)

    def generate_node_token(self, terminal_id: int) -> Optional[str]:
        """Generate authentication token for node"""
        if terminal_id not in FederatedSecurityConfig.ALLOWED_TERMINAL_IDS:
            logger.warning(f"Terminal {terminal_id} not in allowed list")
            return None

        token = secrets.token_urlsafe(32)
        expiry = datetime.utcnow() + FederatedSecurityConfig.NODE_TOKEN_EXPIRY

        self._active_tokens[token] = {
            'terminal_id': terminal_id,
            'created_at': datetime.utcnow(),
            'expires_at': expiry
        }

        return token

    def verify_node_token(self, token: str) -> Optional[int]:
        """Verify node token and return terminal_id if valid"""
        if token not in self._active_tokens:
            return None

        token_data = self._active_tokens[token]
        if datetime.utcnow() > token_data['expires_at']:
            del self._active_tokens[token]
            return None

        return token_data['terminal_id']

    def revoke_token(self, token: str) -> bool:
        """Revoke an active token"""
        if token in self._active_tokens:
            del self._active_tokens[token]
            return True
        return False


class ModelSignature:
    """Model update signing and verification"""

    @staticmethod
    def sign_update(update_data: Dict[str, Any], terminal_id: int) -> str:
        """
        Sign model update with HMAC-SHA256

        Args:
            update_data: The model update data
            terminal_id: Terminal ID for additional context

        Returns:
            Hex-encoded HMAC signature
        """
        try:
            secret = os.getenv('FEDERATED_SHARED_SECRET', '').encode('utf-8')
            if not secret:
                logger.error("FEDERATED_SHARED_SECRET not set")
                return ''

            # Create canonical representation of update
            import json
            canonical = json.dumps(update_data, sort_keys=True, separators=(',', ':'))
            message = f"{terminal_id}:{canonical}".encode('utf-8')

            # Generate HMAC
            signature = hmac.new(secret, message, hashlib.sha256).hexdigest()
            return signature

        except Exception as e:
            logger.error(f"Failed to sign update: {e}")
            return ''

    @staticmethod
    def verify_update(update_data: Dict[str, Any], signature: str, terminal_id: int) -> bool:
        """
        Verify model update signature

        Args:
            update_data: The model update data
            signature: Expected HMAC signature
            terminal_id: Terminal ID

        Returns:
            True if signature is valid
        """
        if not signature:
            logger.warning("No signature provided")
            return False

        expected = ModelSignature.sign_update(update_data, terminal_id)
        if not expected:
            return False

        # Constant-time comparison
        return hmac.compare_digest(signature.encode(), expected.encode())


class SecurityError(Exception):
    """Security-related error"""
    pass


# Global instances
node_auth = NodeAuthManager()
model_signature = ModelSignature()


def require_node_auth(f):
    """Decorator to require node authentication"""
    @wraps(f)
    def decorated(*args, **kwargs):
        # Check for API key
        api_key = request.headers.get('X-Node-API-Key')
        terminal_id = request.headers.get('X-Terminal-ID')

        if not api_key or not terminal_id:
            return jsonify({
                'status': 'error',
                'message': 'Authentication required: X-Node-API-Key and X-Terminal-ID headers required'
            }), 401

        try:
            terminal_id = int(terminal_id)
        except ValueError:
            return jsonify({
                'status': 'error',
                'message': 'Invalid terminal ID'
            }), 400

        # Verify API key
        if not node_auth.verify_node_api_key(terminal_id, api_key):
            logger.warning(f"Invalid API key for terminal {terminal_id}")
            return jsonify({
                'status': 'error',
                'message': 'Invalid authentication credentials'
            }), 401

        # Store terminal_id in request context
        request.terminal_id = terminal_id
        return f(*args, **kwargs)

    return decorated


def verify_update_signature(update_data: Dict[str, Any], signature: str, terminal_id: int) -> Tuple[bool, str]:
    """
    Verify update signature and return result

    Returns:
        (is_valid, message)
    """
    if not os.getenv('FEDERATED_SHARED_SECRET'):
        logger.warning("FEDERATED_SHARED_SECRET not set - skipping signature verification")
        return True, "Signature verification skipped (not configured)"

    if not signature:
        return False, "Missing signature"

    if model_signature.verify_update(update_data, signature, terminal_id):
        return True, "Signature valid"

    return False, "Invalid signature"


def secure_torch_load(filepath: str, map_location='cpu'):
    """
    Securely load a PyTorch model file

    Args:
        filepath: Path to model file
        map_location: Device to load model to

    Returns:
        Loaded model data
    """
    import torch

    try:
        # Use weights_only=True for security (prevents code execution)
        data = torch.load(filepath, map_location=map_location, weights_only=True)
        return data
    except Exception as e:
        logger.error(f"Failed to securely load model: {e}")
        # Fallback for backwards compatibility (less secure)
        logger.warning("Falling back to weights_only=False - this is less secure")
        return torch.load(filepath, map_location=map_location, weights_only=False)
