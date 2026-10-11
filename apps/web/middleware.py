# -*- coding: utf-8 -*-
"""
PalmSecureAI Web Application - Security Middleware

Provides security headers, request validation, and rate limiting.
"""

import os
import re
import logging
from functools import wraps
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from flask import request, Response, current_app, jsonify

logger = logging.getLogger(__name__)


class SecurityHeaders:
    """Security headers middleware"""

    @staticmethod
    def add_security_headers(response: Response) -> Response:
        """Add security headers to response"""

        # Content Security Policy
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: blob:; "
            "connect-src 'self' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )
        response.headers['Content-Security-Policy'] = csp

        # Prevent MIME type sniffing
        if not request.path.startswith('/static/'):
            response.headers['X-Content-Type-Options'] = 'nosniff'

        # Prevent clickjacking
        response.headers['X-Frame-Options'] = 'DENY'

        # XSS Protection
        response.headers['X-XSS-Protection'] = '1; mode=block'

        # HSTS (HTTPS Strict Transport Security)
        if os.getenv('FLASK_ENV') == 'production':
            response.headers['Strict-Transport-Security'] = (
                'max-age=31536000; includeSubDomains; preload'
            )

        # Referrer Policy
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'

        # Permissions Policy
        response.headers['Permissions-Policy'] = (
            'geolocation=(), microphone=(), camera=(self)'
        )

        return response


class RequestValidator:
    """Request validation utilities"""

    # Maximum allowed content length (16 MB)
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024

    # Allowed file extensions for uploads
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

    @classmethod
    def validate_file_upload(cls, file) -> tuple[bool, Optional[str]]:
        """Validate file upload"""
        if not file:
            return False, "No file provided"

        # Check file size
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(0)

        if size > cls.MAX_CONTENT_LENGTH:
            return False, f"File too large (max {cls.MAX_CONTENT_LENGTH // (1024*1024)}MB)"

        # Check file extension
        filename = getattr(file, 'filename', '')
        if not filename:
            return False, "No filename provided"

        ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
        if ext not in cls.ALLOWED_EXTENSIONS:
            return False, f"Invalid file type. Allowed: {', '.join(cls.ALLOWED_EXTENSIONS)}"

        return True, None

    @classmethod
    def sanitize_user_id(cls, user_id: str) -> tuple[bool, str]:
        """Sanitize user ID to prevent injection attacks"""
        if not user_id:
            return False, "User ID cannot be empty"

        # Limit length
        if len(user_id) > 64:
            return False, "User ID too long (max 64 characters)"

        # Only allow alphanumeric, underscore, hyphen
        if not re.match(r'^[a-zA-Z0-9_-]+$', user_id):
            return False, "User ID contains invalid characters"

        return True, user_id

    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """Sanitize filename to prevent path traversal"""
        # Remove path components
        filename = os.path.basename(filename)

        # Remove null bytes
        filename = filename.replace('\x00', '')

        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:255-len(ext)] + ext

        return filename


class RateLimiter:
    """Simple in-memory rate limiter"""

    def __init__(self):
        self._requests: Dict[str, list] = {}
        self._default_limit = 100  # requests
        self._window = 60  # seconds

    def is_allowed(self, key: str, limit: Optional[int] = None) -> bool:
        """Check if request is allowed under rate limit"""
        now = datetime.utcnow().timestamp()
        limit = limit or self._default_limit

        # Clean old requests
        if key in self._requests:
            self._requests[key] = [
                t for t in self._requests[key]
                if now - t < self._window
            ]
        else:
            self._requests[key] = []

        # Check limit
        if len(self._requests[key]) >= limit:
            return False

        # Record request
        self._requests[key].append(now)
        return True

    def get_remaining(self, key: str, limit: Optional[int] = None) -> int:
        """Get remaining requests in current window"""
        now = datetime.utcnow().timestamp()
        limit = limit or self._default_limit

        if key not in self._requests:
            return limit

        # Clean and count
        recent = [t for t in self._requests[key] if now - t < self._window]
        return max(0, limit - len(recent))


# Global rate limiter instance
rate_limiter = RateLimiter()


def rate_limit(limit: int = 100, per: int = 60, key_func=None):
    """Rate limiting decorator"""
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated(*args, **kwargs):
            # Get client identifier
            if key_func:
                key = key_func()
            else:
                key = request.remote_addr or 'unknown'

            # Check rate limit
            if not rate_limiter.is_allowed(key, limit):
                logger.warning(f"Rate limit exceeded for {key}")
                return jsonify({
                    'success': False,
                    'message': 'Rate limit exceeded. Please try again later.'
                }), 429

            return f(*args, **kwargs)
        return decorated
    return decorator


def validate_json_request(*required_fields):
    """Decorator to validate JSON request body"""
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated(*args, **kwargs):
            if not request.is_json:
                return jsonify({
                    'success': False,
                    'message': 'Content-Type must be application/json'
                }), 400

            data = request.get_json()
            if data is None:
                return jsonify({
                    'success': False,
                    'message': 'Invalid JSON body'
                }), 400

            # Check required fields
            missing = [field for field in required_fields if field not in data]
            if missing:
                return jsonify({
                    'success': False,
                    'message': f"Missing required fields: {', '.join(missing)}"
                }), 400

            return f(*args, **kwargs)
        return decorated
    return decorator


def setup_security_middleware(app):
    """Setup security middleware for Flask app"""

    @app.after_request
    def after_request(response):
        """Add security headers after each request"""
        return SecurityHeaders.add_security_headers(response)

    @app.before_request
    def before_request():
        """Validate requests before processing"""
        # Log suspicious requests
        if request.path.startswith('/api/'):
            user_agent = request.headers.get('User-Agent', '')
            if not user_agent or len(user_agent) < 5:
                logger.warning(f"Suspicious request from {request.remote_addr}: no user agent")

    logger.info("Security middleware initialized")
