# -*- coding: utf-8 -*-
"""
PalmSecureAI Centralized Security Configuration

This module provides centralized security configuration and validation
for the entire PalmSecureAI system.
"""

import os
import secrets
import hashlib
from typing import List, Optional


class SecurityConfig:
    """Centralized security configuration"""

    # Minimum key lengths
    MIN_SECRET_KEY_LENGTH = 32
    MIN_JWT_SECRET_LENGTH = 32

    # Default CORS origins (restrict in production)
    DEFAULT_ALLOWED_ORIGINS = ['http://localhost:8000', 'http://localhost:5000']

    # Rate limits
    DEFAULT_RATE_LIMIT = "100 per minute"
    AUTH_RATE_LIMIT = "5 per minute"

    # Security headers
    SECURITY_HEADERS = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'Referrer-Policy': 'strict-origin-when-cross-origin',
        'Permissions-Policy': 'geolocation=(), microphone=(), camera=(self)'
    }

    # Content Security Policy
    CSP_DIRECTIVES = {
        'default-src': "'self'",
        'script-src': "'self' 'unsafe-inline'",
        'style-src': "'self' 'unsafe-inline' https://fonts.googleapis.com",
        'font-src': "'self' https://fonts.gstatic.com",
        'img-src': "'self' data: blob:",
        'connect-src': "'self'",
        'frame-ancestors': "'none'",
        'base-uri': "'self'",
        'form-action': "'self'"
    }

    @classmethod
    def get_secret_key(cls) -> str:
        """Get SECRET_KEY from environment with validation"""
        key = os.getenv('SECRET_KEY')
        if not key:
            raise SecurityError(
                "SECRET_KEY not set. Please set a secure SECRET_KEY environment variable. "
                "Generate one with: openssl rand -hex 32"
            )
        if len(key) < cls.MIN_SECRET_KEY_LENGTH:
            raise SecurityError(
                f"SECRET_KEY must be at least {cls.MIN_SECRET_KEY_LENGTH} characters long"
            )
        return key

    @classmethod
    def get_jwt_secret_key(cls) -> str:
        """Get JWT_SECRET_KEY from environment with validation"""
        key = os.getenv('JWT_SECRET_KEY')
        if not key:
            raise SecurityError(
                "JWT_SECRET_KEY not set. Please set a secure JWT_SECRET_KEY environment variable. "
                "Generate one with: openssl rand -hex 32"
            )
        if len(key) < cls.MIN_JWT_SECRET_LENGTH:
            raise SecurityError(
                f"JWT_SECRET_KEY must be at least {cls.MIN_JWT_SECRET_LENGTH} characters long"
            )
        return key

    @classmethod
    def get_api_key(cls) -> str:
        """Get API_KEY from environment with validation"""
        key = os.getenv('API_KEY')
        if not key:
            raise SecurityError(
                "API_KEY not set. Please set a secure API_KEY environment variable. "
                "Generate one with: openssl rand -hex 32"
            )
        return key

    @classmethod
    def get_federated_shared_secret(cls) -> bytes:
        """Get FEDERATED_SHARED_SECRET from environment"""
        secret = os.getenv('FEDERATED_SHARED_SECRET')
        if not secret:
            raise SecurityError(
                "FEDERATED_SHARED_SECRET not set. Please set a secure FEDERATED_SHARED_SECRET "
                "environment variable. Generate one with: openssl rand -hex 32"
            )
        return secret.encode('utf-8')

    @classmethod
    def get_allowed_origins(cls) -> List[str]:
        """Get allowed CORS origins from environment"""
        origins_str = os.getenv('ALLOWED_ORIGINS', '')
        if origins_str:
            return [o.strip() for o in origins_str.split(',') if o.strip()]
        return cls.DEFAULT_ALLOWED_ORIGINS

    @classmethod
    def get_csp_header(cls) -> str:
        """Generate Content-Security-Policy header value"""
        return '; '.join(f"{k} {v}" for k, v in cls.CSP_DIRECTIVES.items())

    @classmethod
    def validate_configuration(cls) -> List[str]:
        """Validate security configuration and return list of warnings"""
        warnings = []

        # Check SECRET_KEY
        secret_key = os.getenv('SECRET_KEY', '')
        if not secret_key:
            warnings.append("SECRET_KEY not set - using this in production is insecure")
        elif len(secret_key) < cls.MIN_SECRET_KEY_LENGTH:
            warnings.append(f"SECRET_KEY is too short (min {cls.MIN_SECRET_KEY_LENGTH} chars)")
        elif secret_key in ['palmsecureai-secret-key', 'dev-secret-key', 'secret']:
            warnings.append("SECRET_KEY is using a default/weak value")

        # Check JWT_SECRET_KEY
        jwt_key = os.getenv('JWT_SECRET_KEY', '')
        if not jwt_key:
            warnings.append("JWT_SECRET_KEY not set")
        elif jwt_key == secret_key:
            warnings.append("JWT_SECRET_KEY should be different from SECRET_KEY")

        # Check API_KEY
        if not os.getenv('API_KEY'):
            warnings.append("API_KEY not set - API authentication is disabled")

        # Check FEDERATED_SHARED_SECRET
        if not os.getenv('FEDERATED_SHARED_SECRET'):
            warnings.append("FEDERATED_SHARED_SECRET not set - federated learning is insecure")

        # Check CORS
        allowed = cls.get_allowed_origins()
        if '*' in allowed or 'http://localhost' in allowed:
            warnings.append("CORS allows localhost - restrict in production")

        # Check environment
        env = os.getenv('FLASK_ENV', 'development')
        if env == 'development':
            warnings.append("Running in development mode - set FLASK_ENV=production for production")

        return warnings


class SecurityError(Exception):
    """Security configuration error"""
    pass


def generate_secure_key(length: int = 32) -> str:
    """Generate a cryptographically secure random key"""
    return secrets.token_hex(length)


def hash_api_key(api_key: str) -> str:
    """Hash an API key for storage/comparison"""
    return hashlib.sha256(api_key.encode()).hexdigest()


def constant_time_compare(val1: str, val2: str) -> bool:
    """Compare two strings in constant time to prevent timing attacks"""
    if len(val1) != len(val2):
        return False
    return secrets.compare_digest(val1.encode(), val2.encode())
