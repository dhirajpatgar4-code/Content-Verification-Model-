"""
This file is kept for backward compatibility.
The actual models are defined in api/schemas.py
"""

# Import from schemas to maintain compatibility
from api.schemas import Base, ContentVerification, BusinessProfile, PostedContent

# Re-export for backward compatibility
__all__ = ['Base', 'ContentVerification', 'BusinessProfile', 'PostedContent']