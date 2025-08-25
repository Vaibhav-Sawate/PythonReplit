import streamlit as st
from typing import Tuple

def validate_audio_file(uploaded_file) -> Tuple[bool, str]:
    """
    Validate uploaded audio file
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if uploaded_file is None:
        return False, "No file uploaded"
    
    # Check file size (10MB limit)
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB in bytes
    file_size = len(uploaded_file.getvalue())
    
    if file_size > MAX_FILE_SIZE:
        return False, f"File size ({file_size / 1024 / 1024:.1f}MB) exceeds the 10MB limit"
    
    # Check file type
    allowed_extensions = ['mp3', 'wav', 'm4a']
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension not in allowed_extensions:
        return False, f"File type '{file_extension}' not supported. Please use: {', '.join(allowed_extensions)}"
    
    # Check if file is empty
    if file_size == 0:
        return False, "File appears to be empty"
    
    return True, ""

def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable format
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"

def get_trait_interpretation(trait_name: str, score: float) -> str:
    """
    Get interpretation text for a personality trait score
    
    Args:
        trait_name: Name of the personality trait
        score: Normalized score (0-1)
        
    Returns:
        Interpretation text
    """
    interpretations = {
        'Openness': {
            'high': 'You tend to be imaginative, creative, and open to new experiences',
            'medium': 'You balance creativity with practicality',
            'low': 'You prefer familiar routines and practical approaches'
        },
        'Conscientiousness': {
            'high': 'You are organized, disciplined, and goal-oriented',
            'medium': 'You balance structure with flexibility',
            'low': 'You prefer spontaneity and adaptability'
        },
        'Extraversion': {
            'high': 'You are outgoing, energetic, and socially confident',
            'medium': 'You balance social interaction with alone time',
            'low': 'You are reserved and prefer quieter environments'
        },
        'Agreeableness': {
            'high': 'You are cooperative, trusting, and considerate of others',
            'medium': 'You balance cooperation with assertiveness',
            'low': 'You are direct, competitive, and skeptical'
        },
        'Neuroticism': {
            'high': 'You may experience emotions intensely and be sensitive to stress',
            'medium': 'You have moderate emotional responses',
            'low': 'You tend to be emotionally stable and resilient'
        }
    }
    
    if score > 0.7:
        level = 'high'
    elif score > 0.3:
        level = 'medium'
    else:
        level = 'low'
    
    return interpretations.get(trait_name, {}).get(level, 'No interpretation available')
