# Voice Personality Analyzer

## Overview

A Streamlit-based web application that analyzes voice recordings to determine personality traits using the Big Five personality model. The app accepts audio input through either live recording or file upload, processes the audio to extract vocal features, and provides personality insights through interactive visualizations. Built as a demonstration of AI-powered voice analysis capabilities.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit for rapid web app development
- **UI Components**: Single-page application with dual input methods (recording and file upload)
- **Visualization**: Plotly for interactive personality trait charts (radar and bar charts)
- **State Management**: Streamlit's caching system for processor initialization

### Backend Architecture
- **Audio Processing Pipeline**: Multi-stage processing using librosa and pydub
  - Audio format conversion to standardized WAV format
  - Sample rate normalization to 16kHz
  - Duration validation and truncation (3-60 seconds)
  - Feature extraction from audio signals
- **Personality Analysis Engine**: Custom analyzer implementing Big Five personality model
  - Voice feature mapping to personality traits (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism)
  - Statistical normalization of trait scores
  - Trait interpretation with high/low descriptions

### Data Processing
- **Audio Features**: Extraction of pitch variation, spectral characteristics, and MFCC coefficients
- **Feature Engineering**: Statistical analysis of voice patterns to correlate with personality traits
- **Scoring System**: Normalized 0-1 scale for personality trait representation

### Input Validation
- **File Constraints**: 10MB size limit, format validation (MP3, WAV, M4A)
- **Audio Quality**: Duration bounds enforcement and empty file detection
- **Error Handling**: Comprehensive validation with user-friendly error messages

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework for rapid prototyping
- **librosa**: Advanced audio analysis and feature extraction
- **pydub**: Audio file format conversion and manipulation
- **NumPy**: Numerical computing for audio signal processing
- **Plotly**: Interactive data visualization for personality charts
- **Pandas**: Data manipulation for chart generation

### Audio Processing Stack
- **librosa**: Primary audio analysis library for feature extraction
- **pydub**: Audio format conversion and preprocessing
- **tempfile**: Temporary file management for audio processing pipeline

### Visualization Components
- **Plotly Graph Objects**: Radar charts for personality trait visualization
- **Plotly Express**: Bar charts for trait comparison
- **Streamlit Charts**: Integration layer for web display

### System Requirements
- **Python Runtime**: Core application environment
- **Browser Audio API**: For live recording functionality (via Streamlit)
- **File System**: Temporary storage for audio processing workflow