import streamlit as st
import tempfile
import os
import io
import time
from audio_processor import AudioProcessor
from ml_personality_analyzer import MLPersonalityAnalyzer
from speaker_diarization import SpeakerDiarization
from utils import validate_audio_file, format_duration
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Initialize processors
@st.cache_resource
def get_processors():
    audio_processor = AudioProcessor()
    personality_analyzer = MLPersonalityAnalyzer()
    speaker_diarizer = SpeakerDiarization()
    return audio_processor, personality_analyzer, speaker_diarizer

def create_personality_chart(scores):
    """Create a radar chart for personality traits"""
    traits = list(scores.keys())
    values = list(scores.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=traits,
        fill='toself',
        name='Personality Traits',
        line_color='rgb(32, 201, 151)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        title="Personality Trait Analysis",
        height=400
    )
    
    return fig

def create_trait_bars(scores):
    """Create horizontal bar chart for traits"""
    df = pd.DataFrame([
        {'Trait': trait, 'Score': score, 'Percentage': f"{score*100:.1f}%"}
        for trait, score in scores.items()
    ])
    
    fig = px.bar(
        df, 
        x='Score', 
        y='Trait',
        orientation='h',
        text='Percentage',
        color='Score',
        color_continuous_scale='Viridis',
        range_x=[0, 1]
    )
    
    fig.update_layout(
        title="Trait Scores",
        showlegend=False,
        height=300,
        coloraxis_showscale=False
    )
    
    fig.update_traces(textposition='outside')
    
    return fig

def main():
    st.set_page_config(
        page_title="Voice Personality Analyzer",
        page_icon="🎤",
        layout="wide"
    )
    
    # Header
    st.title("🎤 Multi-Speaker Voice Personality Analyzer")
    st.markdown("Advanced personality analysis with **automatic speaker identification** - analyzes each person separately in group conversations using machine learning and psychological research")
    st.divider()
    
    # Get processors
    audio_processor, personality_analyzer, speaker_diarizer = get_processors()
    
    # Audio Input Section
    st.header("Audio Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎙️ Record Audio")
        st.markdown("Record voices directly in the browser (supports multiple speakers)")
        
        # Audio recorder
        audio_data = st.audio_input("Record your voice (max 60 seconds)")
        
        if audio_data is not None:
            st.success("Audio recorded successfully!")
            st.audio(audio_data)
    
    with col2:
        st.subheader("📁 Upload Audio File")
        st.markdown("Upload audio files with conversations or multiple speakers (MP3, WAV, M4A - max 10MB)")
        
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['mp3', 'wav', 'm4a'],
            accept_multiple_files=False
        )
        
        if uploaded_file is not None:
            file_valid, error_msg = validate_audio_file(uploaded_file)
            if file_valid:
                st.success(f"File uploaded: {uploaded_file.name}")
                st.audio(uploaded_file)
            else:
                st.error(error_msg)
                uploaded_file = None
    
    # Processing Section
    audio_source = None
    if audio_data is not None:
        audio_source = audio_data
        source_type = "recorded"
    elif uploaded_file is not None:
        audio_source = uploaded_file
        source_type = "uploaded"
    
    if audio_source is not None:
        st.divider()
        
        if st.button("🔬 Analyze Personality", type="primary", use_container_width=True):
            with st.spinner("Processing audio and analyzing personality traits..."):
                try:
                    # Save audio to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        if source_type == "recorded":
                            tmp_file.write(audio_source.getvalue())
                        else:
                            tmp_file.write(audio_source.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    # Process audio with speaker diarization
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Identifying speakers in audio...")
                    progress_bar.progress(20)
                    
                    # Diarize speakers
                    speaker_segments = speaker_diarizer.diarize(tmp_file_path)
                    
                    status_text.text("Extracting features for each speaker...")
                    progress_bar.progress(40)
                    
                    # Process each speaker
                    speaker_results = {}
                    num_speakers = len(speaker_segments)
                    
                    for i, (speaker_id, speaker_audio) in enumerate(speaker_segments.items()):
                        # Save speaker audio to temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f'_speaker_{speaker_id}.wav') as speaker_tmp:
                            # Convert numpy array back to audio file
                            import scipy.io.wavfile as wavfile
                            wavfile.write(speaker_tmp.name, 16000, (speaker_audio * 32767).astype(np.int16))
                            speaker_tmp_path = speaker_tmp.name
                        
                        try:
                            # Extract features for this speaker
                            speaker_features = audio_processor.extract_features(speaker_tmp_path)
                            
                            # Analyze personality for this speaker
                            personality_scores, additional_traits, summary = personality_analyzer.analyze(speaker_features)
                            
                            speaker_results[speaker_id] = {
                                'personality_scores': personality_scores,
                                'additional_traits': additional_traits,
                                'summary': summary,
                                'duration': len(speaker_audio) / 16000
                            }
                        finally:
                            os.unlink(speaker_tmp_path)
                        
                        # Update progress
                        progress = 40 + (50 * (i + 1) / num_speakers)
                        progress_bar.progress(int(progress))
                        status_text.text(f"Analyzing speaker {speaker_id + 1} of {num_speakers}...")
                    
                    progress_bar.progress(100)
                    status_text.text("Analysis complete!")
                    
                    # Clean up temporary file
                    os.unlink(tmp_file_path)
                    
                    time.sleep(1)  # Brief pause to show completion
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Results Display
                    st.divider()
                    st.header("📊 Multi-Speaker Analysis Results")
                    
                    # Speaker summary
                    if num_speakers > 1:
                        st.success(f"🎙️ Detected {num_speakers} different speakers in the audio")
                        
                        # Show speaker durations
                        col_speakers = st.columns(min(num_speakers, 4))
                        for i, (speaker_id, result) in enumerate(speaker_results.items()):
                            if i < len(col_speakers):
                                with col_speakers[i]:
                                    st.metric(
                                        f"Speaker {speaker_id + 1}",
                                        f"{result['duration']:.1f}s",
                                        delta="speaking time"
                                    )
                        st.divider()
                    else:
                        st.info("📢 Single speaker detected in the audio")
                    
                    # Display results for each speaker
                    for speaker_id, result in speaker_results.items():
                        speaker_num = speaker_id + 1
                        
                        # Speaker header
                        if num_speakers > 1:
                            st.subheader(f"🎭 Speaker {speaker_num} - Personality Analysis")
                            st.caption(f"Speaking duration: {result['duration']:.1f} seconds")
                        else:
                            st.subheader("🎭 Personality Analysis")
                        
                        # Summary
                        st.info(result['summary'])
                        
                        # Big Five Traits Charts
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            fig_radar = create_personality_chart(result['personality_scores'])
                            fig_radar.update_layout(title=f"Speaker {speaker_num} - Personality Traits")
                            st.plotly_chart(fig_radar, use_container_width=True)
                        
                        with col2:
                            fig_bars = create_trait_bars(result['personality_scores'])
                            fig_bars.update_layout(title=f"Speaker {speaker_num} - Trait Scores")
                            st.plotly_chart(fig_bars, use_container_width=True)
                        
                        # Detailed scores
                        st.write("**Detailed Trait Analysis:**")
                        
                        for trait, score in result['personality_scores'].items():
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"**{trait}**")
                                st.progress(float(score))
                            with col2:
                                st.metric("Score", f"{score:.1%}")
                        
                        # Additional Traits
                        st.write("**Additional Analysis:**")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Confidence Level",
                                f"{result['additional_traits']['confidence']:.1%}",
                                delta=None
                            )
                        
                        with col2:
                            st.metric(
                                "Energy Level",
                                f"{result['additional_traits']['energy']:.1%}",
                                delta=None
                            )
                        
                        with col3:
                            st.metric(
                                "Speaking Pace",
                                result['additional_traits']['speaking_pace'],
                                delta=None
                            )
                        
                        with col4:
                            st.metric(
                                "Emotional Tone",
                                result['additional_traits']['emotional_tone'],
                                delta=None
                            )
                        
                        # Add separator between speakers
                        if num_speakers > 1 and speaker_id != max(speaker_results.keys()):
                            st.divider()
                    
                except Exception as e:
                    st.error(f"Error processing audio: {str(e)}")
                    st.info("Please ensure your audio file is clear and contains speech.")
                    if 'tmp_file_path' in locals():
                        try:
                            os.unlink(tmp_file_path)
                        except:
                            pass
    
    else:
        st.info("👆 Please record audio or upload an audio file to begin analysis")
    
    # Footer
    st.divider()
    st.markdown("""
    **Note:** This analysis is for entertainment and educational purposes. 
    Results should not be used for professional psychological assessment.
    """)

if __name__ == "__main__":
    main()
