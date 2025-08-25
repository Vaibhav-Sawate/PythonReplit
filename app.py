import streamlit as st
import tempfile
import os
import io
import time
from audio_processor import AudioProcessor
from personality_analyzer import PersonalityAnalyzer
from utils import validate_audio_file, format_duration
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Initialize processors
@st.cache_resource
def get_processors():
    audio_processor = AudioProcessor()
    personality_analyzer = PersonalityAnalyzer()
    return audio_processor, personality_analyzer

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
    st.title("🎤 Voice Personality Analyzer")
    st.markdown("Analyze personality traits from voice recordings using AI-powered voice analysis")
    st.divider()
    
    # Get processors
    audio_processor, personality_analyzer = get_processors()
    
    # Audio Input Section
    st.header("Audio Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎙️ Record Audio")
        st.markdown("Record your voice directly in the browser")
        
        # Audio recorder
        audio_data = st.audio_input("Record your voice (max 60 seconds)")
        
        if audio_data is not None:
            st.success("Audio recorded successfully!")
            st.audio(audio_data)
    
    with col2:
        st.subheader("📁 Upload Audio File")
        st.markdown("Upload an audio file (MP3, WAV, M4A - max 10MB)")
        
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
                    
                    # Process audio
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Processing audio...")
                    progress_bar.progress(25)
                    
                    features = audio_processor.extract_features(tmp_file_path)
                    
                    status_text.text("Analyzing personality traits...")
                    progress_bar.progress(75)
                    
                    # Analyze personality
                    personality_scores, additional_traits, summary = personality_analyzer.analyze(features)
                    
                    progress_bar.progress(100)
                    status_text.text("Analysis complete!")
                    
                    # Clean up temporary file
                    os.unlink(tmp_file_path)
                    
                    time.sleep(1)  # Brief pause to show completion
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Results Display
                    st.divider()
                    st.header("📊 Analysis Results")
                    
                    # Summary
                    st.subheader("Summary")
                    st.info(summary)
                    
                    # Big Five Traits
                    st.subheader("Big Five Personality Traits")
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        fig_radar = create_personality_chart(personality_scores)
                        st.plotly_chart(fig_radar, use_container_width=True)
                    
                    with col2:
                        fig_bars = create_trait_bars(personality_scores)
                        st.plotly_chart(fig_bars, use_container_width=True)
                    
                    # Detailed scores
                    st.subheader("Detailed Trait Analysis")
                    
                    for trait, score in personality_scores.items():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**{trait}**")
                            st.progress(float(score))
                        with col2:
                            st.metric("Score", f"{score:.1%}")
                    
                    # Additional Traits
                    st.subheader("Additional Analysis")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Confidence Level",
                            f"{additional_traits['confidence']:.1%}",
                            delta=None
                        )
                    
                    with col2:
                        st.metric(
                            "Energy Level",
                            f"{additional_traits['energy']:.1%}",
                            delta=None
                        )
                    
                    with col3:
                        st.metric(
                            "Speaking Pace",
                            additional_traits['speaking_pace'],
                            delta=None
                        )
                    
                    with col4:
                        st.metric(
                            "Emotional Tone",
                            additional_traits['emotional_tone'],
                            delta=None
                        )
                    
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
