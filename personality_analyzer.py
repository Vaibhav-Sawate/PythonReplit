import numpy as np
from typing import Dict, Tuple

class PersonalityAnalyzer:
    def __init__(self):
        """Initialize personality analyzer with trait definitions"""
        self.big_five_traits = [
            'Openness', 'Conscientiousness', 'Extraversion', 
            'Agreeableness', 'Neuroticism'
        ]
        
        # Define trait descriptions for interpretation
        self.trait_descriptions = {
            'Openness': {
                'high': 'creative, curious, open to new experiences',
                'low': 'practical, conventional, prefers routine'
            },
            'Conscientiousness': {
                'high': 'organized, disciplined, goal-oriented',
                'low': 'spontaneous, flexible, casual approach'
            },
            'Extraversion': {
                'high': 'outgoing, energetic, socially confident',
                'low': 'reserved, introspective, prefers solitude'
            },
            'Agreeableness': {
                'high': 'cooperative, trusting, empathetic',
                'low': 'competitive, skeptical, direct'
            },
            'Neuroticism': {
                'high': 'sensitive, emotionally reactive, anxious',
                'low': 'calm, emotionally stable, resilient'
            }
        }
    
    def normalize_score(self, value, min_val, max_val):
        """Normalize a value to 0-1 range"""
        if max_val == min_val:
            return 0.5
        return max(0, min(1, (value - min_val) / (max_val - min_val)))
    
    def analyze_openness(self, features):
        """Analyze openness based on voice features"""
        # Higher pitch variation and spectral diversity suggest openness
        pitch_variation = features.get('pitch_std', 0) / max(features.get('pitch_mean', 1), 1)
        spectral_variation = features.get('spectral_centroid_std', 0) / max(features.get('spectral_centroid_mean', 1), 1)
        
        # MFCC variation indicates vocal expressiveness
        mfcc_variation = np.mean(features.get('mfcc_std', [0]))
        
        openness_score = (
            self.normalize_score(pitch_variation, 0, 0.5) * 0.4 +
            self.normalize_score(spectral_variation, 0, 0.3) * 0.3 +
            self.normalize_score(mfcc_variation, 0, 50) * 0.3
        )
        
        return min(0.9, max(0.1, openness_score))
    
    def analyze_conscientiousness(self, features):
        """Analyze conscientiousness based on voice features"""
        # Consistent speaking patterns suggest conscientiousness
        rms_consistency = 1 - (features.get('rms_std', 0) / max(features.get('rms_mean', 1), 0.01))
        pitch_consistency = 1 - (features.get('pitch_std', 0) / max(features.get('pitch_mean', 1), 1))
        
        # Moderate speaking rate suggests thoughtfulness
        speaking_rate = features.get('speaking_rate', 150)
        rate_score = 1 - abs(speaking_rate - 150) / 150  # Optimal around 150 WPM
        
        conscientiousness_score = (
            self.normalize_score(rms_consistency, 0, 1) * 0.4 +
            self.normalize_score(pitch_consistency, 0, 1) * 0.4 +
            self.normalize_score(rate_score, 0, 1) * 0.2
        )
        
        return min(0.9, max(0.1, conscientiousness_score))
    
    def analyze_extraversion(self, features):
        """Analyze extraversion based on voice features"""
        # Louder, more energetic speech suggests extraversion
        energy_level = features.get('rms_mean', 0)
        pitch_level = features.get('pitch_mean', 0)
        speaking_rate = features.get('speaking_rate', 0)
        
        # Higher tempo and energy suggest extraversion
        tempo_score = self.normalize_score(features.get('tempo', 120), 100, 140)
        energy_score = self.normalize_score(energy_level, 0, 0.1)
        rate_score = self.normalize_score(speaking_rate, 100, 200)
        
        extraversion_score = (
            energy_score * 0.4 +
            tempo_score * 0.3 +
            rate_score * 0.3
        )
        
        return min(0.9, max(0.1, extraversion_score))
    
    def analyze_agreeableness(self, features):
        """Analyze agreeableness based on voice features"""
        # Warmer, softer vocal qualities suggest agreeableness
        # Lower pitch variation and gentler energy patterns
        pitch_gentleness = 1 - self.normalize_score(features.get('pitch_std', 0), 0, 100)
        energy_gentleness = 1 - self.normalize_score(features.get('rms_std', 0), 0, 0.05)
        
        # Moderate formant energy suggests warmth
        formant_warmth = self.normalize_score(features.get('formant_energy', 0), 0, 1000)
        
        agreeableness_score = (
            pitch_gentleness * 0.4 +
            energy_gentleness * 0.4 +
            formant_warmth * 0.2
        )
        
        return min(0.9, max(0.1, agreeableness_score))
    
    def analyze_neuroticism(self, features):
        """Analyze neuroticism based on voice features"""
        # High variation and tension in voice suggests neuroticism
        pitch_instability = features.get('pitch_std', 0) / max(features.get('pitch_mean', 1), 1)
        energy_instability = features.get('rms_std', 0) / max(features.get('rms_mean', 1), 0.01)
        
        # High zero-crossing rate can indicate tension
        zcr_tension = self.normalize_score(features.get('zcr_mean', 0), 0, 0.3)
        
        neuroticism_score = (
            self.normalize_score(pitch_instability, 0, 0.5) * 0.4 +
            self.normalize_score(energy_instability, 0, 2.0) * 0.4 +
            zcr_tension * 0.2
        )
        
        return min(0.9, max(0.1, neuroticism_score))
    
    def analyze_additional_traits(self, features):
        """Analyze additional personality traits"""
        additional_traits = {}
        
        # Confidence level (based on energy and consistency)
        energy_level = features.get('rms_mean', 0)
        consistency = 1 - (features.get('rms_std', 0) / max(features.get('rms_mean', 1), 0.01))
        confidence = (
            self.normalize_score(energy_level, 0, 0.1) * 0.6 +
            self.normalize_score(consistency, 0, 1) * 0.4
        )
        additional_traits['confidence'] = min(0.95, max(0.05, confidence))
        
        # Energy level (based on RMS and tempo)
        energy = (
            self.normalize_score(features.get('rms_mean', 0), 0, 0.1) * 0.7 +
            self.normalize_score(features.get('tempo', 120), 100, 140) * 0.3
        )
        additional_traits['energy'] = min(0.95, max(0.05, energy))
        
        # Speaking pace classification
        speaking_rate = features.get('speaking_rate', 150)
        if speaking_rate < 120:
            additional_traits['speaking_pace'] = "Slow"
        elif speaking_rate > 180:
            additional_traits['speaking_pace'] = "Fast"
        else:
            additional_traits['speaking_pace'] = "Moderate"
        
        # Emotional tone (based on pitch and energy patterns)
        pitch_mean = features.get('pitch_mean', 200)
        energy_mean = features.get('rms_mean', 0.05)
        
        if pitch_mean > 250 and energy_mean > 0.06:
            additional_traits['emotional_tone'] = "Positive"
        elif pitch_mean < 150 or energy_mean < 0.03:
            additional_traits['emotional_tone'] = "Calm"
        else:
            additional_traits['emotional_tone'] = "Neutral"
        
        return additional_traits
    
    def generate_summary(self, personality_scores, additional_traits):
        """Generate a text summary of the personality analysis"""
        # Find dominant traits
        sorted_traits = sorted(personality_scores.items(), key=lambda x: x[1], reverse=True)
        dominant_trait = sorted_traits[0]
        secondary_trait = sorted_traits[1]
        
        # Generate summary based on dominant traits
        summary_parts = []
        
        # Main personality description
        if dominant_trait[1] > 0.7:
            desc_key = 'high'
        elif dominant_trait[1] < 0.3:
            desc_key = 'low'
        else:
            desc_key = 'high'  # Default to high interpretation for moderate scores
        
        trait_desc = self.trait_descriptions[dominant_trait[0]][desc_key]
        summary_parts.append(f"Your voice suggests you are primarily {trait_desc}")
        
        # Secondary trait
        if secondary_trait[1] > 0.6:
            if secondary_trait[1] > 0.7:
                desc_key = 'high'
            else:
                desc_key = 'high'
            secondary_desc = self.trait_descriptions[secondary_trait[0]][desc_key]
            summary_parts.append(f"with strong {secondary_trait[0].lower()} traits ({secondary_desc})")
        
        # Add confidence and energy information
        confidence_level = additional_traits['confidence']
        energy_level = additional_traits['energy']
        
        if confidence_level > 0.7:
            summary_parts.append("Your speech patterns indicate high confidence")
        
        if energy_level > 0.7:
            summary_parts.append(f"with {additional_traits['speaking_pace'].lower()} paced, energetic communication")
        elif energy_level < 0.3:
            summary_parts.append(f"with a calm, measured communication style")
        
        return ". ".join(summary_parts) + "."
    
    def analyze(self, features: Dict) -> Tuple[Dict[str, float], Dict, str]:
        """
        Main analysis function that returns personality scores and summary
        
        Args:
            features: Audio features extracted from voice sample
            
        Returns:
            Tuple of (personality_scores, additional_traits, summary)
        """
        try:
            # Analyze Big Five personality traits
            personality_scores = {
                'Openness': self.analyze_openness(features),
                'Conscientiousness': self.analyze_conscientiousness(features),
                'Extraversion': self.analyze_extraversion(features),
                'Agreeableness': self.analyze_agreeableness(features),
                'Neuroticism': self.analyze_neuroticism(features)
            }
            
            # Analyze additional traits
            additional_traits = self.analyze_additional_traits(features)
            
            # Generate summary
            summary = self.generate_summary(personality_scores, additional_traits)
            
            return personality_scores, additional_traits, summary
            
        except Exception as e:
            raise Exception(f"Error analyzing personality: {str(e)}")
