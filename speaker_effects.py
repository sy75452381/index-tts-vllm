"""
Sound Effect Applier for Audio Stories
Based on research for vocal effects using Spotify's Pedalboard library
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional
import numpy as np
from pedalboard import (
    Pedalboard,
    HighpassFilter,
    LowpassFilter,
    Compressor,
    Gain,
    Distortion,
    Delay,
    Reverb,
    Bitcrush,
    PitchShift,
    Chorus,
    Phaser,
    Limiter,
)


class EffectType(Enum):
    """Available sound effect types for audio stories"""
    # Communication/Technology effects
    PHONE_CALL = "phone_call"
    RADIO_BROADCAST = "radio_broadcast"
    WALKIE_TALKIE = "walkie_talkie"
    BAD_SIGNAL = "bad_signal"
    MEGAPHONE = "megaphone"
    OLD_GRAMOPHONE = "old_gramophone"
    METALLIC_HELMET = "metallic_helmet"
    # Spatial/Environment effects
    REVERB_SMALL_ROOM = "reverb_small_room"
    REVERB_LARGE_HALL = "reverb_large_hall"
    STADIUM_ANNOUNCEMENT = "stadium_announcement"
    DREAM_SEQUENCE = "dream_sequence"
    UNDERWATER = "underwater"
    # Character voice effects
    ROBOTIC_VOICE = "robotic_voice"
    MONSTER_VOICE = "monster_voice"
    INTIMATE_WHISPER = "intimate_whisper"
    GHOST_SPIRIT = "ghost_spirit"
    GIANT_TITAN = "giant_titan"
    FAIRY_TINY = "fairy_tiny"
    ALIEN_VOICE = "alien_voice"
    ANCIENT_GOD = "ancient_god"


@dataclass
class EffectPreset:
    """Preset configuration for a sound effect"""
    name: str
    description: str
    use_case: str
    pedalboard: Pedalboard
    parameters: dict = field(default_factory=dict)


class SoundEffectApplier:
    """
    A class to apply various sound effects to audio for storytelling.
    
    Each effect is designed based on research for enhancing dialogue
    in audio stories, podcasts, and audio dramas.
    """
    
    def __init__(self):
        self._presets: dict[EffectType, EffectPreset] = {}
        self._initialize_presets()
    
    def _initialize_presets(self) -> None:
        """Initialize all effect presets"""
        
        # 1. Phone Call Effect
        # Enhanced with saturation and mid-frequency emphasis for authentic telephone sound
        self._presets[EffectType.PHONE_CALL] = EffectPreset(
            name="Phone Call",
            description=(
                "Simulates a voice heard through a telephone receiver. "
                "Characterized by narrow frequency range, slight compression, "
                "and a lo-fi quality with mid-range emphasis."
            ),
            use_case=(
                "Use when a character is speaking on the phone. Creates a sense "
                "of distance or intimacy depending on the narrative."
            ),
            pedalboard=Pedalboard([
                # Aggressive filtering for telephone bandwidth
                HighpassFilter(cutoff_frequency_hz=400),
                LowpassFilter(cutoff_frequency_hz=3000),
                # Second filter pass for steeper rolloff
                HighpassFilter(cutoff_frequency_hz=350),
                LowpassFilter(cutoff_frequency_hz=3200),
                # Light saturation for that analog phone character
                Distortion(drive_db=3),
                # Heavy compression typical of phone lines
                Compressor(threshold_db=-15, ratio=6, attack_ms=2, release_ms=80),
                # Makeup gain
                Gain(gain_db=4)
            ]),
            parameters={
                "highpass_hz": 400,
                "lowpass_hz": 3000,
                "distortion_db": 3,
                "threshold_db": -15,
                "ratio": 6,
                "gain_db": 4
            }
        )
        
        # 2. Radio Broadcast Effect
        # Enhanced with more authentic radio processing chain
        self._presets[EffectType.RADIO_BROADCAST] = EffectPreset(
            name="Radio Broadcast",
            description=(
                "Emulates the sound of a voice being broadcast over AM/FM radio. "
                "Features classic broadcast compression, presence boost, and "
                "analog warmth."
            ),
            use_case=(
                "Ideal for radio announcers, news reports, or any broadcast "
                "being heard from a radio."
            ),
            pedalboard=Pedalboard([
                # Radio bandwidth (slightly wider than phone)
                HighpassFilter(cutoff_frequency_hz=100),
                LowpassFilter(cutoff_frequency_hz=6000),
                # AM radio has narrower bandwidth
                # Subtle saturation for tube/transformer warmth
                Distortion(drive_db=2.5),
                # Heavy broadcast-style compression (radio is heavily compressed)
                Compressor(threshold_db=-18, ratio=8, attack_ms=3, release_ms=100),
                # Small room ambience (simulates speaker in room)
                Reverb(room_size=0.1, damping=0.8, wet_level=0.08, dry_level=0.92),
                # Final output
                Gain(gain_db=3)
            ]),
            parameters={
                "highpass_hz": 100,
                "lowpass_hz": 6000,
                "distortion_db": 2.5,
                "threshold_db": -18,
                "ratio": 8,
                "room_size": 0.1,
                "gain_db": 3
            }
        )
        
        # 3. Walkie-Talkie Effect
        # Professional military/police radio simulation
        self._presets[EffectType.WALKIE_TALKIE] = EffectPreset(
            name="Walkie-Talkie",
            description=(
                "Simulates a two-way radio with extremely narrow frequency band, "
                "aggressive limiting, and characteristic 'crackle' from cheap "
                "piezoelectric speakers."
            ),
            use_case=(
                "Perfect for police, military, security personnel, or any "
                "characters using short-range radio communication."
            ),
            pedalboard=Pedalboard([
                # Very narrow bandwidth - walkie-talkies are extremely limited
                HighpassFilter(cutoff_frequency_hz=500),
                LowpassFilter(cutoff_frequency_hz=2500),
                # Second pass for steeper cutoff
                HighpassFilter(cutoff_frequency_hz=450),
                LowpassFilter(cutoff_frequency_hz=2700),
                # Pre-distortion compression to even out levels
                Compressor(threshold_db=-30, ratio=15, attack_ms=1, release_ms=50),
                # Heavy distortion for cheap speaker/electronics character
                Distortion(drive_db=20),
                # Bitcrush for digital radio artifacts
                Bitcrush(bit_depth=8),
                # Final limiting to prevent clipping
                Compressor(threshold_db=-8, ratio=20, attack_ms=0.5, release_ms=30),
                Gain(gain_db=-4)
            ]),
            parameters={
                "highpass_hz": 500,
                "lowpass_hz": 2500,
                "threshold_db": -30,
                "ratio": 15,
                "distortion_db": 20,
                "bit_depth": 8,
                "gain_db": -4
            }
        )
        
        # 4. Small Room Reverb
        # Realistic small room with early reflections simulation
        self._presets[EffectType.REVERB_SMALL_ROOM] = EffectPreset(
            name="Small Room Reverb",
            description=(
                "Creates the ambiance of a small, intimate room with tight, "
                "controlled reflections. Simulates offices, bedrooms, closets, "
                "or small studios."
            ),
            use_case=(
                "Place characters in small rooms, offices, intimate indoor spaces, "
                "or create a sense of enclosed proximity."
            ),
            pedalboard=Pedalboard([
                # Very short pre-delay simulates early reflections
                Delay(delay_seconds=0.008, feedback=0.15, mix=0.1),
                # Small tight reverb
                Reverb(room_size=0.15, damping=0.7, wet_level=0.2, dry_level=0.8, width=0.6),
                # High-frequency rolloff for natural room absorption
                LowpassFilter(cutoff_frequency_hz=10000),
                # Subtle room resonance
                Chorus(rate_hz=0.1, depth=0.05, mix=0.05),
                # Light compression for consistency
                Compressor(threshold_db=-18, ratio=3, attack_ms=5, release_ms=80),
                Gain(gain_db=0)
            ]),
            parameters={
                "pre_delay": 0.008,
                "room_size": 0.15,
                "damping": 0.7,
                "wet_level": 0.2,
                "dry_level": 0.8,
                "width": 0.6,
                "gain_db": 0
            }
        )
        
        # 5. Reverb - Large Hall
        # Epic cathedral/cave reverb with long tail
        self._presets[EffectType.REVERB_LARGE_HALL] = EffectPreset(
            name="Large Hall Reverb",
            description=(
                "Creates an expansive, epic reverberant atmosphere like a "
                "cathedral, cave, warehouse, or concert hall with dramatic "
                "long decay."
            ),
            use_case=(
                "Place characters in cathedrals, caves, warehouses, or other "
                "vast acoustic spaces for dramatic effect."
            ),
            pedalboard=Pedalboard([
                # Pre-delay for sense of distance (sound takes time to reach walls)
                Delay(delay_seconds=0.04, feedback=0.1, mix=0.08),
                # Large hall reverb
                Reverb(room_size=0.95, damping=0.3, wet_level=0.4, dry_level=0.6, width=1.0),
                # Additional depth with second reverb layer
                Reverb(room_size=0.8, damping=0.5, wet_level=0.15, dry_level=0.85, width=0.8),
                # Gentle high-frequency absorption (large spaces absorb highs)
                LowpassFilter(cutoff_frequency_hz=9000),
                # Very subtle modulation for air movement
                Chorus(rate_hz=0.08, depth=0.03, mix=0.03)
            ]),
            parameters={
                "pre_delay": 0.04,
                "room_size": 0.95,
                "damping": 0.3,
                "wet_level": 0.4,
                "dry_level": 0.6,
                "width": 1.0
            }
        )
        
        # 7. Bad Signal / Digital Distortion
        # Enhanced with multiple degradation stages for authentic digital corruption
        self._presets[EffectType.BAD_SIGNAL] = EffectPreset(
            name="Bad Signal",
            description=(
                "Simulates a poor-quality digital connection with lo-fi artifacts, "
                "bandwidth limiting, and digital distortion typical of failing "
                "video calls or corrupted transmissions."
            ),
            use_case=(
                "Failing video calls, malfunctioning robots/AI, high-tech "
                "interference, or technological decay."
            ),
            pedalboard=Pedalboard([
                # Simulate codec bandwidth limiting
                HighpassFilter(cutoff_frequency_hz=200),
                LowpassFilter(cutoff_frequency_hz=4000),
                # Bitcrush for digital quantization artifacts
                Bitcrush(bit_depth=6),
                # Digital clipping/saturation
                Distortion(drive_db=8),
                # Compression to simulate codec compression artifacts
                Compressor(threshold_db=-20, ratio=12, attack_ms=1, release_ms=50),
                # Final level adjustment
                Gain(gain_db=-2)
            ]),
            parameters={
                "highpass_hz": 200,
                "lowpass_hz": 4000,
                "bit_depth": 6,
                "distortion_db": 8,
                "threshold_db": -20,
                "ratio": 12,
                "gain_db": -2
            }
        )
        
        # 8. Robotic / Android Voice
        # Professional sci-fi robot voice with metallic resonance
        self._presets[EffectType.ROBOTIC_VOICE] = EffectPreset(
            name="Robotic Voice",
            description=(
                "Creates a synthetic, unnatural voice with metallic resonance, "
                "digital processing artifacts, and inhuman precision. Perfect "
                "for AI, androids, and futuristic settings."
            ),
            use_case=(
                "Androids, robots, AI systems, cyborgs, or digitally altered "
                "realities."
            ),
            pedalboard=Pedalboard([
                # Pre-compress to manage dynamics before effects (prevents transients)
                Compressor(threshold_db=-20, ratio=4, attack_ms=1, release_ms=30),
                # Slight pitch shift for unnatural quality
                PitchShift(semitones=-1.5),
                # Resonant comb-filter effect via very short delay (reduced feedback)
                Delay(delay_seconds=0.005, feedback=0.5, mix=0.35),
                # Phaser for metallic swirling modulation (reduced intensity)
                Phaser(rate_hz=0.15, depth=0.7, feedback=0.6, mix=0.4),
                # Chorus for synthetic multi-voice layering (reduced)
                Chorus(rate_hz=0.8, depth=0.4, mix=0.35),
                # Digital artifacts
                Bitcrush(bit_depth=12),
                # Final compression for consistent robotic delivery
                Compressor(threshold_db=-18, ratio=6, attack_ms=2, release_ms=60),
                Gain(gain_db=2),
                # Limiter at the end to catch any transients
                Limiter(threshold_db=-3, release_ms=50)
            ]),
            parameters={
                "pitch_semitones": -1.5,
                "comb_delay": 0.005,
                "comb_feedback": 0.5,
                "phaser_rate": 0.15,
                "phaser_depth": 0.7,
                "phaser_feedback": 0.6,
                "chorus_rate": 0.8,
                "chorus_depth": 0.4,
                "bit_depth": 12,
                "gain_db": 2
            }
        )
        
        # 9. Monster / Demon Voice
        # Terrifying otherworldly voice with layered processing
        self._presets[EffectType.MONSTER_VOICE] = EffectPreset(
            name="Monster Voice",
            description=(
                "Deep, menacing, otherworldly voice with imposing pitch, "
                "aggressive distortion, unsettling fractured quality, and "
                "vast demonic reverberance."
            ),
            use_case=(
                "Demons, large monsters, corrupted beings, dark gods, eldritch "
                "horrors, or any terrifying antagonist."
            ),
            pedalboard=Pedalboard([
                # Pre-compress to manage dynamics before effects (prevents transients)
                Compressor(threshold_db=-20, ratio=4, attack_ms=1, release_ms=30),
                # Massive pitch drop for inhuman depth
                PitchShift(semitones=-7),
                # Add sub-harmonic resonance
                Delay(delay_seconds=0.025, feedback=0.3, mix=0.2),
                # Aggressive saturation for growl/aggression (reduced from 18dB to prevent clipping)
                Distortion(drive_db=12),
                # Low-pass to emphasize rumbling bass
                LowpassFilter(cutoff_frequency_hz=6000),
                # Slow chorus for unsettling multi-voice effect (reduced)
                Chorus(rate_hz=0.4, depth=0.3, mix=0.35),
                # Dark phaser for otherworldly swirling (reduced)
                Phaser(rate_hz=0.2, depth=0.5, feedback=0.4, mix=0.25),
                # Massive dark reverb
                Reverb(room_size=0.9, damping=0.95, wet_level=0.4, dry_level=0.6, width=1.0),
                # Final compression to tame dynamics
                Compressor(threshold_db=-12, ratio=4, attack_ms=5, release_ms=100),
                Gain(gain_db=-2),
                # Limiter at the end to catch any transients
                Limiter(threshold_db=-3, release_ms=50)
            ]),
            parameters={
                "pitch_semitones": -7,
                "distortion_db": 12,
                "chorus_rate": 0.4,
                "chorus_depth": 0.3,
                "phaser_rate": 0.2,
                "phaser_depth": 0.5,
                "room_size": 0.9,
                "damping": 0.95,
                "gain_db": -2
            }
        )
        
        # 10. Intimate Whisper / Internal Monologue
        # Creates an intimate "inner voice" quality - close, soft, psychological
        # Rather than simulating physical whisper acoustics, we focus on the
        # emotional/psychological quality of intimate inner thoughts
        self._presets[EffectType.INTIMATE_WHISPER] = EffectPreset(
            name="Intimate Whisper",
            description=(
                "Close, intimate 'in-your-head' quality for private thoughts "
                "or whispered secrets. Creates psychological closeness and "
                "the feeling of hearing someone's inner voice."
            ),
            use_case=(
                "Internal monologue, secrets, tender confessions, or building "
                "psychological proximity with the listener."
            ),
            pedalboard=Pedalboard([
                # Highpass to thin out voice - removes "chest" resonance for whisper quality
                HighpassFilter(cutoff_frequency_hz=220),
                # Gentle compression for consistent intimate level
                Compressor(threshold_db=-20, ratio=3, attack_ms=10, release_ms=150),
                # Very subtle chorus for slight airy/breathy shimmer
                Chorus(rate_hz=0.5, depth=0.1, mix=0.1),
                # Small intimate reverb for "close to ear" quality
                Reverb(room_size=0.1, damping=0.9, wet_level=0.1, dry_level=0.9, width=0.2),
                # Significant volume reduction - whispers are much softer
                Gain(gain_db=-10)
            ]),
            parameters={
                "highpass_hz": 220,
                "threshold_db": -20,
                "ratio": 3,
                "chorus_rate": 0.5,
                "chorus_depth": 0.1,
                "room_size": 0.1,
                "gain_db": -10
            }
        )
        
        # =====================================================================
        # CHARACTER-BASED EFFECTS
        # =====================================================================
        
        # 11. Underwater Voice
        # Muffled, bubbly quality as if speaking underwater
        self._presets[EffectType.UNDERWATER] = EffectPreset(
            name="Underwater",
            description=(
                "Creates a muffled, submerged sound as if the voice is heard "
                "underwater. Heavy low-pass filtering with wavering modulation "
                "simulates water distortion and pressure."
            ),
            use_case=(
                "Characters speaking underwater, drowning scenes, mermaid or "
                "sea creature dialogue, submerged flashbacks, or dream sequences "
                "with water themes."
            ),
            pedalboard=Pedalboard([
                # Pre-compress to manage dynamics before effects (prevents transients)
                Compressor(threshold_db=-20, ratio=4, attack_ms=1, release_ms=30),
                # Heavy low-pass to simulate water absorption of high frequencies
                LowpassFilter(cutoff_frequency_hz=800),
                # Second pass for steeper rolloff
                LowpassFilter(cutoff_frequency_hz=1200),
                # Subtle pitch wobble for water movement (reduced)
                Chorus(rate_hz=0.3, depth=0.45, mix=0.3),
                # Phaser for swirling underwater currents (reduced)
                Phaser(rate_hz=0.15, depth=0.4, feedback=0.3, mix=0.25),
                # Short delay for acoustic reflections in water
                Delay(delay_seconds=0.02, feedback=0.25, mix=0.2),
                # Reverb for underwater ambience
                Reverb(room_size=0.6, damping=0.8, wet_level=0.35, dry_level=0.65, width=0.9),
                # Compression for consistent underwater feel
                Compressor(threshold_db=-18, ratio=4, attack_ms=10, release_ms=100),
                Gain(gain_db=3),
                # Limiter at the end to catch any transients
                Limiter(threshold_db=-3, release_ms=50)
            ]),
            parameters={
                "lowpass_hz": 800,
                "chorus_rate": 0.3,
                "chorus_depth": 0.45,
                "phaser_rate": 0.15,
                "room_size": 0.6,
                "gain_db": 3
            }
        )
        
        # 12. Ghost / Spirit Voice
        # Ethereal, otherworldly presence
        self._presets[EffectType.GHOST_SPIRIT] = EffectPreset(
            name="Ghost Spirit",
            description=(
                "An ethereal, haunting voice that seems to come from beyond "
                "the veil. Combines airy breathiness with hollow reverb and "
                "spectral shimmer for supernatural presence."
            ),
            use_case=(
                "Ghosts, spirits, apparitions, voices from beyond the grave, "
                "supernatural entities, or ancestral spirits giving guidance."
            ),
            pedalboard=Pedalboard([
                # Pre-compress to manage dynamics before effects (prevents transients)
                Compressor(threshold_db=-20, ratio=4, attack_ms=1, release_ms=30),
                # Remove body for ethereal quality
                HighpassFilter(cutoff_frequency_hz=300),
                # Slight pitch shift up for otherworldly quality
                PitchShift(semitones=2),
                # Slow chorus for shimmer and phasing (reduced)
                Chorus(rate_hz=0.2, depth=0.5, mix=0.4),
                # Phaser for spectral swirling (reduced)
                Phaser(rate_hz=0.1, depth=0.6, feedback=0.4, mix=0.3),
                # Long ethereal reverb
                Reverb(room_size=0.85, damping=0.2, wet_level=0.55, dry_level=0.45, width=1.0),
                # Additional reverb layer for depth
                Reverb(room_size=0.7, damping=0.4, wet_level=0.2, dry_level=0.8, width=0.8),
                # Echo for ghostly repetition (reduced feedback)
                Delay(delay_seconds=0.3, feedback=0.25, mix=0.15),
                Compressor(threshold_db=-15, ratio=3, attack_ms=15, release_ms=150),
                Gain(gain_db=-1),
                # Limiter at the end to catch any transients
                Limiter(threshold_db=-3, release_ms=50)
            ]),
            parameters={
                "highpass_hz": 300,
                "pitch_semitones": 2,
                "chorus_rate": 0.2,
                "chorus_depth": 0.5,
                "phaser_rate": 0.1,
                "room_size": 0.85,
                "delay_seconds": 0.3,
                "gain_db": -1
            }
        )
        
        # 13. Giant / Titan Voice
        # Massive, booming, earth-shaking presence
        self._presets[EffectType.GIANT_TITAN] = EffectPreset(
            name="Giant Titan",
            description=(
                "A massive, booming voice that conveys immense size and power. "
                "Deep pitch with rumbling bass, slow attack, and vast reverb "
                "create the impression of a towering being."
            ),
            use_case=(
                "Giants, titans, colossal creatures, dragons, ancient beings, "
                "kaiju, or any enormous character whose voice should shake "
                "the ground."
            ),
            pedalboard=Pedalboard([
                # Pre-compress to manage dynamics before effects (prevents transients)
                Compressor(threshold_db=-20, ratio=4, attack_ms=1, release_ms=30),
                # Significant pitch drop for massive scale
                PitchShift(semitones=-5),
                # Boost lows for rumbling presence
                HighpassFilter(cutoff_frequency_hz=40),
                LowpassFilter(cutoff_frequency_hz=8000),
                # Subtle distortion for power
                Distortion(drive_db=4),
                # Slow chorus for size (reduced)
                Chorus(rate_hz=0.15, depth=0.25, mix=0.2),
                # Very short delay for doubling effect (size)
                Delay(delay_seconds=0.03, feedback=0.15, mix=0.15),
                # Massive hall reverb
                Reverb(room_size=0.95, damping=0.4, wet_level=0.35, dry_level=0.65, width=1.0),
                # Slow compression for weight
                Compressor(threshold_db=-12, ratio=3, attack_ms=20, release_ms=200),
                Gain(gain_db=2),
                # Limiter at the end to catch any transients
                Limiter(threshold_db=-3, release_ms=50)
            ]),
            parameters={
                "pitch_semitones": -5,
                "distortion_db": 4,
                "chorus_rate": 0.15,
                "room_size": 0.95,
                "gain_db": 2
            }
        )
        
        # 14. Fairy / Tiny Creature Voice
        # High-pitched, magical, delicate - balanced for clarity and effect
        self._presets[EffectType.FAIRY_TINY] = EffectPreset(
            name="Fairy Tiny",
            description=(
                "A higher-pitched, delicate, and magical voice suitable for tiny "
                "creatures. Sparkly high frequencies with slight shimmer create "
                "a whimsical, enchanted quality while preserving speech clarity."
            ),
            use_case=(
                "Fairies, pixies, sprites, tiny magical creatures, insects with "
                "voices, miniaturized characters, or playful forest spirits."
            ),
            pedalboard=Pedalboard([
                # Pitch shift up for tiny fairy voice (4 semitones for noticeable effect)
                PitchShift(semitones=4),
                # Brighten the voice slightly
                HighpassFilter(cutoff_frequency_hz=200),
                # Gentle chorus for shimmer (very conservative settings)
                Chorus(rate_hz=1.0, depth=0.1, mix=0.15),
                # Small reverb for magical quality
                Reverb(room_size=0.2, damping=0.5, wet_level=0.25, dry_level=0.75, width=0.6),
                # Compression to maintain consistent level
                Compressor(threshold_db=-12, ratio=3, attack_ms=5, release_ms=80),
                Gain(gain_db=2)
            ]),
            parameters={
                "pitch_semitones": 4,
                "highpass_hz": 200,
                "chorus_rate": 1.0,
                "room_size": 0.2,
                "gain_db": 2
            }
        )
        
        # 15. Alien Voice
        # Strange, unearthly, non-human communication
        self._presets[EffectType.ALIEN_VOICE] = EffectPreset(
            name="Alien Voice",
            description=(
                "An otherworldly, non-human voice that sounds distinctly alien. "
                "Combines unusual modulation, metallic resonance, and strange "
                "harmonic content for extraterrestrial communication."
            ),
            use_case=(
                "Extraterrestrials, alien species, interdimensional beings, "
                "unknown entities, or any creature clearly not of this world."
            ),
            pedalboard=Pedalboard([
                # Pre-compress to manage dynamics before effects (prevents transients)
                Compressor(threshold_db=-20, ratio=4, attack_ms=1, release_ms=30),
                # Ring modulator-like effect via fast chorus (reduced intensity)
                Chorus(rate_hz=10, depth=0.5, mix=0.35),
                # Comb filtering for metallic resonance (reduced feedback)
                Delay(delay_seconds=0.003, feedback=0.5, mix=0.35),
                # Phaser for alien warble (significantly reduced)
                Phaser(rate_hz=0.5, depth=0.6, feedback=0.5, mix=0.4),
                # Subtle pitch shift for strangeness
                PitchShift(semitones=-2),
                # Bitcrush for digital/alien tech feel
                Bitcrush(bit_depth=10),
                # Slight distortion
                Distortion(drive_db=3),
                # Small reverb for alien environment
                Reverb(room_size=0.3, damping=0.6, wet_level=0.25, dry_level=0.75, width=0.7),
                Compressor(threshold_db=-15, ratio=5, attack_ms=2, release_ms=60),
                Gain(gain_db=0),
                # Limiter at the end to catch any transients
                Limiter(threshold_db=-3, release_ms=50)
            ]),
            parameters={
                "chorus_rate": 10,
                "chorus_depth": 0.5,
                "comb_delay": 0.003,
                "comb_feedback": 0.5,
                "phaser_rate": 0.5,
                "pitch_semitones": -2,
                "bit_depth": 10,
                "gain_db": 0
            }
        )
        
        # 16. Ancient God / Divine Voice
        # Powerful, majestic, awe-inspiring
        self._presets[EffectType.ANCIENT_GOD] = EffectPreset(
            name="Ancient God",
            description=(
                "A powerful, majestic voice of divine authority. Combines "
                "deep resonance with heavenly shimmer and vast cosmic reverb "
                "for an awe-inspiring, godlike presence."
            ),
            use_case=(
                "Gods, deities, divine beings, cosmic entities, powerful wizards, "
                "ancient prophets, or any character of supreme magical authority."
            ),
            pedalboard=Pedalboard([
                # Pre-compress to manage dynamics before effects (prevents transients)
                Compressor(threshold_db=-20, ratio=4, attack_ms=1, release_ms=30),
                # Moderate pitch drop for authority
                PitchShift(semitones=-3),
                # Layer with original for power (via short delay, reduced feedback)
                Delay(delay_seconds=0.015, feedback=0.3, mix=0.25),
                # Slow majestic chorus (reduced)
                Chorus(rate_hz=0.1, depth=0.3, mix=0.25),
                # Subtle phaser for divine shimmer (reduced)
                Phaser(rate_hz=0.08, depth=0.3, feedback=0.25, mix=0.15),
                # Massive cosmic reverb
                Reverb(room_size=0.98, damping=0.2, wet_level=0.45, dry_level=0.55, width=1.0),
                # Second reverb for infinite space
                Reverb(room_size=0.85, damping=0.4, wet_level=0.2, dry_level=0.8, width=0.9),
                # Compression for consistent power
                Compressor(threshold_db=-10, ratio=3, attack_ms=10, release_ms=150),
                Gain(gain_db=1),
                # Limiter at the end to catch any transients
                Limiter(threshold_db=-3, release_ms=50)
            ]),
            parameters={
                "pitch_semitones": -3,
                "chorus_rate": 0.1,
                "phaser_rate": 0.08,
                "room_size": 0.98,
                "damping": 0.2,
                "gain_db": 1
            }
        )
        
        # =====================================================================
        # ENVIRONMENT / SITUATION EFFECTS
        # =====================================================================
        
        # 19. Dream Sequence
        # Surreal, floaty, disconnected from reality
        self._presets[EffectType.DREAM_SEQUENCE] = EffectPreset(
            name="Dream Sequence",
            description=(
                "A surreal, floaty quality that suggests the voice is heard "
                "in a dream. Slow modulation, heavy reverb, and filtering "
                "create a disconnected, subconscious atmosphere."
            ),
            use_case=(
                "Dream sequences, visions, prophecies, subconscious dialogue, "
                "hallucinations, or any scene set in an altered state of mind."
            ),
            pedalboard=Pedalboard([
                # Pre-compress to manage dynamics before effects (prevents transients)
                Compressor(threshold_db=-20, ratio=4, attack_ms=1, release_ms=30),
                # Slight high-frequency roll-off for softness
                LowpassFilter(cutoff_frequency_hz=7000),
                # Slow dreamy chorus (reduced)
                Chorus(rate_hz=0.15, depth=0.45, mix=0.35),
                # Slow phaser for surreal swirling (reduced)
                Phaser(rate_hz=0.05, depth=0.4, feedback=0.3, mix=0.25),
                # Long pre-delay for disconnection
                Delay(delay_seconds=0.1, feedback=0.2, mix=0.15),
                # Heavy dreamy reverb
                Reverb(room_size=0.9, damping=0.3, wet_level=0.5, dry_level=0.5, width=1.0),
                # Second layer of reverb
                Reverb(room_size=0.75, damping=0.5, wet_level=0.2, dry_level=0.8, width=0.8),
                Compressor(threshold_db=-15, ratio=3, attack_ms=15, release_ms=150),
                Gain(gain_db=0),
                # Limiter at the end to catch any transients
                Limiter(threshold_db=-3, release_ms=50)
            ]),
            parameters={
                "lowpass_hz": 7000,
                "chorus_rate": 0.15,
                "chorus_depth": 0.45,
                "phaser_rate": 0.05,
                "room_size": 0.9,
                "gain_db": 0
            }
        )
        
        # 20. Stadium / Outdoor Announcement
        # Large outdoor PA with slapback
        self._presets[EffectType.STADIUM_ANNOUNCEMENT] = EffectPreset(
            name="Stadium Announcement",
            description=(
                "A large outdoor announcement sound like a stadium PA system. "
                "Characterized by distinct slapback echoes from distant walls, "
                "broadcast-style compression, and open-air ambience."
            ),
            use_case=(
                "Stadium announcers, outdoor event speakers, fairground "
                "announcements, rally speakers, or large outdoor venue dialogue."
            ),
            pedalboard=Pedalboard([
                # PA system bandwidth
                HighpassFilter(cutoff_frequency_hz=150),
                LowpassFilter(cutoff_frequency_hz=7000),
                # Heavy PA compression
                Compressor(threshold_db=-20, ratio=10, attack_ms=2, release_ms=80),
                # Stadium slapback echoes
                Delay(delay_seconds=0.15, feedback=0.2, mix=0.25),
                Delay(delay_seconds=0.35, feedback=0.15, mix=0.15),
                # Large open reverb (not as enclosed as hall)
                Reverb(room_size=0.8, damping=0.6, wet_level=0.25, dry_level=0.75, width=1.0),
                # Slight saturation for PA character
                Distortion(drive_db=2),
                Gain(gain_db=2)
            ]),
            parameters={
                "highpass_hz": 150,
                "lowpass_hz": 7000,
                "threshold_db": -20,
                "ratio": 10,
                "slapback_1": 0.15,
                "slapback_2": 0.35,
                "room_size": 0.8,
                "gain_db": 2
            }
        )
        
        # =====================================================================
        # TECHNOLOGY EFFECTS
        # =====================================================================
        
        # 23. Megaphone / Loudspeaker
        # Aggressive mid-range, distorted announcement
        self._presets[EffectType.MEGAPHONE] = EffectPreset(
            name="Megaphone",
            description=(
                "A harsh, distorted sound of a voice through a megaphone or "
                "bullhorn. Aggressive mid-range emphasis with clipping and "
                "limited bandwidth for that protest/announcement quality."
            ),
            use_case=(
                "Protest leaders, police commands, emergency announcements, "
                "sports coaches, or any character using a handheld loudspeaker."
            ),
            pedalboard=Pedalboard([
                # Extreme bandwidth limiting
                HighpassFilter(cutoff_frequency_hz=600),
                LowpassFilter(cutoff_frequency_hz=3500),
                # Second pass for aggressive filtering
                HighpassFilter(cutoff_frequency_hz=500),
                LowpassFilter(cutoff_frequency_hz=4000),
                # Heavy compression before distortion
                Compressor(threshold_db=-25, ratio=15, attack_ms=1, release_ms=40),
                # Aggressive distortion for cheap speaker
                Distortion(drive_db=15),
                # Final limiting
                Compressor(threshold_db=-6, ratio=20, attack_ms=0.5, release_ms=20),
                Gain(gain_db=-2)
            ]),
            parameters={
                "highpass_hz": 600,
                "lowpass_hz": 3500,
                "threshold_db": -25,
                "ratio": 15,
                "distortion_db": 15,
                "gain_db": -2
            }
        )
        
        # 24. Old Gramophone / Vintage Recording
        # Scratchy, narrow band, nostalgic
        self._presets[EffectType.OLD_GRAMOPHONE] = EffectPreset(
            name="Old Gramophone",
            description=(
                "A vintage, scratchy quality like an old gramophone or "
                "phonograph recording. Extremely narrow bandwidth with "
                "harmonic distortion and wow/flutter for authentic antique sound."
            ),
            use_case=(
                "Historical recordings, old-time radio dramas, vintage flashbacks, "
                "antique music boxes, or early 20th century period pieces."
            ),
            pedalboard=Pedalboard([
                # Extremely narrow vintage bandwidth
                HighpassFilter(cutoff_frequency_hz=300),
                LowpassFilter(cutoff_frequency_hz=3000),
                # Second pass for authentic narrow band
                HighpassFilter(cutoff_frequency_hz=250),
                LowpassFilter(cutoff_frequency_hz=3500),
                # Bitcrush for low-resolution feel
                Bitcrush(bit_depth=8),
                # Warm tube-like saturation
                Distortion(drive_db=8),
                # Wow and flutter for mechanical imperfection
                Chorus(rate_hz=0.8, depth=0.25, mix=0.3),
                Phaser(rate_hz=0.3, depth=0.2, feedback=0.2, mix=0.15),
                # Very small reverb for phonograph horn
                Reverb(room_size=0.08, damping=0.9, wet_level=0.15, dry_level=0.85, width=0.3),
                Compressor(threshold_db=-15, ratio=5, attack_ms=5, release_ms=80),
                Gain(gain_db=0)
            ]),
            parameters={
                "highpass_hz": 300,
                "lowpass_hz": 3000,
                "bit_depth": 8,
                "distortion_db": 8,
                "wow_flutter_rate": 0.8,
                "gain_db": 0
            }
        )
        
        # 23. Metallic / Helmet Voice
        # Resonant, enclosed, armored
        self._presets[EffectType.METALLIC_HELMET] = EffectPreset(
            name="Metallic Helmet",
            description=(
                "A resonant, enclosed voice as if speaking through a metal "
                "helmet or suit of armor. Metallic resonance with comb filtering "
                "and small enclosed reverb create the armored effect."
            ),
            use_case=(
                "Knights in armor, space suit helmets, diving helmets, "
                "masked villains, motorcycle helmet communication, or "
                "any character with head completely enclosed."
            ),
            pedalboard=Pedalboard([
                # Metallic resonance via comb filters
                Delay(delay_seconds=0.004, feedback=0.65, mix=0.4),
                Delay(delay_seconds=0.007, feedback=0.5, mix=0.3),
                # Helmet bandwidth
                HighpassFilter(cutoff_frequency_hz=200),
                LowpassFilter(cutoff_frequency_hz=6000),
                # Slight phaser for metallic sheen
                Phaser(rate_hz=0.1, depth=0.4, feedback=0.5, mix=0.25),
                # Very small enclosed reverb
                Reverb(room_size=0.05, damping=0.3, wet_level=0.25, dry_level=0.75, width=0.2),
                # Compression for consistent enclosed sound
                Compressor(threshold_db=-15, ratio=5, attack_ms=3, release_ms=60),
                Gain(gain_db=2)
            ]),
            parameters={
                "comb_delay_1": 0.004,
                "comb_delay_2": 0.007,
                "comb_feedback": 0.65,
                "highpass_hz": 200,
                "lowpass_hz": 6000,
                "phaser_rate": 0.1,
                "room_size": 0.05,
                "gain_db": 2
            }
        )
    
    def get_available_effects(self) -> list[EffectType]:
        """Get a list of all available effect types"""
        return list(self._presets.keys())
    
    def get_preset(self, effect_type: EffectType) -> EffectPreset:
        """Get the preset configuration for an effect type"""
        return self._presets[effect_type]
    
    def get_preset_info(self, effect_type: EffectType) -> dict:
        """Get detailed information about a preset"""
        preset = self._presets[effect_type]
        return {
            "name": preset.name,
            "description": preset.description,
            "use_case": preset.use_case,
            "parameters": preset.parameters
        }
    
    def apply_effect(
        self,
        audio: np.ndarray,
        sample_rate: int,
        effect_type: EffectType
    ) -> np.ndarray:
        """
        Apply a preset effect to audio data.
        
        Args:
            audio: Input audio as numpy array (shape: channels x samples or samples)
            sample_rate: Sample rate of the audio in Hz
            effect_type: The type of effect to apply
            
        Returns:
            Processed audio as numpy array
        """
        preset = self._presets[effect_type]
        
        # Ensure audio is in the correct shape (channels, samples) for pedalboard
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        elif audio.ndim == 2 and audio.shape[0] > audio.shape[1]:
            # Likely (samples, channels), transpose to (channels, samples)
            audio = audio.T
        
        # Ensure float32 for pedalboard
        audio = audio.astype(np.float32)
        
        # Apply the effect
        processed = preset.pedalboard(audio, sample_rate)
        
        # Handle any inf/nan values that might be created by effects
        processed = np.nan_to_num(processed, nan=0.0, posinf=0.95, neginf=-0.95)
        
        # Use hard clipping instead of normalization to preserve audio levels
        # Normalization would crush the entire audio if there's one large transient
        processed = np.clip(processed, -0.95, 0.95)
        
        return processed
    
    def create_custom_pedalboard(
        self,
        effect_type: EffectType,
        **custom_params
    ) -> Pedalboard:
        """
        Create a custom pedalboard based on a preset with modified parameters.
        
        This allows fine-tuning of individual effect parameters while maintaining
        the effect structure.
        
        Args:
            effect_type: Base effect type to customize
            **custom_params: Custom parameters to override defaults
            
        Returns:
            Customized Pedalboard instance
        """
        # Get default parameters
        preset = self._presets[effect_type]
        params = {**preset.parameters, **custom_params}
        
        # Rebuild pedalboard based on effect type
        if effect_type == EffectType.PHONE_CALL:
            return Pedalboard([
                HighpassFilter(cutoff_frequency_hz=params.get("highpass_hz", 400)),
                LowpassFilter(cutoff_frequency_hz=params.get("lowpass_hz", 3000)),
                HighpassFilter(cutoff_frequency_hz=params.get("highpass_hz", 400) - 50),
                LowpassFilter(cutoff_frequency_hz=params.get("lowpass_hz", 3000) + 200),
                Distortion(drive_db=params.get("distortion_db", 3)),
                Compressor(
                    threshold_db=params.get("threshold_db", -15),
                    ratio=params.get("ratio", 6),
                    attack_ms=params.get("attack_ms", 2),
                    release_ms=params.get("release_ms", 80)
                ),
                Gain(gain_db=params.get("gain_db", 4))
            ])
        
        elif effect_type == EffectType.RADIO_BROADCAST:
            return Pedalboard([
                HighpassFilter(cutoff_frequency_hz=params.get("highpass_hz", 100)),
                LowpassFilter(cutoff_frequency_hz=params.get("lowpass_hz", 6000)),
                Distortion(drive_db=params.get("distortion_db", 2.5)),
                Compressor(
                    threshold_db=params.get("threshold_db", -18),
                    ratio=params.get("ratio", 8),
                    attack_ms=params.get("attack_ms", 3),
                    release_ms=params.get("release_ms", 100)
                ),
                Reverb(
                    room_size=params.get("room_size", 0.1),
                    damping=params.get("damping", 0.8),
                    wet_level=params.get("wet_level", 0.08),
                    dry_level=params.get("dry_level", 0.92)
                ),
                Gain(gain_db=params.get("gain_db", 3))
            ])
        
        elif effect_type == EffectType.WALKIE_TALKIE:
            return Pedalboard([
                HighpassFilter(cutoff_frequency_hz=params.get("highpass_hz", 500)),
                LowpassFilter(cutoff_frequency_hz=params.get("lowpass_hz", 2500)),
                HighpassFilter(cutoff_frequency_hz=params.get("highpass_hz", 500) - 50),
                LowpassFilter(cutoff_frequency_hz=params.get("lowpass_hz", 2500) + 200),
                Compressor(
                    threshold_db=params.get("threshold_db", -30),
                    ratio=params.get("ratio", 15),
                    attack_ms=params.get("attack_ms", 1),
                    release_ms=params.get("release_ms", 50)
                ),
                Distortion(drive_db=params.get("distortion_db", 20)),
                Bitcrush(bit_depth=params.get("bit_depth", 8)),
                Compressor(threshold_db=-8, ratio=20, attack_ms=0.5, release_ms=30),
                Gain(gain_db=params.get("gain_db", -4))
            ])
        
        elif effect_type == EffectType.REVERB_SMALL_ROOM:
            return Pedalboard([
                Delay(
                    delay_seconds=params.get("pre_delay", 0.008),
                    feedback=0.15,
                    mix=0.1
                ),
                Reverb(
                    room_size=params.get("room_size", 0.15),
                    damping=params.get("damping", 0.7),
                    wet_level=params.get("wet_level", 0.2),
                    dry_level=params.get("dry_level", 0.8),
                    width=params.get("width", 0.6)
                ),
                LowpassFilter(cutoff_frequency_hz=10000),
                Chorus(rate_hz=0.1, depth=0.05, mix=0.05),
                Compressor(threshold_db=-18, ratio=3, attack_ms=5, release_ms=80),
                Gain(gain_db=params.get("gain_db", 0))
            ])
        
        elif effect_type == EffectType.REVERB_LARGE_HALL:
            return Pedalboard([
                Delay(
                    delay_seconds=params.get("pre_delay", 0.04),
                    feedback=0.1,
                    mix=0.08
                ),
                Reverb(
                    room_size=params.get("room_size", 0.95),
                    damping=params.get("damping", 0.3),
                    wet_level=params.get("wet_level", 0.4),
                    dry_level=params.get("dry_level", 0.6),
                    width=params.get("width", 1.0)
                ),
                Reverb(room_size=0.8, damping=0.5, wet_level=0.15, dry_level=0.85, width=0.8),
                LowpassFilter(cutoff_frequency_hz=9000),
                Chorus(rate_hz=0.08, depth=0.03, mix=0.03)
            ])
        
        elif effect_type == EffectType.BAD_SIGNAL:
            return Pedalboard([
                HighpassFilter(cutoff_frequency_hz=params.get("highpass_hz", 200)),
                LowpassFilter(cutoff_frequency_hz=params.get("lowpass_hz", 4000)),
                Bitcrush(bit_depth=params.get("bit_depth", 6)),
                Distortion(drive_db=params.get("distortion_db", 8)),
                Compressor(
                    threshold_db=params.get("threshold_db", -20),
                    ratio=params.get("ratio", 12),
                    attack_ms=params.get("attack_ms", 1),
                    release_ms=params.get("release_ms", 50)
                ),
                Gain(gain_db=params.get("gain_db", -2))
            ])
        
        elif effect_type == EffectType.ROBOTIC_VOICE:
            return Pedalboard([
                PitchShift(semitones=params.get("pitch_semitones", -1.5)),
                Delay(
                    delay_seconds=params.get("comb_delay", 0.005),
                    feedback=params.get("comb_feedback", 0.6),
                    mix=0.4
                ),
                Phaser(
                    rate_hz=params.get("phaser_rate", 0.15),
                    depth=params.get("phaser_depth", 0.9),
                    feedback=params.get("phaser_feedback", 0.75),
                    mix=params.get("phaser_mix", 0.5)
                ),
                Chorus(
                    rate_hz=params.get("chorus_rate", 0.8),
                    depth=params.get("chorus_depth", 0.5),
                    mix=params.get("chorus_mix", 0.4)
                ),
                Bitcrush(bit_depth=params.get("bit_depth", 12)),
                Compressor(threshold_db=-18, ratio=6, attack_ms=2, release_ms=60),
                Gain(gain_db=params.get("gain_db", 2))
            ])
        
        elif effect_type == EffectType.MONSTER_VOICE:
            return Pedalboard([
                PitchShift(semitones=params.get("pitch_semitones", -7)),
                Delay(delay_seconds=0.025, feedback=0.3, mix=0.2),
                Distortion(drive_db=params.get("distortion_db", 18)),
                LowpassFilter(cutoff_frequency_hz=6000),
                Chorus(
                    rate_hz=params.get("chorus_rate", 0.4),
                    depth=params.get("chorus_depth", 0.4),
                    mix=params.get("chorus_mix", 0.45)
                ),
                Phaser(
                    rate_hz=params.get("phaser_rate", 0.2),
                    depth=params.get("phaser_depth", 0.6),
                    feedback=0.5,
                    mix=0.3
                ),
                Reverb(
                    room_size=params.get("room_size", 0.9),
                    damping=params.get("damping", 0.95),
                    wet_level=params.get("wet_level", 0.4),
                    dry_level=params.get("dry_level", 0.6),
                    width=1.0
                ),
                Compressor(threshold_db=-12, ratio=4, attack_ms=5, release_ms=100),
                Gain(gain_db=params.get("gain_db", -2))
            ])
        
        elif effect_type == EffectType.INTIMATE_WHISPER:
            return Pedalboard([
                HighpassFilter(cutoff_frequency_hz=params.get("highpass_hz", 200)),
                HighpassFilter(cutoff_frequency_hz=150),  # Second pass for steeper rolloff
                LowpassFilter(cutoff_frequency_hz=params.get("lowpass_hz", 8000)),
                Compressor(
                    threshold_db=params.get("threshold_db", -20),
                    ratio=params.get("ratio", 4),
                    attack_ms=params.get("attack_ms", 5),
                    release_ms=params.get("release_ms", 100)
                ),
                Reverb(
                    room_size=params.get("room_size", 0.05),
                    damping=params.get("damping", 0.95),
                    wet_level=params.get("wet_level", 0.12),
                    dry_level=params.get("dry_level", 0.88),
                    width=params.get("width", 0.3)
                ),
                Gain(gain_db=params.get("gain_db", -2))
            ])
        
        elif effect_type == EffectType.UNDERWATER:
            return Pedalboard([
                LowpassFilter(cutoff_frequency_hz=params.get("lowpass_hz", 800)),
                LowpassFilter(cutoff_frequency_hz=1200),
                Chorus(
                    rate_hz=params.get("chorus_rate", 0.3),
                    depth=params.get("chorus_depth", 0.6),
                    mix=params.get("chorus_mix", 0.4)
                ),
                Phaser(
                    rate_hz=params.get("phaser_rate", 0.15),
                    depth=params.get("phaser_depth", 0.5),
                    feedback=params.get("phaser_feedback", 0.4),
                    mix=params.get("phaser_mix", 0.35)
                ),
                Delay(delay_seconds=0.02, feedback=0.3, mix=0.25),
                Reverb(
                    room_size=params.get("room_size", 0.6),
                    damping=params.get("damping", 0.8),
                    wet_level=params.get("wet_level", 0.35),
                    dry_level=params.get("dry_level", 0.65),
                    width=params.get("width", 0.9)
                ),
                Compressor(threshold_db=-18, ratio=4, attack_ms=10, release_ms=100),
                Gain(gain_db=params.get("gain_db", 3))
            ])
        
        elif effect_type == EffectType.GHOST_SPIRIT:
            return Pedalboard([
                HighpassFilter(cutoff_frequency_hz=params.get("highpass_hz", 300)),
                PitchShift(semitones=params.get("pitch_semitones", 2)),
                Chorus(
                    rate_hz=params.get("chorus_rate", 0.2),
                    depth=params.get("chorus_depth", 0.7),
                    mix=params.get("chorus_mix", 0.5)
                ),
                Phaser(
                    rate_hz=params.get("phaser_rate", 0.1),
                    depth=params.get("phaser_depth", 0.8),
                    feedback=params.get("phaser_feedback", 0.6),
                    mix=params.get("phaser_mix", 0.4)
                ),
                Reverb(
                    room_size=params.get("room_size", 0.85),
                    damping=params.get("damping", 0.2),
                    wet_level=params.get("wet_level", 0.55),
                    dry_level=params.get("dry_level", 0.45),
                    width=1.0
                ),
                Reverb(room_size=0.7, damping=0.4, wet_level=0.2, dry_level=0.8, width=0.8),
                Delay(
                    delay_seconds=params.get("delay_seconds", 0.3),
                    feedback=params.get("delay_feedback", 0.35),
                    mix=params.get("delay_mix", 0.2)
                ),
                Compressor(threshold_db=-15, ratio=3, attack_ms=15, release_ms=150),
                Gain(gain_db=params.get("gain_db", -1))
            ])
        
        elif effect_type == EffectType.GIANT_TITAN:
            return Pedalboard([
                PitchShift(semitones=params.get("pitch_semitones", -5)),
                HighpassFilter(cutoff_frequency_hz=40),
                LowpassFilter(cutoff_frequency_hz=8000),
                Distortion(drive_db=params.get("distortion_db", 4)),
                Chorus(
                    rate_hz=params.get("chorus_rate", 0.15),
                    depth=params.get("chorus_depth", 0.3),
                    mix=params.get("chorus_mix", 0.25)
                ),
                Delay(delay_seconds=0.03, feedback=0.2, mix=0.2),
                Reverb(
                    room_size=params.get("room_size", 0.95),
                    damping=params.get("damping", 0.4),
                    wet_level=params.get("wet_level", 0.35),
                    dry_level=params.get("dry_level", 0.65),
                    width=1.0
                ),
                Compressor(threshold_db=-12, ratio=3, attack_ms=20, release_ms=200),
                Gain(gain_db=params.get("gain_db", 2))
            ])
        
        elif effect_type == EffectType.FAIRY_TINY:
            return Pedalboard([
                PitchShift(semitones=params.get("pitch_semitones", 3)),
                HighpassFilter(cutoff_frequency_hz=params.get("highpass_hz", 250)),
                LowpassFilter(cutoff_frequency_hz=12000),
                Chorus(
                    rate_hz=params.get("chorus_rate", 2.0),
                    depth=params.get("chorus_depth", 0.25),
                    mix=params.get("chorus_mix", 0.25)
                ),
                Phaser(
                    rate_hz=params.get("phaser_rate", 1.0),
                    depth=params.get("phaser_depth", 0.2),
                    feedback=0.2,
                    mix=0.15
                ),
                Reverb(
                    room_size=params.get("room_size", 0.1),
                    damping=params.get("damping", 0.5),
                    wet_level=params.get("wet_level", 0.15),
                    dry_level=params.get("dry_level", 0.85),
                    width=0.5
                ),
                Compressor(threshold_db=-15, ratio=3, attack_ms=3, release_ms=50),
                Gain(gain_db=params.get("gain_db", 2))
            ])
        
        elif effect_type == EffectType.ALIEN_VOICE:
            return Pedalboard([
                Chorus(
                    rate_hz=params.get("chorus_rate", 15),
                    depth=params.get("chorus_depth", 0.8),
                    mix=params.get("chorus_mix", 0.5)
                ),
                Delay(
                    delay_seconds=params.get("comb_delay", 0.003),
                    feedback=params.get("comb_feedback", 0.7),
                    mix=0.45
                ),
                Phaser(
                    rate_hz=params.get("phaser_rate", 0.5),
                    depth=params.get("phaser_depth", 0.9),
                    feedback=params.get("phaser_feedback", 0.8),
                    mix=0.6
                ),
                PitchShift(semitones=params.get("pitch_semitones", -2)),
                Bitcrush(bit_depth=params.get("bit_depth", 10)),
                Distortion(drive_db=params.get("distortion_db", 3)),
                Reverb(room_size=0.3, damping=0.6, wet_level=0.25, dry_level=0.75, width=0.7),
                Compressor(threshold_db=-15, ratio=5, attack_ms=2, release_ms=60),
                Gain(gain_db=params.get("gain_db", 0))
            ])
        
        elif effect_type == EffectType.ANCIENT_GOD:
            return Pedalboard([
                PitchShift(semitones=params.get("pitch_semitones", -3)),
                Delay(delay_seconds=0.015, feedback=0.4, mix=0.3),
                Chorus(
                    rate_hz=params.get("chorus_rate", 0.1),
                    depth=params.get("chorus_depth", 0.4),
                    mix=params.get("chorus_mix", 0.3)
                ),
                Phaser(
                    rate_hz=params.get("phaser_rate", 0.08),
                    depth=params.get("phaser_depth", 0.4),
                    feedback=0.3,
                    mix=0.2
                ),
                Reverb(
                    room_size=params.get("room_size", 0.98),
                    damping=params.get("damping", 0.2),
                    wet_level=params.get("wet_level", 0.45),
                    dry_level=params.get("dry_level", 0.55),
                    width=1.0
                ),
                Reverb(room_size=0.85, damping=0.4, wet_level=0.2, dry_level=0.8, width=0.9),
                Compressor(threshold_db=-10, ratio=3, attack_ms=10, release_ms=150),
                Gain(gain_db=params.get("gain_db", 1))
            ])
        
        elif effect_type == EffectType.DREAM_SEQUENCE:
            return Pedalboard([
                LowpassFilter(cutoff_frequency_hz=params.get("lowpass_hz", 7000)),
                Chorus(
                    rate_hz=params.get("chorus_rate", 0.15),
                    depth=params.get("chorus_depth", 0.6),
                    mix=params.get("chorus_mix", 0.45)
                ),
                Phaser(
                    rate_hz=params.get("phaser_rate", 0.05),
                    depth=params.get("phaser_depth", 0.5),
                    feedback=0.4,
                    mix=0.35
                ),
                Delay(delay_seconds=0.1, feedback=0.25, mix=0.2),
                Reverb(
                    room_size=params.get("room_size", 0.9),
                    damping=params.get("damping", 0.3),
                    wet_level=params.get("wet_level", 0.5),
                    dry_level=params.get("dry_level", 0.5),
                    width=1.0
                ),
                Reverb(room_size=0.75, damping=0.5, wet_level=0.2, dry_level=0.8, width=0.8),
                Compressor(threshold_db=-15, ratio=3, attack_ms=15, release_ms=150),
                Gain(gain_db=params.get("gain_db", 0))
            ])
        
        elif effect_type == EffectType.STADIUM_ANNOUNCEMENT:
            return Pedalboard([
                HighpassFilter(cutoff_frequency_hz=params.get("highpass_hz", 150)),
                LowpassFilter(cutoff_frequency_hz=params.get("lowpass_hz", 7000)),
                Compressor(
                    threshold_db=params.get("threshold_db", -20),
                    ratio=params.get("ratio", 10),
                    attack_ms=2,
                    release_ms=80
                ),
                Delay(
                    delay_seconds=params.get("slapback_1", 0.15),
                    feedback=0.2,
                    mix=0.25
                ),
                Delay(
                    delay_seconds=params.get("slapback_2", 0.35),
                    feedback=0.15,
                    mix=0.15
                ),
                Reverb(
                    room_size=params.get("room_size", 0.8),
                    damping=params.get("damping", 0.6),
                    wet_level=params.get("wet_level", 0.25),
                    dry_level=params.get("dry_level", 0.75),
                    width=1.0
                ),
                Distortion(drive_db=params.get("distortion_db", 2)),
                Gain(gain_db=params.get("gain_db", 2))
            ])
        
        elif effect_type == EffectType.MEGAPHONE:
            return Pedalboard([
                HighpassFilter(cutoff_frequency_hz=params.get("highpass_hz", 600)),
                LowpassFilter(cutoff_frequency_hz=params.get("lowpass_hz", 3500)),
                HighpassFilter(cutoff_frequency_hz=500),
                LowpassFilter(cutoff_frequency_hz=4000),
                Compressor(
                    threshold_db=params.get("threshold_db", -25),
                    ratio=params.get("ratio", 15),
                    attack_ms=1,
                    release_ms=40
                ),
                Distortion(drive_db=params.get("distortion_db", 15)),
                Compressor(threshold_db=-6, ratio=20, attack_ms=0.5, release_ms=20),
                Gain(gain_db=params.get("gain_db", -2))
            ])
        
        elif effect_type == EffectType.OLD_GRAMOPHONE:
            return Pedalboard([
                HighpassFilter(cutoff_frequency_hz=params.get("highpass_hz", 300)),
                LowpassFilter(cutoff_frequency_hz=params.get("lowpass_hz", 3000)),
                HighpassFilter(cutoff_frequency_hz=250),
                LowpassFilter(cutoff_frequency_hz=3500),
                Bitcrush(bit_depth=params.get("bit_depth", 8)),
                Distortion(drive_db=params.get("distortion_db", 8)),
                Chorus(
                    rate_hz=params.get("wow_flutter_rate", 0.8),
                    depth=params.get("wow_flutter_depth", 0.25),
                    mix=params.get("wow_flutter_mix", 0.3)
                ),
                Phaser(rate_hz=0.3, depth=0.2, feedback=0.2, mix=0.15),
                Reverb(room_size=0.08, damping=0.9, wet_level=0.15, dry_level=0.85, width=0.3),
                Compressor(threshold_db=-15, ratio=5, attack_ms=5, release_ms=80),
                Gain(gain_db=params.get("gain_db", 0))
            ])
        
        elif effect_type == EffectType.METALLIC_HELMET:
            return Pedalboard([
                Delay(
                    delay_seconds=params.get("comb_delay_1", 0.004),
                    feedback=params.get("comb_feedback", 0.65),
                    mix=0.4
                ),
                Delay(
                    delay_seconds=params.get("comb_delay_2", 0.007),
                    feedback=0.5,
                    mix=0.3
                ),
                HighpassFilter(cutoff_frequency_hz=params.get("highpass_hz", 200)),
                LowpassFilter(cutoff_frequency_hz=params.get("lowpass_hz", 6000)),
                Phaser(
                    rate_hz=params.get("phaser_rate", 0.1),
                    depth=params.get("phaser_depth", 0.4),
                    feedback=0.5,
                    mix=0.25
                ),
                Reverb(
                    room_size=params.get("room_size", 0.05),
                    damping=params.get("damping", 0.3),
                    wet_level=params.get("wet_level", 0.25),
                    dry_level=params.get("dry_level", 0.75),
                    width=0.2
                ),
                Compressor(threshold_db=-15, ratio=5, attack_ms=3, release_ms=60),
                Gain(gain_db=params.get("gain_db", 2))
            ])
        
        # Default: return the original preset's pedalboard
        return preset.pedalboard
    
    def apply_custom_effect(
        self,
        audio: np.ndarray,
        sample_rate: int,
        effect_type: EffectType,
        **custom_params
    ) -> np.ndarray:
        """
        Apply a customized effect with modified parameters.
        
        Args:
            audio: Input audio as numpy array
            sample_rate: Sample rate in Hz
            effect_type: Base effect type
            **custom_params: Custom parameters to override
            
        Returns:
            Processed audio as numpy array
        """
        custom_board = self.create_custom_pedalboard(effect_type, **custom_params)
        
        # Ensure audio is in the correct shape
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        elif audio.ndim == 2 and audio.shape[0] > audio.shape[1]:
            audio = audio.T
        
        audio = audio.astype(np.float32)
        
        return custom_board(audio, sample_rate)
    
    def add_signal_dropouts(
        self,
        audio: np.ndarray,
        sample_rate: int,
        dropout_probability: float = 0.2,
        chunk_duration_ms: float = 100
    ) -> np.ndarray:
        """
        Add random signal dropouts to simulate a failing connection.
        
        Args:
            audio: Input audio array
            sample_rate: Sample rate in Hz
            dropout_probability: Probability of dropout per chunk (0-1)
            chunk_duration_ms: Duration of each chunk in milliseconds
            
        Returns:
            Audio with random dropouts
        """
        # Ensure correct shape
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        elif audio.ndim == 2 and audio.shape[0] > audio.shape[1]:
            audio = audio.T
        
        audio = audio.copy().astype(np.float32)
        
        chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        num_samples = audio.shape[1]
        
        for i in range(0, num_samples, chunk_size):
            if np.random.rand() < dropout_probability:
                end_idx = min(i + chunk_size, num_samples)
                audio[:, i:end_idx] *= 0.01  # Nearly silent
        
        return audio


# Convenience function for quick effect application
def apply_effect(
    audio: np.ndarray,
    sample_rate: int,
    effect_name: str
) -> np.ndarray:
    """
    Quick function to apply a named effect to audio.
    
    Args:
        audio: Input audio array
        sample_rate: Sample rate in Hz
        effect_name: Name of the effect (e.g., "phone_call", "monster_voice")
        
    Returns:
        Processed audio
    """
    applier = SoundEffectApplier()
    effect_type = EffectType(effect_name)
    return applier.apply_effect(audio, sample_rate, effect_type)

