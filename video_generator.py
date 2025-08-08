import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, VitsModel, AutoTokenizer
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, EulerDiscreteScheduler
import soundfile as sf
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, CompositeAudioClip, VideoFileClip
from moviepy.video.fx.all import fadein, fadeout, speedx
from moviepy.audio.fx.all import volumex
from moviepy.audio.AudioClip import AudioClip
import tempfile
import requests
from bs4 import BeautifulSoup
import re
import warnings
from typing import List, Dict, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
import logging
from enum import Enum
import functools
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import hashlib
from scipy.interpolate import interp1d
import json
import gc
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Enum for different model types available for each component"""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    XL = "xl"

@dataclass
class GenerationConfig:
    """Configuration for video generation parameters"""
    video_length: float = 30.0  # Target video length in seconds
    min_scene_duration: float = 3.0  # Minimum duration per scene
    max_scene_duration: float = 10.0  # Maximum duration per scene
    transition_duration: float = 0.5  # Duration of transitions between scenes
    image_width: int = 768
    image_height: int = 768
    image_steps: int = 25
    image_cfg_scale: float = 7.5
    tts_sampling_rate: int = 16000
    text_model_temp: float = 0.7
    seed: Optional[int] = None
    enable_safety_checker: bool = False
    max_workers: int = 4
    enable_animation: bool = True
    enable_character_consistency: bool = True
    enable_background_music: bool = True
    music_volume: float = 0.2
    max_video_duration: float = 600.0  # 10 minute safety limit
    target_audio_db: float = -20.0  # Target loudness level

@dataclass
class ModelConfig:
    """Configuration for model selection"""
    text_model: ModelType = ModelType.MEDIUM
    tts_model: ModelType = ModelType.MEDIUM
    image_model: ModelType = ModelType.MEDIUM

class FreeTextToVideoGenerator:
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        model_config: ModelConfig = ModelConfig(),
        generation_config: GenerationConfig = GenerationConfig()
    ):
        self.device = device
        self.model_config = model_config
        self.generation_config = generation_config
        self.temp_dir = tempfile.mkdtemp()
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Temporary directory: {self.temp_dir}")
        
        # Initialize model containers (lazy loading)
        self._text_generator = None
        self._tts_model = None
        self._tts_tokenizer = None
        self._image_generator = None
        self._img2img_generator = None
        
        # Font for text overlay
        self.font = self._load_font()
        
        # Image cache with LRU eviction
        self._image_cache = {}
        self._image_cache_size = 20  # Keep last 20 images
        
        # Character reference for consistency
        self._character_reference = None
        
        # Validate configuration
        self._validate_config()
    
    def __del__(self):
        """Clean up resources when instance is garbage collected"""
        self.cleanup()
        if hasattr(self, '_image_generator'):
            del self._image_generator
            self._image_generator = None
        if hasattr(self, '_img2img_generator'):
            del self._img2img_generator
            self._img2img_generator = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        if self.generation_config.min_scene_duration >= self.generation_config.max_scene_duration:
            raise ValueError("min_scene_duration must be less than max_scene_duration")
        
        if self.generation_config.music_volume < 0 or self.generation_config.music_volume > 1:
            raise ValueError("music_volume must be between 0.0 and 1.0")
        
        if self.generation_config.max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        
        if self.generation_config.max_video_duration > 3600:
            raise ValueError("max_video_duration cannot exceed 3600 seconds (1 hour)")
    
    @property
    def text_generator(self):
        """Lazy load text generation model"""
        if self._text_generator is None:
            self._load_text_model()
        return self._text_generator
    
    @property
    def tts_model(self):
        """Lazy load TTS model"""
        if self._tts_model is None:
            self._load_tts_model()
        return self._tts_model
    
    @property
    def tts_tokenizer(self):
        """Lazy load TTS tokenizer"""
        if self._tts_tokenizer is None:
            self._load_tts_model()
        return self._tts_tokenizer
    
    @property
    def image_generator(self):
        """Lazy load image generation model"""
        if self._image_generator is None:
            self._load_image_model()
        return self._image_generator
    
    @property
    def img2img_generator(self):
        """Lazy load img2img model"""
        if self._img2img_generator is None:
            self._load_img2img_model()
        return self._img2img_generator
    
    def _load_text_model(self):
        """Load text generation model based on configuration"""
        model_map = {
            ModelType.SMALL: "google/flan-t5-small",
            ModelType.MEDIUM: "google/flan-t5-base",
            ModelType.LARGE: "google/flan-t5-large",
            ModelType.XL: "google/flan-t5-xl"
        }
        
        model_name = model_map[self.model_config.text_model]
        logger.info(f"Loading text generation model: {model_name}")
        
        self._text_generator = pipeline(
            'text2text-generation',
            model=model_name,
            device=0 if self.device == "cuda" else -1
        )
    
    def _load_tts_model(self):
        """Load TTS model based on configuration"""
        model_map = {
            ModelType.SMALL: "facebook/mms-tts-eng",
            ModelType.MEDIUM: "facebook/mms-tts-eng",
            ModelType.LARGE: "facebook/vits-tts",
            ModelType.XL: "facebook/mms-tts-eng"  # No XL version available, using medium
        }
        
        model_name = model_map[self.model_config.tts_model]
        logger.info(f"Loading TTS model: {model_name}")
        
        self._tts_model = VitsModel.from_pretrained(model_name).to(self.device)
        self._tts_tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def _load_image_model(self):
        """Load image generation model based on configuration"""
        model_map = {
            ModelType.SMALL: "CompVis/stable-diffusion-v1-4",
            ModelType.MEDIUM: "runwayml/stable-diffusion-v1-5",
            ModelType.LARGE: "stabilityai/stable-diffusion-2-1",
            ModelType.XL: "stabilityai/stable-diffusion-xl-base-1.0"
        }
        
        model_name = model_map[self.model_config.image_model]
        logger.info(f"Loading image generation model: {model_name}")
        
        # Use Euler scheduler for better quality
        scheduler = EulerDiscreteScheduler.from_pretrained(model_name, subfolder="scheduler")
        
        self._image_generator = StableDiffusionPipeline.from_pretrained(
            model_name,
            scheduler=scheduler,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None if not self.generation_config.enable_safety_checker else "stabilityai/stable-diffusion-2-1-safety-checker"
        ).to(self.device)
        
        # Enable memory optimizations
        if hasattr(self._image_generator, "enable_xformers_memory_efficient_attention"):
            self._image_generator.enable_xformers_memory_efficient_attention()
        if hasattr(self._image_generator, "enable_model_cpu_offload"):
            self._image_generator.enable_model_cpu_offload()
    
    def _load_img2img_model(self):
        """Load img2img model for character consistency"""
        logger.info("Loading img2img model for character consistency")
        
        self._img2img_generator = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        
        if hasattr(self._img2img_generator, "enable_xformers_memory_efficient_attention"):
            self._img2img_generator.enable_xformers_memory_efficient_attention()
        if hasattr(self._img2img_generator, "enable_model_cpu_offload"):
            self._img2img_generator.enable_model_cpu_offload()
    
    def _load_font(self):
        """Try to load a nice font, fallback to default"""
        font_paths = [
            "arial.ttf",
            "Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/Library/Fonts/Arial.ttf"
        ]
        
        for path in font_paths:
            try:
                return ImageFont.truetype(path, 40)
            except:
                continue
        return ImageFont.load_default()
    
    def generate_script(
        self,
        prompt: str,
        max_length: int = 500,
        num_scenes: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """Generate a video script from text prompt with controlled scene count"""
        if num_scenes is None:
            # Calculate target number of scenes based on desired video length
            avg_scene_duration = (
                self.generation_config.min_scene_duration +
                self.generation_config.max_scene_duration
            ) / 2
            num_scenes = max(3, min(8, int(self.generation_config.video_length / avg_scene_duration)))
        
        # Create a more detailed prompt for script generation
        detailed_prompt = (
            f"Create a {num_scenes}-scene video script about: {prompt}.\n"
            "Each scene should have:\n"
            "1. A detailed visual description (what to show)\n"
            "2. Concise narration text (what to say)\n"
            "Format exactly as:\n"
            "Scene 1: [visual description] | [narration text]\n"
            "Scene 2: [visual description] | [narration text]\n"
            "..."
        )
        
        try:
            result = self.text_generator(
                detailed_prompt,
                max_length=max_length,
                num_return_sequences=1,
                temperature=self.generation_config.text_model_temp,
                do_sample=True
            )
            
            script = result[0]['generated_text']
            return self._parse_script(script)
        except Exception as e:
            logger.error(f"Script generation failed: {str(e)}")
            # Fallback to simple scenes
            return [{
                "visual": prompt,
                "narration": prompt[:100] + "..." if len(prompt) > 100 else prompt
            }]
    
    def _parse_script(self, script_text: str) -> List[Dict[str, str]]:
        """Parse generated script into structured scenes with improved error handling"""
        scenes = []
        
        # Try pipe separator format first
        if "|" in script_text:
            scene_lines = [line.strip() for line in script_text.split("\n") if line.strip()]
            for line in scene_lines:
                if ":" in line and "|" in line:
                    scene_num, rest = line.split(":", 1)
                    visual, narration = rest.split("|", 1)
                    scenes.append({
                        "visual": visual.strip(),
                        "narration": narration.strip()
                    })
            if scenes:
                return scenes
        
        # Fallback to regex parsing
        scene_pattern = r"Scene \d+:\s*(.*?)\s*(.*?)(?=(?:Scene \d+:|$))"
        matches = re.findall(scene_pattern, script_text, re.DOTALL)
        
        if matches:
            for visual, narration in matches:
                scenes.append({
                    "visual": visual.strip(),
                    "narration": narration.strip()
                })
        else:
            # Final fallback: simple split by sentences
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', script_text)
            for i, sentence in enumerate(sentences[:8]):  # Limit to 8 scenes max
                scenes.append({
                    "visual": f"Scene {i+1}: {sentence.strip()}",
                    "narration": sentence.strip()
                })
        
        return scenes
    
    @functools.lru_cache(maxsize=32)
    def generate_speech(self, text: str, output_path: str) -> str:
        """Convert text to speech using TTS model with caching and error handling"""
        try:
            inputs = self.tts_tokenizer(text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                output = self.tts_model(**inputs).waveform
            
            # Convert to numpy and save
            audio = output.cpu().numpy().squeeze()
            sf.write(
                output_path,
                audio,
                samplerate=self.generation_config.tts_sampling_rate
            )
            return output_path
        except Exception as e:
            logger.error(f"TTS generation failed: {str(e)}")
            # Create silent audio as fallback
            duration = min(
                max(
                    len(text.split()) / 3,  # Approximate 3 words per second
                    self.generation_config.min_scene_duration
                ),
                self.generation_config.max_scene_duration
            )
            silent_audio = np.zeros(int(duration * self.generation_config.tts_sampling_rate))
            sf.write(output_path, silent_audio, samplerate=self.generation_config.tts_sampling_rate)
            return output_path
    
    def generate_image(
        self,
        prompt: str,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        use_cache: bool = True
    ) -> Image.Image:
        """Generate an image from text prompt with enhanced controls and caching"""
        # Create cache key
        cache_key = hashlib.md5(f"{prompt}_{seed}_{negative_prompt}".encode()).hexdigest()
        
        if use_cache and cache_key in self._image_cache:
            logger.debug(f"Using cached image for prompt: {prompt[:50]}...")
            return self._image_cache[cache_key]
        
        if seed is None and self.generation_config.seed is not None:
            seed = self.generation_config.seed + hash(prompt) % 1000
        
        if seed:
            torch.manual_seed(seed)
        
        # Add style to the prompt for better results
        styled_prompt = (
            f"{prompt}, high quality, cinematic lighting, detailed, 8k, "
            "professional photography, ultra realistic"
        )
        
        # Default negative prompt
        if negative_prompt is None:
            negative_prompt = (
                "blurry, low quality, cartoon, anime, deformed, distorted, "
                "extra limbs, disfigured"
            )
        
        try:
            image = self.image_generator(
                styled_prompt,
                negative_prompt=negative_prompt,
                height=self.generation_config.image_height,
                width=self.generation_config.image_width,
                num_inference_steps=self.generation_config.image_steps,
                guidance_scale=self.generation_config.image_cfg_scale
            ).images[0]
            
            # Cache the image
            if use_cache:
                # Manage cache size
                if len(self._image_cache) >= self._image_cache_size:
                    # Remove oldest entry
                    oldest_key = next(iter(self._image_cache))
                    del self._image_cache[oldest_key]
                
                self._image_cache[cache_key] = image
            
            return image
        except Exception as e:
            logger.error(f"Image generation failed: {str(e)}")
            # Return black image as fallback
            return Image.new(
                'RGB',
                (self.generation_config.image_width, self.generation_config.image_height),
                color='black'
            )
    
    def generate_consistent_character(self, character_description: str) -> Image.Image:
        """Generate a consistent character reference image"""
        if self._character_reference is None:
            # Generate base character image
            self._character_reference = self.generate_image(
                f"Full body portrait of {character_description}, detailed face, neutral expression",
                seed=42  # Fixed seed for consistency
            )
        return self._character_reference
    
    def generate_character_scene(
        self,
        scene_description: str,
        character_reference: Image.Image,
        seed: Optional[int] = None
    ) -> Image.Image:
        """Generate a scene with consistent character"""
        if seed is None and self.generation_config.seed is not None:
            seed = self.generation_config.seed + hash(scene_description) % 1000
        
        if seed:
            torch.manual_seed(seed)
        
        try:
            image = self.img2img_generator(
                prompt=scene_description,
                image=character_reference,
                strength=0.75,  # Balance between new content and character
                guidance_scale=7.5,
                num_inference_steps=20
            ).images[0]
            return image
        except Exception as e:
            logger.error(f"Character scene generation failed: {str(e)}")
            return character_reference
    
    def generate_keyframes(
        self,
        prompt: str,
        num_frames: int = 5,
        seed: Optional[int] = None
    ) -> List[Image.Image]:
        """Generate keyframes with prompt variations and parallel processing"""
        if num_frames < 2:
            raise ValueError("num_frames must be at least 2")
        
        # Create varied prompts for each frame
        prompts = [
            f"{prompt}, frame {i+1}/{num_frames}, " + 
            f"camera angle: {['wide','medium','close-up'][i%3]}, " +
            f"time: {['morning','afternoon','evening','night'][i%4]}"
            for i in range(num_frames)
        ]
        
        # Generate frames in parallel
        with ThreadPoolExecutor(
            max_workers=min(num_frames, self.generation_config.max_workers)
        ) as executor:
            futures = {
                executor.submit(
                    self.generate_image,
                    p,
                    seed=seed + i if seed else None,
                    use_cache=False
                ): i for i, p in enumerate(prompts)
            }
            
            # Collect results in order
            frames = [None] * num_frames
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    frames[idx] = future.result()
                except Exception as e:
                    logger.error(f"Keyframe generation failed for frame {idx}: {str(e)}")
                    # Fallback to first frame if generation fails
                    frames[idx] = frames[0] if idx > 0 and frames[0] else Image.new(
                        'RGB',
                        (self.generation_config.image_width, self.generation_config.image_height),
                        color='black'
                    )
            
            return frames
    
    def interpolate_frames(
        self,
        keyframes: List[Image.Image],
        duration: float,
        fps: int = 24
    ) -> List[Image.Image]:
        """Interpolate between keyframes to create smooth animation"""
        if len(keyframes) < 2:
            return keyframes
        
        # Calculate number of frames needed
        total_frames = int(duration * fps)
        frames_per_segment = total_frames // (len(keyframes) - 1)
        
        interpolated_frames = []
        
        for i in range(len(keyframes) - 1):
            start_frame = np.array(keyframes[i])
            end_frame = np.array(keyframes[i + 1])
            
            # Create interpolation function for each color channel
            interp_functions = []
            for c in range(3):  # RGB channels
                interp_func = interp1d(
                    [0, frames_per_segment],
                    [start_frame[:, :, c].flatten(), end_frame[:, :, c].flatten()],
                    kind='linear',
                    axis=0
                )
                interp_functions.append(interp_func)
            
            # Generate interpolated frames
            for j in range(frames_per_segment):
                interpolated = np.zeros_like(start_frame)
                for c in range(3):
                    channel_data = interp_functions[c](j).reshape(start_frame[:, :, c].shape)
                    interpolated[:, :, c] = channel_data
                
                interpolated_frames.append(Image.fromarray(interpolated.astype('uint8')))
        
        return interpolated_frames
    
    def generate_background_music(self, mood: str, duration: float) -> np.ndarray:
        """Generate background music (simulated with sine waves)"""
        # This is a simplified version - in a real implementation, you would use a music generation model
        sample_rate = 22050
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # Generate different tones based on mood
        if "happy" in mood.lower():
            frequencies = [440, 554.37, 659.25]  # A major chord
        elif "sad" in mood.lower():
            frequencies = [220, 246.94, 293.66]  # A minor chord
        elif "exciting" in mood.lower():
            frequencies = [523.25, 659.25, 783.99]  # C major chord
        else:
            frequencies = [440, 554.37, 659.25]  # Default to A major
        
        # Generate the music
        music = np.zeros_like(t)
        for freq in frequencies:
            music += np.sin(2 * np.pi * freq * t) * 0.3
        
        # Add some variation
        envelope = np.exp(-t / (duration / 3))  # Fade out
        music = music * envelope
        
        # Normalize
        music = music / np.max(np.abs(music)) if np.max(np.abs(music)) > 0 else music
        
        return music
    
    def create_text_overlay(self, image: Image.Image, text: str) -> Image.Image:
        """Add text overlay to image with improved layout"""
        draw = ImageDraw.Draw(image)
        
        # Split long text into multiple lines
        max_line_length = 40
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            if len(' '.join(current_line + [word])) <= max_line_length:
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
        if current_line:
            lines.append(' '.join(current_line))
        
        text = '\n'.join(lines[:3])  # Max 3 lines
        
        # Calculate text position (centered at bottom)
        text_height = sum([self.font.getsize(line)[1] for line in text.split('\n')])
        position = (20, image.height - text_height - 20)
        
        # Add semi-transparent background for better readability
        bg_padding = 10
        bg_coords = (
            position[0] - bg_padding,
            position[1] - bg_padding,
            image.width - 20 + bg_padding,
            image.height - 20 + bg_padding
        )
        bg = Image.new('RGBA', image.size, (0, 0, 0, 0))
        bg_draw = ImageDraw.Draw(bg)
        bg_draw.rectangle(bg_coords, fill=(0, 0, 0, 128))
        image = Image.alpha_composite(image.convert('RGBA'), bg).convert('RGB')
        
        # Recreate draw object after alpha composition
        draw = ImageDraw.Draw(image)
        
        # Add text with shadow
        shadow_offset = 2
        for i, line in enumerate(text.split('\n')):
            y = position[1] + (i * self.font.getsize(line)[1])
            draw.text((position[0]+shadow_offset, y+shadow_offset), line, font=self.font, fill="black")
            draw.text((position[0], y), line, font=self.font, fill="white")
        
        return image
    
    def _calculate_scene_duration(self, narration: str) -> float:
        """Calculate appropriate duration for a scene based on its content"""
        # Base duration on word count (average 3 words per second)
        word_count = len(narration.split())
        base_duration = word_count / 3
        
        # Constrain within min/max bounds
        return min(
            max(
                base_duration,
                self.generation_config.min_scene_duration
            ),
            self.generation_config.max_scene_duration
        )
    
    def _normalize_audio(self, audio_clip: AudioFileClip) -> AudioFileClip:
        """Normalize audio levels to target dB level"""
        try:
            # Calculate current loudness (RMS)
            max_volume = audio_clip.max_volume()
            
            if max_volume < 0.001:  # Silent audio
                return audio_clip
            
            # Calculate desired gain (simplified loudness normalization)
            target_linear = 10 ** (self.generation_config.target_audio_db / 20)
            current_linear = max_volume
            gain = target_linear / current_linear
            
            # Apply gain with safety limits
            safe_gain = min(max(gain, 0.1), 10.0)  # Limit between 0.1x and 10x
            return audio_clip.fx(volumex, safe_gain)
        
        except Exception as e:
            logger.error(f"Audio normalization failed: {str(e)}")
            return audio_clip
    
    def _get_video_preset(self, num_scenes: int) -> str:
        """Determine optimal encoding preset based on complexity"""
        total_pixels = self.generation_config.image_width * self.generation_config.image_height
        
        if num_scenes > 8 or total_pixels > 2000000:  # >2MP
            return 'fast'
        elif num_scenes > 5 or total_pixels > 1000000:  # >1MP
            return 'medium'
        else:
            return 'slow'
    
    def _safety_check(self, video_path: str) -> bool:
        """Verify the output video meets quality standards"""
        try:
            # Check file exists and has reasonable size
            if not os.path.exists(video_path):
                return False
                
            file_size = os.path.getsize(video_path)
            if file_size < 1024:  # Less than 1KB
                return False
                
            # Check video properties
            with VideoFileClip(video_path) as clip:
                if clip.duration < 0.1:  # Less than 100ms
                    return False
                    
                if clip.size[0] < 16 or clip.size[1] < 16:  # Minimum 16x16 pixels
                    return False
                    
                # Check for black frames (sign of generation failure)
                first_frame = clip.get_frame(0)
                if np.mean(first_frame) < 10:  # Average pixel value < 10
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Video safety check failed: {str(e)}")
            return False
    
    def process_single_scene(
        self,
        scene: Dict[str, str],
        scene_index: int,
        character_reference: Optional[Image.Image] = None,
        callback: Optional[Callable[[int, str], None]] = None
    ) -> Tuple[List[Image.Image], AudioFileClip]:
        """Process a single scene with enhanced error handling"""
        try:
            if callback:
                callback(scene_index * 10, f"Processing scene {scene_index + 1}")
            
            # Generate audio for narration
            audio_path = os.path.join(self.temp_dir, f"audio_{scene_index}.wav")
            self.generate_speech(scene["narration"], audio_path)
            
            # Load and normalize audio
            audio_clip = AudioFileClip(audio_path)
            audio_clip = self._normalize_audio(audio_clip)
            
            # Calculate target duration with constraints
            target_duration = min(
                max(
                    len(scene["narration"].split()) / 3,  # 3 words per second
                    self.generation_config.min_scene_duration
                ),
                self.generation_config.max_scene_duration
            )
            
            # Adjust audio to match target duration
            if audio_clip.duration > target_duration + 0.5:
                speed_factor = audio_clip.duration / target_duration
                audio_clip = audio_clip.fx(speedx, min(speed_factor, 2.0))  # Max 2x speed
            elif audio_clip.duration < target_duration - 0.5:
                speed_factor = audio_clip.duration / target_duration
                audio_clip = audio_clip.fx(speedx, max(speed_factor, 0.5))  # Min 0.5x speed
            
            # Generate visual content
            frames = []
            if (self.generation_config.enable_animation and 
                target_duration > self.generation_config.min_scene_duration * 1.5):
                
                # Create animated scene
                keyframes = self.generate_keyframes(
                    scene["visual"],
                    num_frames=5,
                    seed=42 + scene_index + (self.generation_config.seed or 0)
                )
                frames = self.interpolate_frames(
                    keyframes,
                    target_duration
                )
                
            elif (self.generation_config.enable_character_consistency and 
                  character_reference and "character" in scene["visual"].lower()):
                
                # Generate consistent character scene
                image = self.generate_character_scene(
                    scene["visual"],
                    character_reference,
                    seed=42 + scene_index + (self.generation_config.seed or 0)
                )
                frames = [image] * int(target_duration * 24)  # 24 FPS
                
            else:
                # Generate static image
                image = self.generate_image(
                    scene["visual"],
                    seed=42 + scene_index + (self.generation_config.seed or 0)
                )
                frames = [image] * int(target_duration * 24)  # 24 FPS
            
            # Add text overlay to first frame
            if frames:
                frames[0] = self.create_text_overlay(
                    frames[0],
                    scene["narration"][:100] + ("..." if len(scene["narration"]) > 100 else "")
                )
            
            return frames, audio_clip
            
        except Exception as e:
            logger.error(f"Scene {scene_index} processing failed: {str(e)}")
            # Return fallback content
            fallback_image = Image.new(
                'RGB',
                (self.generation_config.image_width, self.generation_config.image_height),
                color='black'
            )
            fallback_frames = [fallback_image] * int(self.generation_config.min_scene_duration * 24)
            
            # Create silent audio
            silent_audio = AudioClip(
                lambda t: 0,
                duration=self.generation_config.min_scene_duration,
                fps=self.generation_config.tts_sampling_rate
            )
            
            return fallback_frames, silent_audio
    
    def create_video(
        self,
        scenes: List[Dict[str, str]],
        output_filename: str = "output.mp4",
        add_transitions: bool = True,
        callback: Optional[Callable[[int, str], None]] = None
    ) -> str:
        """Create video with enhanced quality controls"""
        try:
            # Validate input duration
            estimated_duration = sum(
                self._calculate_scene_duration(s["narration"]) for s in scenes
            )
            if estimated_duration > self.generation_config.max_video_duration:
                raise ValueError(
                    f"Estimated duration {estimated_duration}s exceeds maximum allowed "
                    f"{self.generation_config.max_video_duration}s"
                )
            
            # Process scenes (parallel)
            results = self._process_scenes_parallel(scenes, callback)
            
            # Create video clips
            video_clips = self._create_video_clips(results, add_transitions, callback)
            
            # Render final video
            output_path = self._render_final_video(
                video_clips,
                [r[1] for r in results],
                output_filename,
                callback
            )
            
            # Verify output quality
            if not self._safety_check(output_path):
                raise RuntimeError("Generated video failed quality checks")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Video creation failed: {str(e)}")
            raise
    
    def _process_scenes_parallel(
        self,
        scenes: List[Dict[str, str]],
        callback: Optional[Callable[[int, str], None]]
    ) -> List[Tuple[List[Image.Image], AudioFileClip]]:
        """Process scenes in parallel with progress tracking"""
        results = []
        completed = 0
        total = len(scenes)
        
        with ThreadPoolExecutor(max_workers=self.generation_config.max_workers) as executor:
            futures = {
                executor.submit(
                    self.process_single_scene,
                    scene,
                    i,
                    self._character_reference if self.generation_config.enable_character_consistency else None,
                    None  # Don't pass callback to individual scenes
                ): i for i, scene in enumerate(scenes)
            }
            
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    results.append((idx, result))
                    completed += 1
                    
                    if callback:
                        progress = 10 + (completed / total) * 60  # 10-70% range
                        callback(int(progress), f"Processed {completed}/{total} scenes")
                        
                except Exception as e:
                    logger.error(f"Scene {idx} processing failed: {str(e)}")
                    # Add empty result as fallback
                    results.append((idx, ([], AudioClip(lambda t: 0, duration=1, fps=44100))))
        
        # Sort by original scene order
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]
    
    def _create_video_clips(
        self,
        scene_results: List[Tuple[List[Image.Image], AudioFileClip]],
        add_transitions: bool,
        callback: Optional[Callable[[int, str], None]]
    ) -> List[ImageClip]:
        """Create video clips from processed scenes"""
        video_clips = []
        
        for i, (frames, audio_clip) in enumerate(scene_results):
            if callback:
                callback(70 + (i / len(scene_results)) * 10, f"Creating scene {i + 1}")
            
            if not frames:
                continue
                
            # Create video clip from frames
            frame_duration = 1 / 24  # 24 FPS
            clips = [ImageClip(np.array(frame)).set_duration(frame_duration) for frame in frames]
            scene_clip = concatenate_videoclips(clips)
            
            # Add fade transitions if enabled
            if add_transitions and i > 0:
                scene_clip = scene_clip.crossfadein(self.generation_config.transition_duration)
            
            video_clips.append(scene_clip)
        
        return video_clips
    
    def _render_final_video(
        self,
        video_clips: List[ImageClip],
        audio_clips: List[AudioFileClip],
        output_filename: str,
        callback: Optional[Callable[[int, str], None]]
    ) -> str:
        """Render the final video file with quality controls"""
        if callback:
            callback(85, "Combining video clips")
        
        # Concatenate video clips with transitions
        final_video = concatenate_videoclips(
            video_clips,
            method="compose",
            padding=-self.generation_config.transition_duration if len(video_clips) > 1 else 0
        )
        
        if callback:
            callback(88, "Combining audio tracks")
        
        # Combine audio tracks with normalization
        normalized_audio = [self._normalize_audio(a) for a in audio_clips]
        final_audio = CompositeAudioClip(normalized_audio)
        
        # Add background music if enabled
        if self.generation_config.enable_background_music:
            if callback:
                callback(90, "Adding background music")
            
            music = self.generate_background_music("neutral", final_video.duration)
            music_audio = AudioClip(
                lambda t: music[int(t * 22050)] if int(t * 22050) < len(music) else 0,
                duration=final_video.duration,
                fps=22050
            ).fx(volumex, self.generation_config.music_volume)
            
            final_audio = CompositeAudioClip([final_audio, music_audio])
        
        if callback:
            callback(92, "Final composition")
        
        # Set audio of the final video
        final_video = final_video.set_audio(final_audio)
        
        # Determine encoding preset
        preset = self._get_video_preset(len(video_clips))
        
        if callback:
            callback(95, "Encoding video")
        
        # Write output file
        output_path = os.path.join(self.temp_dir, output_filename)
        final_video.write_videofile(
            output_path,
            fps=24,
            codec='libx264',
            audio_codec='aac',
            threads=4,
            logger=None,
            bitrate="8000k",
            preset=preset,
            ffmpeg_params=[
                '-movflags', '+faststart',  # Enable streaming
                '-pix_fmt', 'yuv420p',      # Wider compatibility
                '-crf', '18'                # High quality
            ]
        )
        
        # Clean up resources
        for clip in video_clips + audio_clips:
            clip.close()
        final_video.close()
        
        if callback:
            callback(100, "Video rendering complete")
        
        return output_path
    
    def generate_from_prompt(
        self,
        prompt: str,
        output_filename: str = "output.mp4",
        callback: Optional[Callable[[int, str], None]] = None,
        **kwargs
    ) -> str:
        """Complete pipeline from text prompt to video with configurable parameters"""
        logger.info(f"Generating script for prompt: {prompt}")
        
        if callback:
            callback(0, "Generating script")
        
        # Update generation config if kwargs provided
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.generation_config, key):
                    setattr(self.generation_config, key, value)
        
        script = self.generate_script(prompt)
        logger.info(f"Generated script with {len(script)} scenes")
        
        if callback:
            callback(10, "Script generated")
        
        logger.info("Creating video...")
        video_path = self.create_video(script, output_filename, callback=callback)
        logger.info(f"Video created at: {video_path}")
        
        return video_path
    
    def learn_from_url(self, url: str) -> str:
        """Extract text content from a URL for context with improved parsing and validation"""
        # Validate URL
        if not re.match(r'^https?://', url):
            logger.warning(f"Invalid URL format: {url}")
            return ""
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            
            # Check if response is successful
            if response.status_code != 200:
                logger.warning(f"Failed to fetch URL: {response.status_code}")
                return ""
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'iframe', 'noscript']):
                element.decompose()
            
            # Get text from main content areas
            paragraphs = []
            for tag in ['article', 'main', 'div', 'p']:
                elements = soup.find_all(tag)
                for el in elements:
                    text = el.get_text(separator=' ', strip=True)
                    if len(text.split()) > 10:  # Only keep substantial paragraphs
                        paragraphs.append(text)
            
            content = ' '.join(paragraphs)
            
            # Clean up excessive whitespace
            content = re.sub(r'\s+', ' ', content).strip()
            return content[:5000]  # Limit to 5000 characters
        except Exception as e:
            logger.error(f"Error learning from {url}: {str(e)}")
            return ""
    
    def cleanup(self):
        """Clean up temporary files with error handling"""
        try:
            for file in os.listdir(self.temp_dir):
                try:
                    os.remove(os.path.join(self.temp_dir, file))
                except Exception as e:
                    logger.warning(f"Could not delete temporary file {file}: {str(e)}")
            os.rmdir(self.temp_dir)
            logger.info("Cleaned up temporary files")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    def save_config(self, filepath: str):
        """Save current configuration to file"""
        config = {
            "model_config": {
                "text_model": self.model_config.text_model.value,
                "tts_model": self.model_config.tts_model.value,
                "image_model": self.model_config.image_model.value
            },
            "generation_config": {
                "video_length": self.generation_config.video_length,
                "min_scene_duration": self.generation_config.min_scene_duration,
                "max_scene_duration": self.generation_config.max_scene_duration,
                "transition_duration": self.generation_config.transition_duration,
                "image_width": self.generation_config.image_width,
                "image_height": self.generation_config.image_height,
                "image_steps": self.generation_config.image_steps,
                "image_cfg_scale": self.generation_config.image_cfg_scale,
                "tts_sampling_rate": self.generation_config.tts_sampling_rate,
                "text_model_temp": self.generation_config.text_model_temp,
                "seed": self.generation_config.seed,
                "enable_safety_checker": self.generation_config.enable_safety_checker,
                "max_workers": self.generation_config.max_workers,
                "enable_animation": self.generation_config.enable_animation,
                "enable_character_consistency": self.generation_config.enable_character_consistency,
                "enable_background_music": self.generation_config.enable_background_music,
                "music_volume": self.generation_config.music_volume,
                "max_video_duration": self.generation_config.max_video_duration,
                "target_audio_db": self.generation_config.target_audio_db
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration saved to {filepath}")
    
    def load_config(self, filepath: str):
        """Load configuration from file"""
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        # Update model config
        self.model_config.text_model = ModelType(config["model_config"]["text_model"])
        self.model_config.tts_model = ModelType(config["model_config"]["tts_model"])
        self.model_config.image_model = ModelType(config["model_config"]["image_model"])
        
        # Update generation config
        for key, value in config["generation_config"].items():
            setattr(self.generation_config, key, value)
        
        # Re-validate config
        self._validate_config()
        
        logger.info(f"Configuration loaded from {filepath}")

# Example usage with all enhancements
if __name__ == "__main__":
    def progress_callback(percent: int, message: str):
        print(f"[{percent}%] {message}")
    
    # Configuration with all enhanced settings
    config = GenerationConfig(
        video_length=60.0,
        min_scene_duration=4.0,
        max_scene_duration=15.0,
        transition_duration=1.0,
        image_width=1024,
        image_height=576,
        image_steps=30,
        enable_animation=True,
        enable_character_consistency=True,
        enable_background_music=True,
        music_volume=0.15,
        target_audio_db=-16.0,
        max_video_duration=300.0  # 5 minute limit
    )
    
    generator = FreeTextToVideoGenerator(generation_config=config)
    
    try:
        # Save configuration
        generator.save_config("video_config.json")
        
        # Generate a complex video with all features
        result = generator.generate_from_prompt(
            "A documentary about the future of AI in healthcare, showing doctors "
            "working with AI assistants to diagnose patients",
            "ai_healthcare.mp4",
            callback=progress_callback,
            enable_animation=True,
            enable_character_consistency=True
        )
        
        print(f"\nSuccessfully created video: {result}")
        print(f"Video duration: {VideoFileClip(result).duration:.1f} seconds")
        
        # Learn from a URL and generate context-aware video
        context = generator.learn_from_url("https://en.wikipedia.org/wiki/Artificial_intelligence")
        if context:
            print(f"\nLearned context ({len(context)} characters)")
            
            # Generate video with context
            context_video_path = generator.generate_from_prompt(
                f"Explain {context[:200]} in simple terms",
                "ai_explanation.mp4",
                callback=progress_callback
            )
            
            print(f"\nContext-aware video saved at: {context_video_path}")
            print(f"File size: {os.path.getsize(context_video_path) / (1024*1024):.2f} MB")
        
    except Exception as e:
        print(f"\nError during video generation: {str(e)}")
    
    finally:
        generator.cleanup()