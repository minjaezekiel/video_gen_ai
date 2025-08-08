from video_generator import FreeTextToVideoGenerator, GenerationConfig

# Initialize generator with custom configuration
config = GenerationConfig(
    video_length=30.0,
    enable_animation=True,
    enable_character_consistency=True,
    enable_background_music=True
)

generator = FreeTextToVideoGenerator(generation_config=config)

# Generate video from prompt
video_path = generator.generate_from_prompt(
    "A futuristic city where robots and humans coexist peacefully",
    "future_city.mp4"
)

print(f"Video saved to: {video_path}")
generator.cleanup()