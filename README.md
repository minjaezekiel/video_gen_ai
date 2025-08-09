

# FreeTextToVideoGenerator

A comprehensive Python implementation for generating videos with sound and voice from text prompts using free AI models. This system leverages state-of-the-art models for text generation, speech synthesis, and image generation to create complete videos with animations, character consistency, and background music.

## Features

### Core Capabilities
- **Text-to-Video Generation**: Convert text prompts into complete videos
- **Natural Voice Synthesis**: High-quality text-to-speech with multiple model options
- **AI-Generated Imagery**: Create visuals using Stable Diffusion models
- **Scene Animation**: Generate smooth transitions between scenes
- **Character Consistency**: Maintain character appearance across scenes
- **Background Music**: Automatically generate mood-appropriate background music
- **Audio Normalization**: Ensure consistent audio levels throughout videos
- **Parallel Processing**: Utilize multiple workers for faster generation
- **Quality Controls**: Comprehensive checks for output video quality

### Advanced Features
- **Lazy Model Loading**: Models load only when needed to save memory
- **Caching System**: Intelligent caching of generated images
- **Configuration Management**: Save and load generation settings
- **Progress Callbacks**: Real-time progress reporting during generation
- **Error Handling**: Robust fallbacks for failed operations
- **Resource Management**: Automatic cleanup and memory optimization
- **Web Integration**: Easy integration with web frameworks

## Installation

### Prerequisites
- Python 3.8+
- FFmpeg (for video processing)
- CUDA-compatible GPU (recommended for faster generation)

### Install Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers diffusers soundfile moviepy pillow requests beautifulsoup4 scipy numpy
```

### Install FFmpeg
**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:** Download from [https://ffmpeg.org/](https://ffmpeg.org/)

## Quick Start

### Basic Usage 
-Use the test.py file instead, you can change the prompts from there...
```python
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
```

## Framework Integration

### Django Integration

#### 1. Create Django App
```bash
django-admin startproject video_project
cd video_project
python manage.py startapp video_generator
```

#### 2. Update Settings (`video_project/settings.py`)
```python
INSTALLED_APPS = [
    ...
    'video_generator',
]
```

#### 3. Create View (`video_generator/views.py`)
```python
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import os
from .models import VideoGenerationTask
from .tasks import generate_video_task
import json

@csrf_exempt
def generate_video(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            prompt = data.get('prompt')
            
            # Create task record
            task = VideoGenerationTask.objects.create(
                prompt=prompt,
                status='pending'
            )
            
            # Start background task
            generate_video_task.delay(task.id, prompt)
            
            return JsonResponse({
                'task_id': task.id,
                'status': 'pending'
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

def video_status(request, task_id):
    try:
        task = VideoGenerationTask.objects.get(id=task_id)
        return JsonResponse({
            'status': task.status,
            'video_path': task.video_path if task.video_path else None
        })
    except VideoGenerationTask.DoesNotExist:
        return JsonResponse({'error': 'Task not found'}, status=404)
```

#### 4. Create Model (`video_generator/models.py`)
```python
from django.db import models

class VideoGenerationTask(models.Model):
    prompt = models.TextField()
    status = models.CharField(max_length=20, default='pending')
    video_path = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
```

#### 5. Create Celery Task (`video_generator/tasks.py`)
```python
from celery import shared_task
from django.conf import settings
import os
from .models import VideoGenerationTask
from free_text_to_video import FreeTextToVideoGenerator, GenerationConfig

@shared_task
def generate_video_task(task_id, prompt):
    try:
        task = VideoGenerationTask.objects.get(id=task_id)
        task.status = 'processing'
        task.save()
        
        # Initialize generator
        config = GenerationConfig(
            video_length=30.0,
            enable_animation=True,
            enable_character_consistency=True
        )
        
        generator = FreeTextToVideoGenerator(generation_config=config)
        
        # Generate video
        video_filename = f"video_{task_id}.mp4"
        video_path = generator.generate_from_prompt(prompt, video_filename)
        
        # Move to media directory
        media_path = os.path.join(settings.MEDIA_ROOT, video_filename)
        os.rename(video_path, media_path)
        
        # Update task
        task.status = 'completed'
        task.video_path = media_path
        task.save()
        
        # Cleanup
        generator.cleanup()
        
    except Exception as e:
        task.status = 'failed'
        task.save()
        raise e
```

#### 6. Configure URLs (`video_project/urls.py`)
```python
from django.urls import path
from video_generator import views

urlpatterns = [
    path('generate/', views.generate_video, name='generate_video'),
    path('status/<int:task_id>/', views.video_status, name='video_status'),
]
```

### FastAPI Integration

#### 1. Create FastAPI App (`main.py`)
```python
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import uuid
import os
from free_text_to_video import FreeTextToVideoGenerator, GenerationConfig

app = FastAPI()

# In-memory task storage (use a database in production)
tasks = {}

class VideoRequest(BaseModel):
    prompt: str
    video_length: float = 30.0
    enable_animation: bool = True
    enable_character_consistency: bool = True

class TaskStatus(BaseModel):
    task_id: str
    status: str
    video_path: str = None

def generate_video_task(task_id: str, request: VideoRequest):
    try:
        tasks[task_id]['status'] = 'processing'
        
        # Initialize generator
        config = GenerationConfig(
            video_length=request.video_length,
            enable_animation=request.enable_animation,
            enable_character_consistency=request.enable_character_consistency
        )
        
        generator = FreeTextToVideoGenerator(generation_config=config)
        
        # Generate video
        video_path = generator.generate_from_prompt(
            request.prompt,
            f"{task_id}.mp4"
        )
        
        # Update task
        tasks[task_id]['status'] = 'completed'
        tasks[task_id]['video_path'] = video_path
        
        # Cleanup
        generator.cleanup()
        
    except Exception as e:
        tasks[task_id]['status'] = f'failed: {str(e)}'

@app.post("/generate/", response_model=TaskStatus)
async def generate_video(request: VideoRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        'status': 'pending',
        'video_path': None
    }
    
    background_tasks.add_task(generate_video_task, task_id, request)
    
    return TaskStatus(task_id=task_id, status='pending')

@app.get("/status/{task_id}", response_model=TaskStatus)
async def get_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return TaskStatus(
        task_id=task_id,
        status=tasks[task_id]['status'],
        video_path=tasks[task_id].get('video_path')
    )
```

### Flask Integration

#### 1. Create Flask App (`app.py`)
```python
from flask import Flask, request, jsonify, send_file
import uuid
import os
import threading
from free_text_to_video import FreeTextToVideoGenerator, GenerationConfig

app = Flask(__name__)

# In-memory task storage (use a database in production)
tasks = {}

def generate_video_task(task_id, prompt, config):
    try:
        tasks[task_id]['status'] = 'processing'
        
        # Initialize generator
        generator = FreeTextToVideoGenerator(generation_config=config)
        
        # Generate video
        video_path = generator.generate_from_prompt(
            prompt,
            f"{task_id}.mp4"
        )
        
        # Update task
        tasks[task_id]['status'] = 'completed'
        tasks[task_id]['video_path'] = video_path
        
        # Cleanup
        generator.cleanup()
        
    except Exception as e:
        tasks[task_id]['status'] = f'failed: {str(e)}'

@app.route('/generate', methods=['POST'])
def generate_video():
    data = request.get_json()
    prompt = data.get('prompt')
    
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400
    
    # Create config from request
    config = GenerationConfig(
        video_length=data.get('video_length', 30.0),
        enable_animation=data.get('enable_animation', True),
        enable_character_consistency=data.get('enable_character_consistency', True)
    )
    
    # Create task
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        'status': 'pending',
        'video_path': None
    }
    
    # Start background thread
    thread = threading.Thread(
        target=generate_video_task,
        args=(task_id, prompt, config)
    )
    thread.start()
    
    return jsonify({
        'task_id': task_id,
        'status': 'pending'
    })

@app.route('/status/<task_id>')
def get_status(task_id):
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    return jsonify({
        'task_id': task_id,
        'status': tasks[task_id]['status'],
        'video_path': tasks[task_id].get('video_path')
    })

@app.route('/download/<task_id>')
def download_video(task_id):
    if task_id not in tasks or tasks[task_id]['status'] != 'completed':
        return jsonify({'error': 'Video not ready'}), 404
    
    video_path = tasks[task_id]['video_path']
    return send_file(video_path, as_attachment=True)
```

## Tutorial: Building a Video Generation Web Service

### Step 1: Set Up the Project
```bash
mkdir video_service
cd video_service
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install fastapi uvicorn
```

### Step 2: Create the FastAPI Application
Save the FastAPI code above as `main.py`

### Step 3: Add Frontend (Optional)
Create `templates/index.html`:
```html
<!DOCTYPE html>
<html>
<head>
    <title>AI Video Generator</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; }
        input, textarea, select { width: 100%; padding: 8px; }
        button { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; cursor: pointer; }
        .progress { margin-top: 20px; }
        .hidden { display: none; }
    </style>
</head>
<body>
    <h1>AI Video Generator</h1>
    
    <form id="videoForm">
        <div class="form-group">
            <label for="prompt">Video Prompt:</label>
            <textarea id="prompt" rows="4" required></textarea>
        </div>
        
        <div class="form-group">
            <label for="videoLength">Video Length (seconds):</label>
            <input type="number" id="videoLength" value="30" min="10" max="120">
        </div>
        
        <div class="form-group">
            <label>
                <input type="checkbox" id="enableAnimation" checked> Enable Animation
            </label>
        </div>
        
        <div class="form-group">
            <label>
                <input type="checkbox" id="enableCharacterConsistency" checked> Character Consistency
            </label>
        </div>
        
        <button type="submit">Generate Video</button>
    </form>
    
    <div id="progress" class="progress hidden">
        <h3>Generation Progress</h3>
        <div id="status"></div>
        <div id="downloadLink" class="hidden">
            <a href="" download>Download Video</a>
        </div>
    </div>
    
    <script>
        document.getElementById('videoForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const prompt = document.getElementById('prompt').value;
            const videoLength = document.getElementById('videoLength').value;
            const enableAnimation = document.getElementById('enableAnimation').checked;
            const enableCharacterConsistency = document.getElementById('enableCharacterConsistency').checked;
            
            // Show progress
            document.getElementById('progress').classList.remove('hidden');
            document.getElementById('status').textContent = 'Starting generation...';
            document.getElementById('downloadLink').classList.add('hidden');
            
            try {
                // Start generation
                const response = await fetch('/generate/', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        prompt,
                        videoLength: parseFloat(videoLength),
                        enableAnimation,
                        enableCharacterConsistency
                    })
                });
                
                const data = await response.json();
                const taskId = data.task_id;
                
                // Poll for status
                const checkStatus = async () => {
                    const statusResponse = await fetch(`/status/${taskId}`);
                    const statusData = await statusResponse.json();
                    
                    document.getElementById('status').textContent = `Status: ${statusData.status}`;
                    
                    if (statusData.status === 'completed') {
                        document.getElementById('downloadLink').querySelector('a').href = `/download/${taskId}`;
                        document.getElementById('downloadLink').classList.remove('hidden');
                    } else if (statusData.status.startsWith('failed')) {
                        document.getElementById('status').textContent = `Error: ${statusData.status}`;
                    } else {
                        setTimeout(checkStatus, 2000);
                    }
                };
                
                checkStatus();
                
            } catch (error) {
                document.getElementById('status').textContent = `Error: ${error.message}`;
            }
        });
    </script>
</body>
</html>
```

### Step 4: Run the Service
```bash
uvicorn main:app --reload
```

Visit `http://localhost:8000` to use the video generator.

## Other Usage Examples

### Command-Line Interface
Create `generate_video.py`:
```python
import argparse
from free_text_to_video import FreeTextToVideoGenerator, GenerationConfig

def main():
    parser = argparse.ArgumentParser(description='Generate videos from text prompts')
    parser.add_argument('prompt', help='Text prompt for video generation')
    parser.add_argument('--output', default='output.mp4', help='Output video file')
    parser.add_argument('--length', type=float, default=30.0, help='Video length in seconds')
    parser.add_argument('--no-animation', action='store_true', help='Disable animation')
    parser.add_argument('--no-character-consistency', action='store_true', help='Disable character consistency')
    parser.add_argument('--no-music', action='store_true', help='Disable background music')
    
    args = parser.parse_args()
    
    config = GenerationConfig(
        video_length=args.length,
        enable_animation=not args.no_animation,
        enable_character_consistency=not args.no_character_consistency,
        enable_background_music=not args.no_music
    )
    
    generator = FreeTextToVideoGenerator(generation_config=config)
    
    def progress_callback(percent, message):
        print(f"[{percent}%] {message}")
    
    video_path = generator.generate_from_prompt(
        args.prompt,
        args.output,
        callback=progress_callback
    )
    
    print(f"\nVideo saved to: {video_path}")
    generator.cleanup()

if __name__ == '__main__':
    main()
```

Run with:
```bash
python generate_video.py "A magical forest with glowing mushrooms" --output forest.mp4 --length 45
```

### Jupyter Notebook Integration
```python
from video_generator import FreeTextToVideoGenerator, GenerationConfig
from IPython.display import Video

# Initialize generator
config = GenerationConfig(
    video_length=20.0,
    enable_animation=True,
    enable_character_consistency=True
)

generator = FreeTextToVideoGenerator(generation_config=config)

# Generate video
video_path = generator.generate_from_prompt(
    "A robot exploring an alien planet",
    "alien_planet.mp4"
)

# Display video in notebook
Video(video_path)

# Cleanup
generator.cleanup()
```

## Configuration Options

### GenerationConfig Parameters
- `video_length`: Target video length in seconds (default: 30.0)
- `min_scene_duration`: Minimum duration per scene (default: 3.0)
- `max_scene_duration`: Maximum duration per scene (default: 10.0)
- `transition_duration`: Duration of transitions between scenes (default: 0.5)
- `image_width`: Width of generated images (default: 768)
- `image_height`: Height of generated images (default: 768)
- `image_steps`: Number of inference steps for image generation (default: 25)
- `image_cfg_scale`: Guidance scale for image generation (default: 7.5)
- `tts_sampling_rate`: Sampling rate for text-to-speech (default: 16000)
- `text_model_temp`: Temperature for text generation (default: 0.7)
- `seed`: Random seed for reproducible generation (default: None)
- `enable_safety_checker`: Enable image safety checker (default: False)
- `max_workers`: Number of parallel workers (default: 4)
- `enable_animation`: Enable scene animation (default: True)
- `enable_character_consistency`: Enable character consistency (default: True)
- `enable_background_music`: Enable background music (default: True)
- `music_volume`: Background music volume (0.0-1.0) (default: 0.2)
- `max_video_duration`: Maximum allowed video duration (default: 600.0)
- `target_audio_db`: Target audio loudness in dB (default: -20.0)

### Model Selection
```python
from free_text_to_video import ModelConfig

config = ModelConfig(
    text_model=ModelType.LARGE,    # Options: SMALL, MEDIUM, LARGE, XL
    tts_model=ModelType.LARGE,     # Options: SMALL, MEDIUM, LARGE
    image_model=ModelType.XL      # Options: SMALL, MEDIUM, LARGE, XL
)
```

## Performance Optimization

### GPU Usage
For optimal performance, use a CUDA-compatible GPU:
```python
generator = FreeTextToVideoGenerator(device="cuda")
```

### Memory Management
For systems with limited memory:
```python
config = GenerationConfig(
    max_workers=2,  # Reduce parallel workers
    image_steps=20,  # Reduce inference steps
    enable_animation=False  # Disable animation for complex scenes
)
```

### Caching
Images are automatically cached to speed up repeated generations with similar prompts.

## Contributing

We welcome contributions to improve FreeTextToVideoGenerator! Here's how you can help:

### Development Setup
1. Fork the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```
4. Create a branch for your feature:
   ```bash
   git checkout -b feature-name
   ```

### Contribution Guidelines
- Follow PEP 8 style guidelines
- Write unit tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting a pull request

### Areas for Contribution
1. **New Model Integration**: Add support for additional AI models
2. **Performance Improvements**: Optimize memory usage and generation speed
3. **Animation Enhancements**: Improve frame interpolation and motion generation
4. **Audio Features**: Add more sophisticated music generation and sound effects
5. **UI/UX Improvements**: Enhance web interfaces and user experience
6. **Documentation**: Improve tutorials and API documentation
7. **Fix bugs/errors**: Fix any bugs or errors attributed to the project
8. **Testing**: Expand test coverage for better reliability

### Submitting Changes
1. Commit your changes:
   ```bash
   git commit -m "Add feature: description of changes"
   ```
2. Push to your fork:
   ```bash
   git push origin feature-name
   ```
3. Create a pull request to the main repository

### Reporting Issues
If you encounter bugs or have suggestions:
1. Check existing issues to avoid duplicates
2. Create a new issue with:
   - Clear description of the problem
   - Steps to reproduce
   - Expected behavior
   - System information (OS, Python version, etc.)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/) for NLP models
- [Diffusers](https://huggingface.co/docs/diffusers) for image generation
- [MoviePy](https://zulko.github.io/moviepy/) for video processing
- [FFmpeg](https://ffmpeg.org/) for video encoding

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review example implementations

---

**FreeTextToVideoGenerator** - Transform your ideas into videos with the power of AI!