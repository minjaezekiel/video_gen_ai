# A container that can actually make a film with sound.
#
#   docker build -t video-gen .
#   docker run --rm -v "$PWD/output:/work/output" video-gen --self-test
#   docker run --rm -v "$PWD/output:/work/output" video-gen \
#       --prompt "a 30s advert for a bike shop" --json --output output/ad.mp4
#
# espeak-ng is the point of this file. Without a synthesiser every line falls
# back to timed silence, the render still "succeeds", and what comes out is
# subtitles over silence -- which is why that case exits 2 rather than 0.
FROM python:3.12-slim

# espeak-ng   the speech synthesiser, and the voices casting draws from
# libsndfile1 soundfile's native library
# fonts       captions have nothing to draw with on a slim image
RUN apt-get update && apt-get install -y --no-install-recommends \
        espeak-ng \
        libsndfile1 \
        fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# Dependencies first, so editing the source does not reinstall torch-sized
# wheels on every build.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md LICENSE ./
COPY video_gen ./video_gen
RUN pip install --no-cache-dir --no-deps -e .

# Fail the build rather than ship an image whose voices do not work.
RUN python -m video_gen --doctor --json > /tmp/doctor.json \
    && python -c "import json,sys; d=json.load(open('/tmp/doctor.json')); \
        sys.exit(0) if d['capabilities']['speech'] and d['voices']['castable'] > 1 \
        else sys.exit('no working speech in image: ' + str(d['speech']))"

ENTRYPOINT ["python", "-m", "video_gen"]
CMD ["--doctor"]
