# Pull python image from Dockerhub
FROM python:3.11.1

# Prevents Python from generating .pyc files in the container.
ENV PYTHONDONTWRITEBYTECODE=1
# Turns off buffering for easier container logging.
ENV PYTHONUNBUFFERED=1
# Force UTF8 encoding for funky characters.
ENV PYTHONIOENCODING=utf8

# Install curl
RUN apt-get update && apt-get install -y curl

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry path to PATH
ENV PATH="${PATH}:/root/.local/bin"

# Set the working directory
WORKDIR /app

# Copy the pyproject.toml and poetry.lock files
COPY pyproject.toml poetry.lock /app/

# Install project dependencies with Poetry
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root \
    && rm pyproject.toml poetry.lock

# Copy the rest of the application code
COPY . /app

# Set the command to run the application
CMD ["python", "app.py"]