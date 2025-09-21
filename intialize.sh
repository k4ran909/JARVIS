#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install Ollama
install_ollama() {
    if command_exists ollama; then
        echo "✅ Ollama is already installed."
    else
        echo "Installing Ollama..."
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            curl -fsSL https://ollama.com/install.sh | sh
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            brew install ollama
        else
            echo "❌ Unsupported OS for Ollama installation."
            exit 1
        fi
    fi
}

# Function to install ADB
install_adb() {
    if command_exists adb; then
        echo "✅ ADB is already installed."
    else
        echo "Installing ADB..."
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            if command_exists apt; then
                sudo apt update && sudo apt install -y adb
            elif command_exists pacman; then
                sudo pacman -Sy --noconfirm android-tools
            else
                echo "❌ Unsupported package manager for ADB installation."
                exit 1
            fi
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            brew install android-platform-tools
        else
            echo "❌ Unsupported OS for ADB installation."
            exit 1
        fi
    fi
}

# Function to install PortAudio
install_portaudio() {
    echo "🔍 Checking for PortAudio..."

    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if dpkg -l | grep -q "portaudio"; then
            echo "✅ PortAudio is already installed."
            return
        fi

        echo "Installing PortAudio..."
        if command_exists apt; then
            sudo apt update && sudo apt install -y portaudio19-dev libportaudio2
        elif command_exists pacman; then
            sudo pacman -Sy --noconfirm portaudio
        else
            echo "❌ Unsupported package manager for PortAudio installation."
            exit 1
        fi

    elif [[ "$OSTYPE" == "darwin"* ]]; then
        if brew list --formula | grep -q "^portaudio\$"; then
            echo "✅ PortAudio is already installed."
            return
        fi

        echo "Installing PortAudio..."
        brew install portaudio

    else
        echo "❌ Unsupported OS for PortAudio installation."
        exit 1
    fi
}


# Function to check and install Ollama models
install_models() {
    models=("gemma3:4b" "granite3.1-dense:2b" "nomic-embed-text")

    for model in "${models[@]}"; do
        if ollama list | grep -q "$model"; then
            echo "✅ Model '$model' is already installed."
        else
            echo "Installing model '$model'..."
            ollama pull "$model"
        fi
    done
}

# Detect OS
echo "🔍 Detecting OS..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "🖥️ OS: Linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🖥️ OS: macOS"
else
    echo "❌ Unsupported OS detected."
    exit 1
fi

# Install dependencies if not already installed
install_ollama
install_adb
install_portaudio
install_models

echo "✅ All installations complete!"
