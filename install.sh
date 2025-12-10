#!/usr/bin/env bash
set -e

# ===== CONFIG =====
INSTALL_DIR="$HOME/.local"
PKG_DIR="$INSTALL_DIR/ccutils"
REPO_URL="https://github.com/ThomasPasquali/ccutils.git"

echo "[1] Creating directories..."
mkdir -p "$PKG_DIR"

echo "[2] Cloning repository..."
if [ ! -d "$PKG_DIR/.git" ]; then
    git clone "$REPO_URL" "$PKG_DIR"
else
    echo "Repository already cloned, pulling latest changes..."
    git -C "$PKG_DIR" pull
fi

cd "$PKG_DIR"

echo "[3] Configuring CMake..."
cmake -B build -S . \
    -DCCUTILS_ENABLE_CUDA=ON \
    -DCCUTILS_ENABLE_MPI=ON \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"

echo "[4] Building..."
cmake --build build --parallel

echo "[5] Installing..."
cmake --install build --prefix install

echo "[6] Updating ~/.bashrc if needed..."
BASHRC="$HOME/.bashrc"
ADDED=0

if ! grep -q "CMAKE_PREFIX_PATH=\"$INSTALL_DIR" "$BASHRC"; then
    echo "export CMAKE_PREFIX_PATH=\"$INSTALL_DIR:\$CMAKE_PREFIX_PATH\"" >> "$BASHRC"
    ADDED=1
fi

if [ $ADDED -eq 1 ]; then
    echo "Updated ~/.bashrc. Remember to run:"
    echo "    source ~/.bashrc"
else
    echo "~/.bashrc already configured."
fi

echo "Installation complete."