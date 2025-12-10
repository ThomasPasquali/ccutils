#!/usr/bin/env bash
set -euo pipefail

#####################################
# CONFIGURATION
#####################################

INSTALL_PREFIX="$HOME/.local"
PKG_DIR="$INSTALL_PREFIX/ccutils"
REPO_URL="https://github.com/ThomasPasquali/ccutils.git"

#####################################
# FUNCTIONS
#####################################

detect_shell_rc() {
    local shell_name rc

    shell_name=$(basename "${SHELL:-}")

    case "$shell_name" in
        bash)
            rc="$HOME/.bashrc"
            ;;
        zsh)
            rc="$HOME/.zshrc"
            ;;
        fish)
            rc="$HOME/.config/fish/config.fish"
            mkdir -p "$(dirname "$rc")"
            ;;
        *)
            # fallback for sh, dash, etc.
            rc="$HOME/.profile"
            ;;
    esac

    echo "$rc"
}

ensure_line_present() {
    local file="$1"
    local line="$2"

    if [ ! -f "$file" ]; then
        touch "$file"
    fi

    if ! grep -Fxq "$line" "$file"; then
        echo "$line" >> "$file"
        echo "Updated $file"
    else
        echo "$file already configured."
    fi
}

#####################################
# INSTALLATION
#####################################

echo "[1] Creating installation directory at $PKG_DIR ..."
mkdir -p "$PKG_DIR"

echo "[2] Cloning or updating repository..."
if [ ! -d "$PKG_DIR/.git" ]; then
    git clone "$REPO_URL" "$PKG_DIR"
else
    echo "Repository already exists. Pulling latest changes..."
    git -C "$PKG_DIR" pull --rebase
fi

cd "$PKG_DIR"

echo "[3] Configuring CMake..."
cmake -B build -S . \
    -DCCUTILS_ENABLE_CUDA=ON \
    -DCCUTILS_ENABLE_MPI=ON \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX"

echo "[4] Building..."
cmake --build build --parallel

echo "[5] Installing to $INSTALL_PREFIX ..."
cmake --install build --prefix "$INSTALL_PREFIX/install"

#####################################
# ENVIRONMENT UPDATES
#####################################

echo "[6] Updating shell configuration..."

RC_FILE=$(detect_shell_rc)

echo "Using RC file: $RC_FILE"

# Update CMAKE_PREFIX_PATH
ensure_line_present "$RC_FILE" \
"export CMAKE_PREFIX_PATH=\"$INSTALL_PREFIX:\$CMAKE_PREFIX_PATH\""
ensure_line_present "$RC_FILE" \
"export CCUTILS_INCLUDE=\"$INSTALL_PREFIX/install/include\""

echo
echo "Installation complete."
echo "To activate changes, run:"
echo "    source \"$RC_FILE\""
echo
