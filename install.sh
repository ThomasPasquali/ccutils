#!/usr/bin/env bash
set -euo pipefail

#####################################
# CONFIGURATION
#####################################

INSTALL_PREFIX="$HOME/.local"
PKG_DIR="$INSTALL_PREFIX/share/ccutils"

REPO_URL="https://github.com/ThomasPasquali/ccutils.git"

#####################################
# FUNCTIONS
#####################################

ask_yes_no() {
    local prompt="$1"
    local default="${2:-y}"

    local choice
    read -rp "$prompt [y/n] (default: $default): " choice
    choice="${choice:-$default}"

    case "$choice" in
        [Yy]*) return 0 ;;
        [Nn]*) return 1 ;;
        *) echo "Invalid input. Assuming $default." ; [[ "$default" == "y" ]] ;;
    esac
}

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
# USER OPTIONS: CUDA / MPI
#####################################

echo "--------- ccutils Installation Options ---------"
ENABLE_CUDA=OFF
ENABLE_MPI=OFF

if ask_yes_no "Enable CUDA support?" n; then
    ENABLE_CUDA=ON
fi

if ask_yes_no "Enable MPI support?" n; then
    ENABLE_MPI=ON
fi

echo
echo "Selected options:"
echo "  CUDA: $ENABLE_CUDA"
echo "  MPI : $ENABLE_MPI"
echo "------------------------------------------------"
echo

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
    -DCCUTILS_ENABLE_CUDA="$ENABLE_CUDA" \
    -DCCUTILS_ENABLE_MPI="$ENABLE_MPI" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX"

echo "[4] Building..."
cmake --build build --parallel

echo "[5] Installing to $INSTALL_PREFIX ..."
cmake --install build --prefix "$INSTALL_PREFIX/ccutils/install"

#####################################
# ENVIRONMENT UPDATES
#####################################

echo "[6] Updating shell configuration..."

RC_FILE=$(detect_shell_rc)

echo "Using RC file: $RC_FILE"

# Update CMAKE_PREFIX_PATH
# TODO double check if the path works
ensure_line_present "$RC_FILE" \
"export CMAKE_PREFIX_PATH=\"$INSTALL_PREFIX/ccutils/lib/cmake:\$CMAKE_PREFIX_PATH\""
ensure_line_present "$RC_FILE" \
"export CCUTILS_INCLUDE=\"$INSTALL_PREFIX/ccutils/install/include\""

echo
echo "Installation complete."
echo "To activate changes, run:"
echo "    source \"$RC_FILE\""
echo
