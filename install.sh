#!/usr/bin/env bash
set -euo pipefail

#####################################
# CONFIGURATION
#####################################

INSTALL_PREFIX="$HOME/.local"
PKG_DIR="$INSTALL_PREFIX/share/ccutils"
INSTALL_DIR="$INSTALL_PREFIX/ccutils/install"

REPO_URL="https://github.com/ThomasPasquali/ccutils.git"

#####################################
# FUNCTIONS
#####################################

ask_yes_no() {
    local prompt="$1"
    local default="${2:-y}"
    local answer=""

    if [ -t 0 ]; then
        # stdin is a TTY → fully interactive
        read -rp "$prompt [y/n] (default: $default): " answer
    else
        # piped install → try to read from /dev/tty
        if [ -e /dev/tty ]; then
            read -rp "$prompt [y/n] (default: $default): " answer < /dev/tty
        else
            echo
            echo "No interactive terminal detected — defaulting to '$default'"
            answer="$default"
        fi
    fi

    answer="${answer:-$default}"

    case "$answer" in
        [Yy]*) return 0 ;;
        [Nn]*) return 1 ;;
        *) return 0 ;; # default
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
    -DCCUTILS_ENABLE_MPI="$ENABLE_MPI"

echo "[4] Building..."
cmake --build build --parallel 

echo "[5] Installing to $INSTALL_DIR ..."
cmake --install build --prefix "$INSTALL_DIR"

#####################################
# ENVIRONMENT UPDATES
#####################################

echo "[6] Updating shell configuration..."

RC_FILE=$(detect_shell_rc)

echo "Using RC file: $RC_FILE"

# Update CMAKE_PREFIX_PATH
ensure_line_present "$RC_FILE" \
"export CMAKE_PREFIX_PATH=\"$INSTALL_DIR/lib/cmake/ccutils:\$CMAKE_PREFIX_PATH\""
ensure_line_present "$RC_FILE" \
"export CCUTILS_INCLUDE=\"$INSTALL_DIR/include/ccutils\""

echo
echo "Installation complete."
echo "To activate changes, run:"
echo "    source \"$RC_FILE\""
echo
