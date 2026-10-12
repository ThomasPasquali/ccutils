#!/usr/bin/env bash
set -euo pipefail

#####################################
# CONFIGURATION
#####################################

INSTALL_PREFIX="$HOME/.local"
PKG_DIR="$INSTALL_PREFIX/share/ccutils"
INSTALL_DIR="$INSTALL_PREFIX/ccutils/install"
INSTALL_BIN="$INSTALL_PREFIX/bin"

REPO_URL="https://github.com/ThomasPasquali/ccutils.git"
REPO_BRANCH="main"
REPO_TAG=""
BRANCH_SPECIFIED=0
TAG_SPECIFIED=0
SOURCE_DIR=""

usage() {
    cat <<'USAGE'
Usage: install.sh [--branch NAME | --tag NAME] [--repo PATH]

Install from the default CCUTILS repository, or pass the path to an existing
local CCUTILS clone with --repo. A supplied clone is used as-is and is not
pulled or otherwise updated. --branch and --tag select a remote ref and cannot
be combined with --repo or with each other. The default branch is main.
USAGE
}

while (($#)); do
    case "$1" in
        --repo)
            if (($# < 2)) || [[ -z "$2" ]]; then
                echo "Error: --repo requires a path to a local clone." >&2
                usage >&2
                exit 2
            fi
            SOURCE_DIR="$2"
            shift 2
            ;;
        --branch)
            if (($# < 2)) || [[ -z "$2" ]]; then
                echo "Error: --branch requires a branch name." >&2
                usage >&2
                exit 2
            fi
            if ((TAG_SPECIFIED)); then
                echo "Error: --branch and --tag cannot be used together." >&2
                usage >&2
                exit 2
            fi
            REPO_BRANCH="$2"
            BRANCH_SPECIFIED=1
            shift 2
            ;;
        --tag)
            if (($# < 2)) || [[ -z "$2" ]]; then
                echo "Error: --tag requires a tag name." >&2
                usage >&2
                exit 2
            fi
            if ((BRANCH_SPECIFIED)); then
                echo "Error: --branch and --tag cannot be used together." >&2
                usage >&2
                exit 2
            fi
            REPO_TAG="$2"
            TAG_SPECIFIED=1
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -n "$SOURCE_DIR" && ( $BRANCH_SPECIFIED -eq 1 || $TAG_SPECIFIED -eq 1 ) ]]; then
    echo "Error: --branch and --tag apply only to the default upstream clone, not --repo." >&2
    usage >&2
    exit 2
fi

if [[ -n "$SOURCE_DIR" ]]; then
    if [[ ! -d "$SOURCE_DIR" ]]; then
        echo "Error: repository directory does not exist: $SOURCE_DIR" >&2
        exit 2
    fi
    SOURCE_DIR="$(cd -- "$SOURCE_DIR" && pwd -P)"
    GIT_ROOT="$(git -C "$SOURCE_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
    if [[ ! -f "$SOURCE_DIR/CMakeLists.txt" ]] || [[ "$GIT_ROOT" != "$SOURCE_DIR" ]]; then
        echo "Error: --repo must point to the root of a Git checkout containing CMakeLists.txt." >&2
        exit 2
    fi
fi

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
if [[ -n "$SOURCE_DIR" ]]; then
    echo "Source clone: $SOURCE_DIR"
elif [[ -n "$REPO_TAG" ]]; then
    echo "Source tag: $REPO_TAG"
else
    echo "Source branch: $REPO_BRANCH"
fi
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

GREEN="\033[32m"
RESET="\033[0m"

if [[ -n "$SOURCE_DIR" ]]; then
    echo -e "${GREEN}[1] Using local repository clone at $SOURCE_DIR ...${RESET}"
    BUILD_SOURCE_DIR="$SOURCE_DIR"
else
    echo -e "${GREEN}[1] Creating installation directory at $PKG_DIR ...${RESET}"
    mkdir -p "$PKG_DIR"
    echo -e "${GREEN}[2] Cloning or updating repository...${RESET}"
    INSTALL_REF="$REPO_BRANCH"
    if [[ -n "$REPO_TAG" ]]; then
        INSTALL_REF="$REPO_TAG"
    fi
    if [ ! -d "$PKG_DIR/.git" ]; then
        git clone --branch "$INSTALL_REF" --single-branch "$REPO_URL" "$PKG_DIR"
    elif [[ -n "$REPO_TAG" ]]; then
        git -C "$PKG_DIR" fetch --tags origin
        if ! git -C "$PKG_DIR" rev-parse --verify --quiet "refs/tags/$REPO_TAG^{commit}" >/dev/null; then
            echo "Error: tag '$REPO_TAG' was not found in $REPO_URL." >&2
            exit 2
        fi
        git -C "$PKG_DIR" checkout --detach "refs/tags/$REPO_TAG"
    else
        echo "Repository already exists. Updating branch '$REPO_BRANCH'..."
        git -C "$PKG_DIR" fetch origin \
            "+refs/heads/$REPO_BRANCH:refs/remotes/origin/$REPO_BRANCH"
        if git -C "$PKG_DIR" show-ref --verify --quiet "refs/heads/$REPO_BRANCH"; then
            git -C "$PKG_DIR" checkout "$REPO_BRANCH"
            git -C "$PKG_DIR" pull --rebase origin "$REPO_BRANCH"
        else
            git -C "$PKG_DIR" checkout --track -b "$REPO_BRANCH" "origin/$REPO_BRANCH"
        fi
    fi
    BUILD_SOURCE_DIR="$PKG_DIR"
fi
mkdir -p "$INSTALL_BIN"

cd "$BUILD_SOURCE_DIR"

echo -e "${GREEN}[3] Configuring CMake...${RESET}"
cmake -B build -S . \
    -DCCUTILS_ENABLE_CUDA="$ENABLE_CUDA" \
    -DCCUTILS_ENABLE_MPI="$ENABLE_MPI"

echo -e "${GREEN}[4] Building...${RESET}"
cmake --build build --parallel 

echo -e "${GREEN}[5] Installing to $INSTALL_DIR ...${RESET}"
cmake --install build --prefix "$INSTALL_DIR"

#####################################
# ENVIRONMENT UPDATES
#####################################

echo -e "${GREEN}[6] Updating shell configuration...${RESET}"

RC_FILE=$(detect_shell_rc)

echo "Using RC file: $RC_FILE"

# Update CMAKE_PREFIX_PATH
ensure_line_present "$RC_FILE" \
"export CMAKE_PREFIX_PATH=\"$INSTALL_DIR:\$CMAKE_PREFIX_PATH\""
ensure_line_present "$RC_FILE" \
"export CCUTILS_INCLUDE=\"$INSTALL_DIR/include\""
PYTHON_SITE_REL=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib', vars={'base': '_CCUTILS_PREFIX_', 'platbase': '_CCUTILS_PREFIX_'}).replace('_CCUTILS_PREFIX_', '', 1).lstrip('/\\\\'))")
ensure_line_present "$RC_FILE" \
"export PYTHONPATH=\"$INSTALL_DIR/$PYTHON_SITE_REL:\$PYTHONPATH\""
ensure_line_present "$RC_FILE" \
"export PATH=\"$INSTALL_BIN:\$PATH\""

# Add the upstream updater only for the default source. A custom clone may be
# a fork or a worktree, so replacing it with an upstream updater would switch
# the next upgrade to a different source.
if [[ -z "$SOURCE_DIR" ]]; then
    cd "$INSTALL_BIN"
    wget "https://raw.githubusercontent.com/ThomasPasquali/ccutils/${INSTALL_REF}/install.sh"
    mv install.sh ccutils-upgrade
    chmod +x ccutils-upgrade
fi

echo
echo -e "${GREEN}Installation complete."
echo -e "To activate changes, run:${RESET}"
echo "    source \"$RC_FILE\""
echo
if [[ -z "$SOURCE_DIR" ]]; then
    echo -e "${GREEN}To upgrade CCUTILS, run:${RESET}"
    echo "    ccutils-upgrade"
else
    echo "This install used a local clone. Re-run install.sh --repo with that clone to update it."
fi
echo
