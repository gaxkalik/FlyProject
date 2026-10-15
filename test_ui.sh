#!/usr/bin/env bash
# ==============================================================================
# FlyMesh: Interactive UI Testing & Dashboard Launch Utility
# ==============================================================================
# Provides automated UI integrity verification, interactive web dashboard serving,
# and visual inspection tools for the Drosophila Connectome Swarm Mesh project.
#
# Usage:
#   ./test_ui.sh                 # Interactive terminal UI testbench menu
#   ./test_ui.sh --serve         # Launch local HTTP server and open browser
#   ./test_ui.sh --test          # Run automated headless UI & DOM test suite
#   ./test_ui.sh --plots         # Open all generated scientific figures
#   ./test_ui.sh --port <PORT>   # Specify custom server port (default: 8080)
#   ./test_ui.sh --help          # Show command-line help
# ==============================================================================

set -e

# Resolve repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ANSI Terminal Colors
BOLD='\033[1m'
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

PORT=8080
SERVER_PID=""

# Graceful cleanup on exit
cleanup() {
    if [ -n "$SERVER_PID" ]; then
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            echo -e "\n${YELLOW}[UI Server] Shutting down background server (PID: $SERVER_PID)...${NC}"
            kill "$SERVER_PID" 2>/dev/null || true
            wait "$SERVER_PID" 2>/dev/null || true
        fi
    fi
}
trap cleanup EXIT INT TERM

# Helper to open URLs or files across OS platforms
open_target() {
    local target="$1"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open "$target" 2>/dev/null || true
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v xdg-open &>/dev/null; then
            xdg-open "$target" 2>/dev/null || true
        elif command -v sensible-browser &>/dev/null; then
            sensible-browser "$target" 2>/dev/null || true
        fi
    elif [[ "$OSTYPE" == "msys"* || "$OSTYPE" == "cygwin"* ]]; then
        start "$target" 2>/dev/null || true
    fi
}

# Function to run automated UI tests
run_automated_tests() {
    echo -e "\n${CYAN}${BOLD}======================================================================${NC}"
    echo -e "${CYAN}${BOLD}RUNNING AUTOMATED UI & DOM VERIFICATION TESTS${NC}"
    echo -e "${CYAN}${BOLD}======================================================================${NC}\n"

    if python3 tests/test_ui.py; then
        echo -e "\n${GREEN}${BOLD}✓ ALL UI INTEGRITY TESTS PASSED!${NC}\n"
        return 0
    else
        echo -e "\n${RED}${BOLD}✗ UI TESTS FAILED!${NC}\n"
        return 1
    fi
}

# Function to launch the interactive web dashboard
serve_dashboard() {
    local requested_port="${1:-$PORT}"

    # Verify if port is already in use, increment if needed
    while lsof -Pi :"$requested_port" -sTCP:LISTEN -t &>/dev/null; do
        echo -e "${YELLOW}[Warning] Port $requested_port is busy, checking next port...${NC}"
        requested_port=$((requested_port + 1))
    done

    local dashboard_url="http://localhost:${requested_port}/ui/dashboard.html"

    echo -e "\n${BLUE}${BOLD}======================================================================${NC}"
    echo -e "${BLUE}${BOLD}LAUNCHING FLYMESH INTERACTIVE WEB DASHBOARD${NC}"
    echo -e "${BLUE}${BOLD}======================================================================${NC}"
    echo -e "  • Dashboard URL:  ${CYAN}${BOLD}${dashboard_url}${NC}"
    echo -e "  • Project Root:   ${SCRIPT_DIR}"
    echo -e "  • Serving Port:   ${requested_port}"
    echo -e "${BLUE}======================================================================${NC}\n"

    # Start local Python HTTP server
    python3 -m http.server "$requested_port" --directory "$SCRIPT_DIR" > /dev/null 2>&1 &
    SERVER_PID=$!

    # Wait briefly for server to bind
    sleep 0.8

    echo -e "${GREEN}${BOLD}✓ Web server active (PID: $SERVER_PID)${NC}"
    echo -e "${PURPLE}Opening browser at: ${dashboard_url}${NC}"
    open_target "$dashboard_url"

    echo -e "\n${YELLOW}${BOLD}[Interactive Controls Guide]:${NC}"
    echo -e "  1. Click ${BLUE}'Trigger Convective Shock'${NC} to simulate a localized storm hitting Fly 0."
    echo -e "  2. Toggle ${CYAN}Visual (LC11)${NC}, ${CYAN}Acoustic (bLN1)${NC}, and ${CYAN}Pheromone${NC} modalities."
    echo -e "  3. Enable ${RED}'Optical Fog Corridor'${NC} to test multi-modal acoustic/chemical penetration."
    echo -e "  4. Enable ${RED}'25% Node Mortality'${NC} to test anti-fragile swarm rerouting."
    echo -e "  5. Press ${BOLD}Ctrl+C${NC} in this terminal when finished to stop the server.\n"

    # Wait on background server process
    wait "$SERVER_PID" 2>/dev/null || true
}

# Function to open generated milestone plots
open_plots() {
    echo -e "\n${PURPLE}${BOLD}======================================================================${NC}"
    echo -e "${PURPLE}${BOLD}OPENING SCIENTIFIC BENCHMARK FIGURES${NC}"
    echo -e "${PURPLE}${BOLD}======================================================================${NC}\n"

    if ls plots/*.png 1> /dev/null 2>&1; then
        echo -e "${GREEN}Opening all high-resolution figures in system viewer...${NC}"
        for f in plots/*.png; do
            echo -e "  • ${f}"
            open_target "$f"
        done
        echo -e "\n${GREEN}${BOLD}✓ Figures opened successfully!${NC}\n"
    else
        echo -e "${RED}No generated plots found in plots/. Run milestones first via python3 run_tests.py --all.${NC}\n"
    fi
}

# Interactive CLI menu
interactive_menu() {
    while true; do
        echo -e "${CYAN}${BOLD}======================================================================${NC}"
        echo -e "${CYAN}${BOLD}       FLYMESH: DROSOPHILA CONNECTOME SWARM TESTBENCH & UI            ${NC}"
        echo -e "${CYAN}${BOLD}======================================================================${NC}"
        echo -e "  ${BOLD}1)${NC} ${GREEN}Launch Interactive Web UI Dashboard${NC} (starts server & opens browser)"
        echo -e "  ${BOLD}2)${NC} ${BLUE}Run Automated UI & DOM Integrity Tests${NC} (tests/test_ui.py)"
        echo -e "  ${BOLD}3)${NC} ${PURPLE}View Scientific Milestone Visualizations${NC} (opens plots in viewer)"
        echo -e "  ${BOLD}4)${NC} ${YELLOW}Run Fast Physics & Connectome Unit Tests${NC} (python3 run_tests.py)"
        echo -e "  ${BOLD}5)${NC} ${CYAN}Run All 6 Milestones Benchmark${NC} (python3 run_tests.py --all)"
        echo -e "  ${BOLD}6)${NC} Exit"
        echo -e "${CYAN}----------------------------------------------------------------------${NC}"
        read -r -p "Select an option [1-6]: " choice

        case "$choice" in
            1)
                serve_dashboard "$PORT"
                ;;
            2)
                run_automated_tests
                ;;
            3)
                open_plots
                ;;
            4)
                echo -e "\n${YELLOW}Running Unit Tests...${NC}"
                python3 run_tests.py
                echo ""
                ;;
            5)
                echo -e "\n${CYAN}Running Full Benchmark Suite (Milestones 1 to 6)...${NC}"
                python3 run_tests.py --all
                echo ""
                ;;
            6|q|Q)
                echo -e "\n${GREEN}Exiting FlyMesh UI Testbench. Goodbye!${NC}\n"
                exit 0
                ;;
            *)
                echo -e "\n${RED}Invalid option '${choice}'. Please select 1 through 6.${NC}\n"
                ;;
        esac
    done
}

# Print CLI usage help
show_help() {
    cat << EOF
FlyMesh UI Testing & Dashboard CLI

Usage:
  ./test_ui.sh [options]

Options:
  -s, --serve          Start local web server and launch interactive dashboard in browser
  -t, --test           Run automated headless UI and DOM integrity test suite
  -p, --plots          Open all 9 generated scientific milestone figures in system viewer
  -u, --unit           Run fast unit tests for physics, connectome, and meteorology
  -a, --all            Run unit tests, all 6 milestones, and UI automated verification
  --port <PORT>        Specify local HTTP server port (default: 8080)
  -h, --help           Display this help manual

Examples:
  ./test_ui.sh                       # Launch interactive terminal selection menu
  ./test_ui.sh --serve               # Directly serve and open dashboard
  ./test_ui.sh --test                # Run automated UI tests for CI/CD
  ./test_ui.sh --serve --port 9090   # Serve on custom port 9090
EOF
}

# Parse CLI arguments
if [ $# -eq 0 ]; then
    interactive_menu
    exit 0
fi

while [ $# -gt 0 ]; do
    case "$1" in
        -s|--serve|--open)
            serve_dashboard "$PORT"
            exit 0
            ;;
        -t|--test|--auto)
            run_automated_tests
            exit $?
            ;;
        -p|--plots)
            open_plots
            exit 0
            ;;
        -u|--unit)
            python3 run_tests.py
            exit $?
            ;;
        -a|--all)
            python3 run_tests.py --all
            run_automated_tests
            exit $?
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done
