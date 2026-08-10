#!/bin/bash

echo "=== 5GSOS Rover Bridge Startup ==="

# Load environment variables from .env if exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "Loaded environment from .env"
fi

# Display current configuration
echo ""
echo "Configuration:"
echo "  BACKEND_URL: ${BACKEND_URL:-http://localhost:5000}"
echo "  ROVER_ID: ${ROVER_ID:-ROVER-1}"
echo "  SERIAL_PORT: ${SERIAL_PORT:-/dev/ttyACM0}"
echo "  BAUD_RATE: ${BAUD_RATE:-115200}"
echo ""

# Check if serial port exists
SERIAL_PORT=${SERIAL_PORT:-/dev/ttyACM0}
if [ ! -e "$SERIAL_PORT" ]; then
    echo "ERROR: Serial port $SERIAL_PORT not found!"
    echo "Available serial ports:"
    ls -l /dev/tty* 2>/dev/null | grep -o '/dev/tty[^ ]*'
    exit 1
fi

echo "Starting rover bridge..."
node bridge.js