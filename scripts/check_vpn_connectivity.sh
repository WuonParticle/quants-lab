#!/bin/bash
set -e

echo "=== Checking VPN connectivity ==="

# Kill any existing port forwarding processes
pkill -f "nc -l -p 5432" || true
pkill -f "nc -l -p 27017" || true
pkill -f "socat.*:5432" || true
pkill -f "socat.*:27017" || true

echo "Starting socat port forwarding..."

# Allow custom IPs via environment variables
TIMESCALE_HOST=${TIMESCALE_HOST:-"timescaledb"}
MONGO_HOST=${MONGO_HOST:-"mongodb"}

# Get IPs from DNS (may fail in some setups)
TIMESCALE_IP=$(getent hosts $TIMESCALE_HOST | awk "{ print \$1 }")
MONGO_IP=$(getent hosts $MONGO_HOST | awk "{ print \$1 }")

# If DNS lookup failed, try Docker inspection
if [ -z "$TIMESCALE_IP" ]; then
    echo "DNS lookup for $TIMESCALE_HOST failed, trying Docker inspect..."
    TIMESCALE_IP=$(docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $TIMESCALE_HOST 2>/dev/null || echo "")
fi

if [ -z "$MONGO_IP" ]; then
    echo "DNS lookup for $MONGO_HOST failed, trying Docker inspect..."
    MONGO_IP=$(docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $MONGO_HOST 2>/dev/null || echo "")
fi

# Last resort: Try to use docker network inspect to find the IPs
if [ -z "$TIMESCALE_IP" ] || [ -z "$MONGO_IP" ]; then
    echo "Docker inspect failed, trying network inspection..."
    NETWORK=${DOCKER_NETWORK:-"hummingbot-vpn"}
    
    # Get all container IPs in the network
    NETWORK_INFO=$(docker network inspect $NETWORK 2>/dev/null || echo "")
    
    if [ -n "$NETWORK_INFO" ]; then
        # Try to extract Timescale IP based on container name
        if [ -z "$TIMESCALE_IP" ]; then
            TIMESCALE_IP=$(echo "$NETWORK_INFO" | grep -A 10 "\"Name\": \"$TIMESCALE_HOST\"" | grep -oP '"IPv4Address": "\K[^/\"]+')
        fi
        
        # Try to extract MongoDB IP based on container name
        if [ -z "$MONGO_IP" ]; then
            MONGO_IP=$(echo "$NETWORK_INFO" | grep -A 10 "\"Name\": \"$MONGO_HOST\"" | grep -oP '"IPv4Address": "\K[^/\"]+')
        fi
    fi
fi

# Fail if we still don't have IPs
if [ -z "$TIMESCALE_IP" ]; then
    echo "ERROR: Could not determine TimescaleDB IP address."
    echo "Please set TIMESCALE_HOST or use docker-compose with proper networking."
    exit 1
fi

if [ -z "$MONGO_IP" ]; then
    echo "ERROR: Could not determine MongoDB IP address."
    echo "Please set MONGO_HOST or use docker-compose with proper networking."
    exit 1
fi

echo "Database IPs: TimescaleDB ($TIMESCALE_HOST)=$TIMESCALE_IP, MongoDB ($MONGO_HOST)=$MONGO_IP"

# Start port forwarding for TimescaleDB
socat TCP4-LISTEN:5432,fork,reuseaddr TCP4:$TIMESCALE_IP:5432 &

# Start port forwarding for MongoDB
socat TCP4-LISTEN:27017,fork,reuseaddr TCP4:$MONGO_IP:27017 &

# Verify port forwarding
sleep 2
if nc -z -4 localhost 5432; then
    echo "✅ Port forwarding for TimescaleDB is active"
else
    echo "⚠️ Port forwarding for TimescaleDB failed to start"
    exit 1
fi

if nc -z -4 localhost 27017; then
    echo "✅ Port forwarding for MongoDB is active"
else
    echo "⚠️ Port forwarding for MongoDB failed to start"
    exit 1
fi

echo "Port forwarding setup complete." 