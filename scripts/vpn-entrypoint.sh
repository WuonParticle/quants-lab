#!/bin/bash
echo "=== VPN Container Starting ==="
date

# Get environment variables for services
export TIMESCALE_HOST=${TIMESCALE_HOST:-"timescaledb"}
export MONGO_HOST=${MONGO_HOST:-"mongodb"}
export DOCKER_NETWORK=${DOCKER_NETWORK:-"hummingbot-vpn"}

echo "Configured to connect to:"
echo "- TimescaleDB: $TIMESCALE_HOST"
echo "- MongoDB: $MONGO_HOST"
echo "- Network: $DOCKER_NETWORK"

# Run connectivity check script
/check_vpn_connectivity.sh

# Check if we should run OpenVPN
if [ -f "/vpn/config.ovpn" ]; then
    # Copy the OpenVPN config
    cp /vpn/config.ovpn /tmp/vpn.ovpn

    # Check if auth.txt exists and add it to config if not already present
    if [ -f "/vpn/auth.txt" ]; then
        echo "Using authentication from /vpn/auth.txt"
        # Fix permissions on auth file
        chmod 600 /vpn/auth.txt
        # Remove existing auth-user-pass lines
        sed -i "/auth-user-pass/d" /tmp/vpn.ovpn
        # Add the auth-user-pass line
        echo "auth-user-pass /vpn/auth.txt" >> /tmp/vpn.ovpn
    fi

    # Add DNS configuration to the OpenVPN config
    echo "Adding DNS configuration to OpenVPN config..."
    echo "dhcp-option DNS 1.1.1.1" >> /tmp/vpn.ovpn
    echo "dhcp-option DNS 8.8.8.8" >> /tmp/vpn.ovpn

    # Start OpenVPN in the foreground
    echo "Starting OpenVPN..."
    exec openvpn --config /tmp/vpn.ovpn --verb 3 --auth-nocache --auth-retry nointeract
else
    echo "No OpenVPN config found, running in port-forwarding-only mode"
    # Keep container running
    exec tail -f /dev/null
fi 