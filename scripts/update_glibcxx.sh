#!/bin/bash

# Function to check if GLIBCXX_3.4.32 is available
check_glibcxx_version() {
    strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep -q "GLIBCXX_3.4.32"
}

# Check if we already have the required version
if check_glibcxx_version; then
    echo "GLIBCXX_3.4.32 is already available"
    exit 0
fi

# Update GLIBCXX to version 3.4.32
echo "Updating GLIBCXX to version 3.4.32..."
echo 'deb http://deb.debian.org/debian testing main' >> /etc/apt/sources.list
apt-get update
apt-get install -y -t testing libstdc++6

# Verify the update
if check_glibcxx_version; then
    echo "Successfully updated to GLIBCXX_3.4.32"
    echo "Available GLIBCXX versions:"
    strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX
else
    echo "Failed to update GLIBCXX to version 3.4.32"
    exit 1
fi 