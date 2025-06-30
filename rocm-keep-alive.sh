#!/bin/bash
while true; do
    sleep 2;
    # Switch tabs at random every 2 seconds
    xdotool key --delay 100 alt+Tab;
    # Simulate a mouse movement to keep the session active
    xdotool mousemove_relative -- 5 5;
    # Click random stuff
    xdotool click 1;
    # Switch tabs
    xdotool key --delay 100 ctrl+Tab;
        # Simulate a mouse movement to keep the session active
    xdotool mousemove_relative -- 50 50;
    # Click random stuff
    xdotool click 1;
    # Wait for 10 seconds before the next iteration
    sleep 18;
done                                        