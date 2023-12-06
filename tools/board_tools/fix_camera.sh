#!/bin/bash

v4l2-ctl -d /dev/video0 --set-ctrl gain_automatic=1
v4l2-ctl -d /dev/video0 --set-ctrl white_balance_automatic=1
