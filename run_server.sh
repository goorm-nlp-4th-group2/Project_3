#!/bin/sh

gunicorn app:app -b 0.0.0.0:5001 -w 1 --timeout=120
