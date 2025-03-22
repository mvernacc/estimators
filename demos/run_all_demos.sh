#!/bin/bash

for file in *.py; do
  if [ -f "$file" ]; then
    python "$file"
  fi
done
