---
title: 3D Map Backend
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---

# 3D Map Generator Backend

This is the backend for the 3D Map Generator application.
It uses FastAPI, Open3D, and Trimesh to generate 3D models from OSM data.

## Configuration

This Space expects the following Secrets:
- `FIREBASE_STORAGE_BUCKET`
- `FIREBASE_CREDENTIALS_JSON`
