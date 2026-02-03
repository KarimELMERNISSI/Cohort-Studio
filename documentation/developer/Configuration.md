# Configuration Documentation

## Overview

The `config/` directory contains JSON files that define various settings for the application, such as variables, labels, units, and application-specific parameters.

## Files

- **config.json**: The main configuration file used by the application in production.
- **config-cardiateam-2025.json**: A specific configuration version for the Cardiateam 2025 cohort.
- **config copy.json**: Logic backup or template.

## Structure

The configuration files typically contain:

1. **Application Settings**: Global flags and paths.
2. **Variable Definitions**: Metadata for variables used in the app (e.g., standard names, types).
3. **UI Labels**: Strings used in the frontend to allow easy text updates.
4. **Enrichment Rules**: Definitions for how certain variables should be calculated or enriched.

## Usage

Configuration is loaded via `utils.config_loader.py`. The application reads these files at startup to initialize the global state.
