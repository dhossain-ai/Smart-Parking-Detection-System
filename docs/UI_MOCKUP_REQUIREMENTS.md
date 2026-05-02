# UI Mockup Requirements

## Goal

Create a modern local Streamlit dashboard for the Smart Parking Detection System.

The app should look like an offline monitoring tool.

## Main Layout

### Top Section

Show:

- Project title: Smart Parking Detection System
- Subtitle: Offline Local Monitoring Tool
- Upload Image button
- Upload Video button
- Start Detection button
- Save Output button

### Sidebar Navigation

Include:

- Dashboard
- Image Detection
- Video Detection
- Compare Models
- Weather Robustness
- Metrics
- Explainability
- Calibration
- Settings

### Main Detection View

Use two columns:

Left:
- Original Feed

Right:
- Processed Feed

Processed feed should show:

- green overlays for vacant spaces
- red overlays for occupied spaces
- confidence scores
- slot IDs

### Right Summary Panel

Show:

- Total Slots
- Occupied
- Vacant
- Occupancy Rate
- Current Model
- Inference Time
- Weather
- Legend

### Bottom Analytics

Show:

- Occupancy trend chart
- Model comparison table
- Weather samples
- Confusion matrix when available

## Colors

- Green = Vacant
- Red = Occupied
- Yellow = Changed Status or uncertain
- Teal/blue = main UI accent

## App Modes

The app must support:

1. Image Detection
2. Video Detection
3. Classical vs CNN comparison
4. Weather Robustness
5. Metrics
6. Explainability