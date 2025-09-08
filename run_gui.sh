#!/bin/bash

# Reddit Posts Insights Viewer Launcher
echo "🚀 Starting Reddit Posts Insights Viewer..."

# Check if we're in the correct directory
if [ ! -f "data/db.sqlite" ]; then
  echo "❌ Error: Please run this script from the Reddit_Scrapper root directory"
  echo "   Current directory: $(pwd)"
  echo "   Expected files: data/db.sqlite"
  exit 1
fi

# Launch the Streamlit app
echo "🌐 Opening interface in your browser..."
streamlit run gui/gui.py --server.port 8501 --server.address localhost

echo "✅ Interface closed"
