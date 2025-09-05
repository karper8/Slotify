#!/usr/bin/env python3
"""
Warehouse Management System - Flask Backend
Run script with proper error handling and logging setup
"""

import os
import sys
import logging
from datetime import datetime

def setup_logging():
    """Setup logging configuration."""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = f"warehouse_app_{datetime.now().strftime('%Y%m%d')}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'flask', 'flask_cors', 'pandas', 'numpy', 
        'matplotlib', 'seaborn', 'sklearn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install them using: pip install -r requirements.txt")
        return False
    
    return True

def create_directories():
    """Create necessary directories."""
    directories = [
        'temp_uploads',
        'logs',
        'utils',
        'models'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
    # Create __init__.py files for Python packages
    init_files = [
        'utils/__init__.py',
        'models/__init__.py'
    ]
    
    for init_file in init_files:
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('# Package initialization\n')

def main():
    """Main function to run the application."""
    try:
        print("=" * 60)
        print("Warehouse Management System - Flask Backend")
        print("=" * 60)
        
        # Setup logging
        setup_logging()
        logger = logging.getLogger(__name__)
        logger.info("Starting Warehouse Management System")
        
        # Check dependencies
        print("Checking dependencies...")
        if not check_dependencies():
            logger.error("Dependencies check failed")
            sys.exit(1)
        print("✓ All dependencies are installed")
        
        # Create necessary directories
        print("Creating directories...")
        create_directories()
        print("✓ Directories created")
        
        # Import and run the Flask app
        print("Starting Flask application...")
        logger.info("Importing Flask application")
        
        try:
            from app import app
        except ImportError as e:
            logger.error(f"Failed to import Flask app: {str(e)}")
            print(f"Error importing Flask app: {str(e)}")
            sys.exit(1)
        
        # Start the application
        print("\n" + "=" * 60)
        print("Server starting on http://localhost:5000")
        print("Available endpoints:")
        print("  - POST /api/signup           - User registration")
        print("  - POST /api/login            - User login")
        print("  - POST /api/logout           - User logout")
        print("  - GET  /api/session          - Check session")
        print("  - POST /api/upload-dataset   - Upload CSV file")
        print("  - GET  /api/models           - Get available ML models")
        print("  - POST /api/process-slotting - Process slotting optimization")
        print("  - GET  /api/inventory-analytics - Get inventory analytics")
        print("  - GET  /api/health           - Health check")
        print("=" * 60)
        print("Press Ctrl+C to stop the server")
        print("=" * 60)
        
        logger.info("Flask application started successfully")
        
        # Run the app
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            use_reloader=False  # Disable reloader to prevent double execution
        )
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
        print("\n\nServer stopped by user")
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        print(f"Application error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()