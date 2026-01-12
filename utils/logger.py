import logging
import os
from datetime import datetime

class LoggerSetup:
    @staticmethod
    def setup_logger(name='NN_Project', log_dir='logs'):
        """
        Sets up a logger that writes to console (INFO) and file (DEBUG).
        """
        # 1. Create logs directory if it doesn't exist
        # We look for the 'logs' folder inside the project root or current dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 2. Generate a unique filename based on time
        # Example: training_2026-01-11_15-30-00.log
        log_filename = datetime.now().strftime("training_%Y-%m-%d_%H-%M-%S.log")
        log_filepath = os.path.join(log_dir, log_filename)

        # 3. Initialize Logger
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG) # The logger itself captures everything

        # Prevent duplicate logs if setup is called multiple times
        if logger.hasHandlers():
            logger.handlers.clear()

        # 4. Create Handlers
        
        # A. Console Handler (Standard Output)
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.INFO) # Only show important stuff on screen

        # B. File Handler (Save to disk)
        f_handler = logging.FileHandler(log_filepath)
        f_handler.setLevel(logging.DEBUG) # Save EVERYTHING to file

        # 5. Create Formatter
        # Format: [Time] - [Level] - Message
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(formatter)
        f_handler.setFormatter(formatter)

        # 6. Add Handlers to Logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

        return logger