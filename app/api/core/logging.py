# # app/api/core/log_config.py
# import logging
# import sys
# from pathlib import Path

# def configure_logging(app_env: str = "development"):
#     """Configure logging system"""
    
#     # Clear existing handlers
#     logging.root.handlers = []
    
#     # Basic config
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#         handlers=[
#             logging.StreamHandler(sys.stdout)
#         ]
#     )
    
#     # Add file handler
#     log_file = Path("logs/app.log")
#     log_file.parent.mkdir(exist_ok=True)
#     file_handler = logging.FileHandler(log_file)
#     logging.getLogger().addHandler(file_handler)

#     if app_env == "production":
#         logging.getLogger().setLevel(logging.WARNING)

import logging
import sys
from pathlib import Path

def configure_logging():
    logger = logging.getLogger("oilapp")
    logger.setLevel(logging.INFO)
    
    # Create logs directory
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # File handler
    file_handler = logging.FileHandler(log_dir/"app.log")
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_format = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger