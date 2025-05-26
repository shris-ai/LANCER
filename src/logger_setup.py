import logging
import os
from datetime import datetime


log_dir = datetime.now().strftime("logs/%Y-%m-%d_%H-%M-%S")
os.makedirs(log_dir, exist_ok=True) 

log_file = os.path.join(log_dir, 'app.log')

logging.basicConfig(
    filename=log_file,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)

logger = logging.getLogger()


