from .detr import build
import logging
logger = logging.getLogger(__name__)

def build_model(args):
    logger.info("Building Sliced DETR model...")
    return build(args)