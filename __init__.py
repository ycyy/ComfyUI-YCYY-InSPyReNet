from .inspyrenet import InspyrenetRembg, InspyrenetRembgAdvanced

NODE_CLASS_MAPPINGS = {
    "YCYY-InspyrenetRembg" : InspyrenetRembg,
    "YCYY-InspyrenetRembgAdvanced" : InspyrenetRembgAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YCYY-InspyrenetRembg": "Inspyrenet Rembg",
    "YCYY-InspyrenetRembgAdvanced": "Inspyrenet Rembg Advanced"
}
__all__ = ['NODE_CLASS_MAPPINGS', "NODE_DISPLAY_NAME_MAPPINGS"]
