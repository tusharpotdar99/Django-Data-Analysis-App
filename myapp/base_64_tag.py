from django import template
import base64

register = template.Library()

@register.filter
def base64encode(data):
    """Encodes data in Base64 format."""
    if isinstance(data, str):
        data = data.encode('utf-8')
    return base64.b64encode(data).decode('utf-8')
