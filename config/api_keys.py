def get_api_key_by_name(name):
    return {
        "openai": "WRITE_YOUR_OPENAI_API_KEY_HERE",
        "google": "WRITE_YOUR_GOOGLE_API_KEY_HERE",
        "anthropic": "WRITE_YOUR_ANTHROPIC_API",
        "local": "local"
    }[name]