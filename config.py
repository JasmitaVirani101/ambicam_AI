import os

class Config:
    DEBUG = True
    SMTP_SERVER = os.getenv('SMTP_SERVER')
    SMTP_PORT = os.getenv('SMTP_PORT', 587)
    SMTP_USERNAME = os.getenv('SMTP_USERNAME')
    SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
    SENDER_EMAIL = os.getenv('SENDER_EMAIL')
    RECIPIENT_EMAIL = os.getenv('RECIPIENT_EMAIL')
    MODEL_BASE_PATH = os.getenv('MODEL_BASE_PATH', 'prebuilt_model/')
