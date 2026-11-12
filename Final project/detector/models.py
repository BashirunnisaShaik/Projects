from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User
import os


class TrafficSignDetection(models.Model):
    """
    Stores traffic sign detection records including user ownership, original
    and annotated images, recognized sign name, category, meaning, driver action,
    confidence score, dynamic processing time, and bounding coordinates.
    """
    CATEGORY_CHOICES = [
        ('regulatory', 'Regulatory'),
        ('prohibitory', 'Prohibited / Prohibitory'),
        ('warning', 'Warning'),
        ('mandatory', 'Mandatory'),
        ('informative', 'Information'),
        ('none', 'No Sign Detected'),
    ]

    user = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='detections'
    )
    title = models.CharField(max_length=150, blank=True, default='Traffic Scan')
    original_image = models.ImageField(upload_to='uploads/%Y/%m/%d/')
    processed_image = models.ImageField(upload_to='processed/%Y/%m/%d/', blank=True, null=True)
    cropped_sign = models.ImageField(upload_to='crops/%Y/%m/%d/', blank=True, null=True)

    # Detection outcomes
    sign_detected = models.BooleanField(default=False)
    sign_name = models.CharField(max_length=100, default='No Sign Detected')
    category = models.CharField(max_length=30, choices=CATEGORY_CHOICES, default='none')
    confidence = models.FloatField(default=0.0, help_text="Detection confidence in percentage")

    # Explanatory safety information
    meaning = models.TextField(blank=True, default='', help_text="What the traffic sign means")
    driver_action = models.TextField(blank=True, default='', help_text="What action the driver must take")

    # Computer vision analysis metadata
    detected_shape = models.CharField(max_length=50, blank=True, default='Unknown')
    dominant_color = models.CharField(max_length=50, blank=True, default='Unknown')
    bounding_box = models.CharField(max_length=100, blank=True, default='', help_text="x,y,w,h coordinates")
    processing_time = models.FloatField(default=0.0, help_text="Dynamically calculated processing time in seconds")
    description = models.TextField(blank=True, help_text="Sign meaning or summary instruction")

    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Traffic Sign Detection'
        verbose_name_plural = 'Traffic Sign Detections'

    def __str__(self):
        return f"{self.sign_name} ({self.created_at.strftime('%Y-%m-%d %H:%M')})"

    @property
    def filename(self):
        return os.path.basename(self.original_image.name) if self.original_image else ''


class Feedback(models.Model):
    """
    Stores user and visitor feedback submitted from the Feedback page.
    """
    user = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='feedbacks'
    )
    name = models.CharField(max_length=100)
    email = models.EmailField(blank=True, default='')
    rating = models.IntegerField(default=5, help_text="Rating from 1 to 5 stars")
    message = models.TextField(help_text="Feedback message")
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'User Feedback'
        verbose_name_plural = 'User Feedback'

    def __str__(self):
        return f"Feedback from {self.name} ({self.rating}★) - {self.created_at.strftime('%Y-%m-%d')}"


class ContactMessage(models.Model):
    """
    Stores feedback and inquiries sent from the contact page.
    """
    name = models.CharField(max_length=100)
    email = models.EmailField()
    subject = models.CharField(max_length=150)
    message = models.TextField()
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Contact Message'
        verbose_name_plural = 'Contact Messages'

    def __str__(self):
        return f"Message from {self.name} - {self.subject}"
