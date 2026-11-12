from django.contrib import admin
from .models import TrafficSignDetection, ContactMessage, Feedback


@admin.register(TrafficSignDetection)
class TrafficSignDetectionAdmin(admin.ModelAdmin):
    list_display = (
        'id', 'user', 'title', 'sign_name', 'category',
        'confidence', 'sign_detected', 'processing_time', 'created_at'
    )
    list_filter = ('sign_detected', 'category', 'created_at', 'user')
    search_fields = ('title', 'sign_name', 'description', 'meaning', 'driver_action', 'user__username')
    readonly_fields = ('created_at', 'processing_time', 'bounding_box')
    ordering = ('-created_at',)
    fieldsets = (
        ('Detection Info', {
            'fields': ('user', 'title', 'sign_name', 'category', 'sign_detected', 'confidence')
        }),
        ('Sign Description', {
            'fields': ('meaning', 'driver_action', 'description')
        }),
        ('Technical Details', {
            'fields': ('detected_shape', 'dominant_color', 'bounding_box', 'processing_time')
        }),
        ('Images', {
            'fields': ('original_image', 'processed_image', 'cropped_sign', 'filename')
        }),
        ('Timestamps', {
            'fields': ('created_at',)
        }),
    )


@admin.register(Feedback)
class FeedbackAdmin(admin.ModelAdmin):
    list_display = ('name', 'email', 'rating', 'user', 'created_at')
    list_filter = ('rating', 'created_at')
    search_fields = ('name', 'email', 'message')
    readonly_fields = ('created_at',)
    ordering = ('-created_at',)


@admin.register(ContactMessage)
class ContactMessageAdmin(admin.ModelAdmin):
    list_display = ('name', 'email', 'subject', 'created_at')
    search_fields = ('name', 'email', 'subject', 'message')
    readonly_fields = ('created_at',)
    ordering = ('-created_at',)

