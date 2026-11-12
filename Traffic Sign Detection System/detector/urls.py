from django.urls import path
from . import views

urlpatterns = [
    # Public Pages
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('signs/', views.traffic_signs_guide, name='traffic_signs_guide'),
    path('feedback/', views.feedback_view, name='feedback'),
    path('contact/', views.contact, name='contact'),

    # User Authentication (same login for user and admin)
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),

    # Authenticated Normal User Pages
    path('dashboard/', views.dashboard, name='dashboard'),
    path('detect/', views.detect, name='detect'),
    path('result/<int:pk>/', views.result, name='detection_result'),
    path('history/', views.history, name='history'),
    path('history/delete/<int:pk>/', views.delete_detection, name='delete_detection'),
    path('history/clear/', views.clear_history, name='clear_history'),

    # Authenticated Admin Dashboard (Staff Only)
    path('admin-dashboard/', views.admin_dashboard, name='admin_dashboard'),
]
