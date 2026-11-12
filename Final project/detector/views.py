"""
Views for Traffic Sign Detection System.
Handles user authentication (register, login, logout), user dashboard,
admin dashboard, traffic sign detection, result visualization,
traffic signs guide, feedback submission, and MySQL history auditing.
"""

from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.core.paginator import Paginator
from django.core.files.base import ContentFile
from django.conf import settings
from django.utils import timezone
from django.db.models import Avg, Count, Q
import os
import time

from .models import TrafficSignDetection, ContactMessage, Feedback
from .forms import ImageUploadForm, ContactForm, UserRegisterForm, UserLoginForm, FeedbackForm
from .core.detection import (
    detect_traffic_signs,
    SIGN_CATALOG,
    CATEGORY_EXPLANATIONS,
    CATEGORY_DISPLAY_NAMES
)
from .core.preprocessing import load_image, load_image_from_bytes, save_image


def admin_required(view_func):
    """Decorator to ensure only authorized admin/staff users can access the view."""
    @login_required(login_url='login')
    def _wrapped_view(request, *args, **kwargs):
        if not request.user.is_staff:
            messages.error(request, "Access denied. Admin privileges are required to view that page.")
            return redirect('dashboard')
        return view_func(request, *args, **kwargs)
    return _wrapped_view


def home(request):
    """
    Clean, professional road/traffic-themed Home Page.
    Accessible to all visitors.
    """
    total_scans = TrafficSignDetection.objects.count()
    detected_count = TrafficSignDetection.objects.filter(sign_detected=True).count()
    recent_detections = TrafficSignDetection.objects.all()[:4]
    detection_rate = round((detected_count / total_scans * 100), 1) if total_scans > 0 else 95.0

    context = {
        'total_scans': total_scans,
        'detected_count': detected_count,
        'detection_rate': detection_rate,
        'recent_detections': recent_detections,
        'catalog': SIGN_CATALOG,
    }
    return render(request, 'detector/home.html', context)


def about(request):
    """
    Detailed and easy-to-understand About Project page.
    """
    return render(request, 'detector/about.html', {'catalog': SIGN_CATALOG})


def traffic_signs_guide(request):
    """
    Traffic Sign Guide page displaying signs organized into 5 clear categories:
    Warning, Regulatory, Mandatory, Prohibited / Prohibitory, Information.
    Shows Sign Name, Category, Meaning, Driver Action, Shape, and Dominant Color.
    """
    categories = {
        'warning': {
            'title': 'Warning Signs',
            'badge_class': 'badge-warning',
            'description': 'Cautionary indicators warning drivers about possible danger, hazards, or road changes ahead.',
            'signs': []
        },
        'regulatory': {
            'title': 'Regulatory Signs',
            'badge_class': 'badge-regulatory',
            'description': 'Signs that give mandatory rules and priorities that drivers must strictly follow.',
            'signs': []
        },
        'mandatory': {
            'title': 'Mandatory Signs',
            'badge_class': 'badge-mandatory',
            'description': 'Compulsory driving instructions such as mandatory turns, straight-ahead lanes, and roundabouts.',
            'signs': []
        },
        'prohibitory': {
            'title': 'Prohibited / Prohibitory Signs',
            'badge_class': 'badge-regulatory',
            'description': 'Signs indicating that a particular action, entry, or parking is strictly not allowed.',
            'signs': []
        },
        'informative': {
            'title': 'Information Signs',
            'badge_class': 'badge-information',
            'description': 'Guidance signs providing useful information such as hospitals, parking areas, and facilities.',
            'signs': []
        },
    }

    # Map sample images for visual representation where available
    sign_images = {
        'Stop Sign': 'sample_stop_sign.jpg',
        'No Entry': 'sample_no_entry.jpg',
        'No Parking': 'sample_no_entry.jpg',
        'Speed Limit Sign': 'sample_speed_limit_60.jpg',
        'Yield / Give Way': 'sample_yield_sign.jpg',
        'Turn Left': 'sample_turn_left.jpg',
        'Turn Right': 'sample_turn_right.jpg',
    }

    for name, data in SIGN_CATALOG.items():
        cat = data.get('category', 'regulatory')
        if cat in categories:
            categories[cat]['signs'].append({
                'name': name,
                'category': cat,
                'category_display': CATEGORY_DISPLAY_NAMES.get(cat, cat.title()),
                'shape': data.get('shape', 'Standard'),
                'color': data.get('dominant_color', 'Color'),
                'meaning': data.get('meaning', data.get('description', '')),
                'driver_action': data.get('driver_action', 'Observe road safety rules.'),
                'description': data.get('description', ''),
                'sample_image': sign_images.get(name, None),
            })

    return render(request, 'detector/guide.html', {'categories': categories})


def register_view(request):
    """
    User registration view with username, email, password, and confirm password.
    Redirects to dashboard upon successful registration.
    """
    if request.user.is_authenticated:
        if request.user.is_staff:
            return redirect('admin_dashboard')
        return redirect('dashboard')

    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data['password'])
            user.save()
            login(request, user)
            messages.success(request, f"Welcome to Traffic Sign AI, {user.username}! Your account has been created.")
            return redirect('dashboard')
        else:
            messages.error(request, "Registration could not be completed. Please correct the errors below.")
    else:
        form = UserRegisterForm()

    return render(request, 'detector/register.html', {'form': form})


def login_view(request):
    """
    Session-based login view accepting either username or email with password.
    Redirects normal users to User Dashboard and admin/staff users to Admin Dashboard.
    """
    if request.user.is_authenticated:
        if request.user.is_staff:
            return redirect('admin_dashboard')
        return redirect('dashboard')

    next_url = request.GET.get('next', '')

    if request.method == 'POST':
        form = UserLoginForm(request.POST)
        if form.is_valid():
            user = form.cleaned_data['user']
            login(request, user)
            messages.success(request, f"Welcome back, {user.username}!")

            target_next = request.POST.get('next') or request.GET.get('next')
            if user.is_staff:
                if target_next and target_next not in ['dashboard', '/dashboard/']:
                    return redirect(target_next)
                return redirect('admin_dashboard')
            else:
                if target_next and 'admin' not in target_next:
                    return redirect(target_next)
                return redirect('dashboard')
        else:
            messages.error(request, "Invalid username/email or password. Please try again.")
    else:
        form = UserLoginForm()

    return render(request, 'detector/login.html', {'form': form, 'next': next_url})


def logout_view(request):
    """
    Logs out the user and clears session.
    """
    logout(request)
    messages.info(request, "You have been logged out successfully.")
    return redirect('login')


@login_required
def dashboard(request):
    """
    Professional Dashboard for authenticated normal users:
    - Welcome message with username
    - Total detections count
    - Recent detections count & summary (user-scoped)
    - Quick action buttons (Start Detection, View History)
    - Traffic sign categories
    - System explanation
    """
    user_detections = TrafficSignDetection.objects.filter(user=request.user)
    user_total = user_detections.count()
    user_detected_count = user_detections.filter(sign_detected=True).count()
    recent_detections = user_detections[:5]
    recent_count = len(recent_detections)

    # Compute average processing time for this user
    user_avg = user_detections.aggregate(Avg('processing_time'))['processing_time__avg']
    if user_avg is not None:
        avg_processing_time = round(float(user_avg), 3)
    else:
        avg_processing_time = 0.0

    system_total = TrafficSignDetection.objects.count()

    context = {
        'username': request.user.username,
        'user_total': user_total,
        'user_detected_count': user_detected_count,
        'recent_count': recent_count,
        'avg_processing_time': avg_processing_time,
        'recent_detections': recent_detections,
        'fallback_recent': [],
        'system_total': system_total,
        'catalog': SIGN_CATALOG,
    }
    return render(request, 'detector/dashboard.html', context)


@admin_required
def admin_dashboard(request):
    """
    Professional Admin Dashboard for authorized admin/staff users:
    - Summary cards: Total Users, Total Detections, Today's Detections, Total Feedback, Average Processing Time
    - Section A: User Details (all registered users)
    - Section B: Detection History (all users' detections with search)
    - Section C: Feedback (exact heading 'Feedback')
    - Section D: Today's Detections (exact heading 'Today's Detections')
    """
    today = timezone.localdate()

    # Database summary counts
    total_users = User.objects.count()
    total_detections = TrafficSignDetection.objects.count()
    today_detections_qs = TrafficSignDetection.objects.filter(created_at__date=today).select_related('user').order_by('-created_at')
    today_count = today_detections_qs.count()
    total_feedback = Feedback.objects.count()
    avg_proc = TrafficSignDetection.objects.aggregate(Avg('processing_time'))['processing_time__avg']
    avg_processing_time = round(float(avg_proc), 3) if avg_proc is not None else 0.0

    # Section A: User Details
    users_list = User.objects.all().order_by('-date_joined').annotate(detection_count=Count('detections'))

    # Section B: All Users Detection History with optional search & filter
    history_search = request.GET.get('q', '').strip()
    history_cat = request.GET.get('category', '').strip()
    all_detections_qs = TrafficSignDetection.objects.all().select_related('user').order_by('-created_at')

    if history_search:
        all_detections_qs = all_detections_qs.filter(
            Q(sign_name__icontains=history_search) |
            Q(user__username__icontains=history_search) |
            Q(title__icontains=history_search)
        )
    if history_cat and history_cat != 'all':
        all_detections_qs = all_detections_qs.filter(category=history_cat)

    paginator = Paginator(all_detections_qs, 15)
    page_obj = paginator.get_page(request.GET.get('page'))

    # Section C: Feedback from all users and visitors
    feedbacks = Feedback.objects.all().select_related('user').order_by('-created_at')

    context = {
        'total_users': total_users,
        'total_detections': total_detections,
        'today_detections_count': today_count,
        'today_detections': today_detections_qs[:20],
        'total_feedback': total_feedback,
        'avg_processing_time': avg_processing_time,
        'users_list': users_list,
        'all_detections': page_obj,
        'feedbacks': feedbacks,
        'history_search': history_search,
        'history_cat': history_cat,
        'today_date': today,
    }
    return render(request, 'detector/admin_dashboard.html', context)


@login_required
def detect(request):
    """
    Traffic sign upload and detection processing view.
    Measures dynamic processing time and stores results in MySQL.
    Protected: requires login.
    """
    sample_dir = os.path.join(settings.BASE_DIR, 'static', 'images', 'sample_signs')
    sample_files = []
    if os.path.exists(sample_dir):
        sample_files = [
            f for f in os.listdir(sample_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

    if request.method == 'POST':
        sample_choice = request.POST.get('sample_choice')

        if sample_choice:
            sample_path = os.path.join(sample_dir, sample_choice)
            if os.path.exists(sample_path):
                with open(sample_path, 'rb') as f:
                    file_content = f.read()

                detection = TrafficSignDetection(
                    user=request.user,
                    title=f"Sample: {sample_choice.replace('sample_', '').replace('.jpg', '').replace('_', ' ').title()}",
                )
                detection.original_image.save(
                    sample_choice,
                    ContentFile(file_content),
                    save=False
                )
                detection.save()

                try:
                    _process_detection(detection)
                    messages.success(request, f"Detection completed: {detection.sign_name}")
                    return redirect('detection_result', pk=detection.pk)
                except Exception:
                    messages.error(request, "Something went wrong while processing the image. Please try again.")
                    return redirect('detect')

        # File upload form
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                detection = form.save(commit=False)
                detection.user = request.user
                if not detection.title:
                    detection.title = "Road Image Scan"
                detection.save()

                _process_detection(detection)

                if detection.sign_detected:
                    messages.success(request, f"Traffic sign identified: {detection.sign_name} ({detection.confidence}% confidence)")
                else:
                    messages.info(request, "Traffic sign could not be identified confidently.")

                return redirect('detection_result', pk=detection.pk)

            except ValueError as val_err:
                messages.error(request, "Unable to process this image. Please upload another valid image.")
                return redirect('detect')
            except Exception:
                messages.error(request, "Something went wrong while processing the image. Please try again.")
                return redirect('detect')
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f"{error}")
    else:
        form = ImageUploadForm()

    context = {
        'form': form,
        'sample_files': sample_files,
    }
    return render(request, 'detector/detect.html', context)


def _process_detection(detection: TrafficSignDetection):
    """
    Safely process detection using uploaded image bytes:
    1. Reads bytes directly from storage.
    2. Measures dynamic elapsed processing time with time.perf_counter().
    3. Runs computer vision detection.
    4. Saves annotated results, cropped ROI, meaning, driver action, and latency to MySQL.
    """
    # 1. Read uploaded image bytes safely
    image_file = detection.original_image
    image_file.open('rb')
    image_bytes = image_file.read()
    image_file.close()

    # 2. Benchmark processing time dynamically
    start_time = time.perf_counter()

    # 3. Decode safely via NumPy buffer and OpenCV imdecode
    cv_img = load_image_from_bytes(image_bytes)
    if cv_img is None:
        raise ValueError("Failed to decode image from uploaded bytes.")

    # 4. Computer vision pipeline
    results = detect_traffic_signs(cv_img)

    # Stop dynamic timer
    end_time = time.perf_counter()
    elapsed_seconds = round(end_time - start_time, 3)

    # 5. Save annotated image to storage if sign detected
    if results['detected'] and results.get('annotated_image') is not None:
        annotated_bytes = save_image(results['annotated_image'], extension='.jpg')
        if annotated_bytes:
            filename = f"annotated_{os.path.basename(detection.original_image.name)}"
            detection.processed_image.save(filename, ContentFile(annotated_bytes), save=False)

    # 6. Save cropped ROI thumbnail if sign detected
    if results['detected'] and results.get('primary_roi') is not None:
        roi_bytes = save_image(results['primary_roi'], extension='.jpg')
        if roi_bytes:
            crop_name = f"roi_{os.path.basename(detection.original_image.name)}"
            detection.cropped_sign.save(crop_name, ContentFile(roi_bytes), save=False)

    # 7. Persist detection outcomes in MySQL
    detection.sign_detected = results['detected']
    detection.sign_name = results['primary_sign']
    detection.category = results['category']
    detection.confidence = results['confidence']
    detection.meaning = results.get('meaning', '')
    detection.driver_action = results.get('driver_action', '')
    detection.detected_shape = results['shape']
    detection.dominant_color = results['dominant_color']
    detection.bounding_box = results['bbox_str']
    detection.processing_time = elapsed_seconds
    detection.description = results['description']
    detection.save()


@login_required
def result(request, pk):
    """
    Detection Result view displaying:
    - Uploaded Traffic Sign Image
    - Detected Traffic Sign / Symbol
    - Sign Name
    - Category & Category Explanation
    - Specific Meaning
    - Driver Action
    - Confidence Score
    - Actual Dynamic Processing Time
    - Detection Date and Time
    """
    detection = get_object_or_404(TrafficSignDetection, pk=pk)

    # Security check: normal users cannot view another user's detection record
    if detection.user and detection.user != request.user and not request.user.is_staff:
        messages.error(request, "You do not have permission to view this detection result.")
        return redirect('history')

    category_exp = CATEGORY_EXPLANATIONS.get(
        detection.category,
        "This sign provides regulatory or safety guidance for road users."
    )
    category_disp = CATEGORY_DISPLAY_NAMES.get(
        detection.category,
        detection.get_category_display()
    )

    context = {
        'detection': detection,
        'category_display': category_disp,
        'category_explanation': category_exp,
        'catalog_info': SIGN_CATALOG.get(detection.sign_name, None),
    }
    return render(request, 'detector/result.html', context)


@login_required
def history(request):
    """
    User Detection History view:
    Displays ONLY the logged-in user's own detection records.
    """
    search_query = request.GET.get('q', '').strip()
    category_filter = request.GET.get('category', '').strip()

    # STRICTLY logged-in user's detections
    records = TrafficSignDetection.objects.filter(user=request.user).order_by('-created_at')

    if search_query:
        records = records.filter(
            Q(sign_name__icontains=search_query) | Q(title__icontains=search_query)
        )

    if category_filter and category_filter != 'all':
        records = records.filter(category=category_filter)

    paginator = Paginator(records, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    total_count = records.count()
    detected_count = records.filter(sign_detected=True).count()

    context = {
        'page_obj': page_obj,
        'search_query': search_query,
        'category_filter': category_filter,
        'total_count': total_count,
        'detected_count': detected_count,
    }
    return render(request, 'detector/history.html', context)


@login_required
def delete_detection(request, pk):
    """Delete a single detection record."""
    if request.method == 'POST':
        if request.user.is_staff:
            detection = get_object_or_404(TrafficSignDetection, pk=pk)
        else:
            detection = get_object_or_404(TrafficSignDetection, pk=pk, user=request.user)

        sign_name = detection.sign_name

        try:
            if detection.original_image and os.path.exists(detection.original_image.path):
                os.remove(detection.original_image.path)
            if detection.processed_image and os.path.exists(detection.processed_image.path):
                os.remove(detection.processed_image.path)
            if detection.cropped_sign and os.path.exists(detection.cropped_sign.path):
                os.remove(detection.cropped_sign.path)
        except Exception:
            pass

        detection.delete()
        messages.success(request, f"Detection record for '{sign_name}' was deleted.")

    if request.user.is_staff and 'admin' in request.META.get('HTTP_REFERER', ''):
        return redirect('admin_dashboard')
    return redirect('history')


@login_required
def clear_history(request):
    """Clear detection history for current user."""
    if request.method == 'POST':
        count = TrafficSignDetection.objects.filter(user=request.user).count()
        TrafficSignDetection.objects.filter(user=request.user).delete()
        messages.info(request, f"Cleared {count} scan record(s) from your history.")
    return redirect('history')


def feedback_view(request):
    """
    User Feedback view:
    Supports submissions by both logged-in users and logged-out visitors.
    Prefills name and email from account for logged-in users.
    Saves feedback persistently to MySQL.
    """
    if request.method == 'POST':
        form = FeedbackForm(request.POST)
        if form.is_valid():
            fb = form.save(commit=False)
            if request.user.is_authenticated:
                fb.user = request.user
            fb.save()
            messages.success(request, "Thank you! Your feedback has been submitted successfully.")
            return redirect('feedback')
        else:
            messages.error(request, "Please correct the errors in the feedback form below.")
    else:
        initial = {}
        if request.user.is_authenticated:
            initial['name'] = request.user.get_full_name() or request.user.username
            initial['email'] = request.user.email
            initial['rating'] = 5
        form = FeedbackForm(initial=initial)

    return render(request, 'detector/feedback.html', {'form': form})


def contact(request):
    """Backwards compatibility redirect to Feedback page."""
    return redirect('feedback')
