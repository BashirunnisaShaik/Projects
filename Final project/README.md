# 🚦 Traffic Sign Detection System

A complete, full-stack, computer-vision-powered web application built with **Django**, **Python OpenCV**, **MySQL**, and modern **HTML/CSS/JavaScript**.

The system enables registered drivers and operators to upload road and traffic images, preprocesses them to remove noise and standardize contrast, applies **HSV color segmentation**, **Canny edge detection**, and **contour shape analysis** to detect and recognize common traffic signs (such as **Stop**, **No Entry**, **Speed Limit**, **Turn Left**, **Turn Right**, **Yield**, **School Zone**, **Traffic Signal**, **Hospital**, etc.), displays the legal **Sign Meaning** and required **Driver Action**, measures **dynamic processing time**, visualizes annotated results with bounding boxes and confidence scores, and stores all scan logs persistently in **MySQL**.

---

## 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [Key Modules & Pages](#-key-modules--pages)
3. [Technology Stack](#-technology-stack)
4. [Computer Vision Methodology](#-computer-vision-methodology)
5. [Database Architecture (MySQL)](#-database-architecture-mysql)
6. [Project Folder Structure](#-project-folder-structure)
7. [Installation & Setup Guide](#-installation--setup-guide)
8. [Automated Test Suite](#-automated-test-suite)
9. [Student Viva / Demonstration Guide](#-student-viva--demonstration-guide)

---

## 🌟 Project Overview

Automated traffic sign recognition is a critical pillar of Intelligent Transportation Systems (ITS) and Autonomous Driving Technology. This project provides an accessible, fully modular web-based demonstration:
- **User Authentication**: Secure session-based registration, login (username or email), logout, and protected route access.
- **Interactive Dashboard**: Real-time driver dashboard tracking personal and system detections, signs identified, recent scans, and average processing latency.
- **Instant Computer Vision Processing**: Analyzes uploaded road photos with safe byte buffer decoding (`numpy.frombuffer` + `cv2.imdecode`) and PIL fallback, eliminating file-locking and decoding failures.
- **Accurate Localization**: Highlights identified traffic signs with colored bounding boxes and corner brackets.
- **Legal Meaning & Driver Action**: Clearly educates drivers on what each detected sign means and what action must be taken immediately.
- **Dynamic Timer**: Measures real-time computer vision execution time (`time.perf_counter()`), never hardcoded.
- **Traffic Sign Guide**: Educational reference catalog categorized by Regulatory, Warning, Mandatory, and Informative signs.
- **Mobile Responsive Design**: Clean road-themed UI with hamburger mobile navigation, touch-friendly buttons, and responsive history cards.
- **Relational Auditing**: Every inspection is logged in MySQL for historical audit, search, and filtering.

---

## 🧩 Key Modules & Pages

| Module / Page | URL | Description | Access |
| :--- | :--- | :--- | :--- |
| **Home Page** | `/` | Traffic-themed landing page with live detection stats, feature highlights, and quick CTAs. | Public |
| **About Project** | `/about/` | Detailed explanation of traffic signs, detection concepts, limitations of traditional methods, proposed pipeline, and advantages. | Public |
| **Traffic Sign Guide** | `/signs/` | Comprehensive catalog organized by category (Regulatory, Warning, Mandatory, Informative) with meanings and driver actions. | Public |
| **User Registration** | `/register/` | Account registration with validation and automatic login. | Public |
| **User Login** | `/login/` | Session authentication supporting username or email. | Public |
| **Driver Dashboard** | `/dashboard/` | Statistics cards (Total Detections, Recognized Signs, Recent Scans, Average Latency), how the engine works, and recent inspection history. | Authenticated |
| **Detection Studio** | `/detect/` | Drag-and-drop image upload (JPG/JPEG/PNG), client-side preview, remove/change option, and 1-click test gallery. | Authenticated |
| **Detection Result** | `/result/<id>/` | Visual inspection (Original vs Annotated), ROI crop, detected sign, meaning, driver action, confidence score, dynamic processing time, and print report. | Authenticated |
| **Detection History** | `/history/` | User-scoped MySQL logs with responsive table (desktop) and mobile cards (smartphones), search, and category filters. | Authenticated |
| **Contact & Feedback** | `/contact/` | Contact message form saved directly to MySQL. | Public |

---

## 🛠 Technology Stack

- **Backend Framework**: Python 3.13 / Django 5.2
- **Authentication**: Django Authentication (Session-based, user-scoped history)
- **Computer Vision**: OpenCV (`opencv-python`), NumPy, Pillow
- **Database**: MySQL 8.0 (connected via `PyMySQL`)
- **Frontend**: HTML5, CSS3 (Modern Flexbox & CSS Grid, Custom Variables), Vanilla JavaScript (ES6+)
- **Testing**: Django Test Framework (16 automated test suites)

---

## 🔬 Computer Vision Methodology

```
Uploaded Image File (Bytes)
      │
      ▼
[1. Safe Buffer Decoding via NumPy & cv2.imdecode()]
      │
      ▼
[2. Resolution Normalization & Aspect Ratio Preservation]
      │
      ▼
[3. Color Space Conversion (BGR to Grayscale & HSV)]
      │
      ▼
[4. Gaussian Blur Noise Reduction (k=5)]
      │
      ▼
[5. CLAHE Adaptive Contrast Equalization]
      │
      ▼
[6. HSV Color Segmentation] ──────────┐
  • Red Masks (Hue: 0-12 & 160-180)    │
  • Blue Mask (Hue: 95-135)            │
  • Yellow Mask (Hue: 15-38)           ▼
      │                       [7. Canny Edge Detection]
      ▼                                │
[8. Morphological Closing & Opening]   │
      │                                │
      ▼                                ▼
[9. Contour Extraction & Filtering (Area, Aspect Ratio)]
      │
      ▼
[10. Geometric Shape & Polygonal Approximation]
  • Octagon (8 vertices / circularity)  ──> Stop Sign
  • Inverted Triangle (3 vertices)      ──> Yield / Give Way
  • Equilateral Triangle (3 vertices)   ──> Hazard, School Zone, Traffic Signal
  • Circle (Circularity > 0.70)         ──> Speed Limit, No Entry, Turn Arrows
  • Rectangle                           ──> Hospital, Parking Area
      │
      ▼
[11. Internal ROI Feature Analysis (Arrows, Bars, Text)]
      │
      ▼
[12. Annotated Bounding Box, Meaning, Driver Action & Dynamic Latency]
```

### Shape & Circularity Formulas:
- **Circularity Metric**:
  $$\text{Circularity} = \frac{4 \pi \times \text{Area}}{\text{Perimeter}^2}$$
  - A circle has circularity close to $1.0$.
  - An octagon has circularity around $0.78 - 0.88$.
- **Polygon Approximation**:
  $$\epsilon = 0.035 \times \text{Perimeter}$$
  $$\text{Polygon} = \text{cv2.approxPolyDP}(\text{contour}, \epsilon, \text{True})$$

---

## 🗄 Database Architecture (MySQL)

### Database: `traffic_sign_db`

#### Table: `detector_trafficsigndetection`
| Column | Type | Purpose |
| :--- | :--- | :--- |
| `id` | BIGINT (Auto Inc) | Primary Key |
| `user_id` | INT (FK) | Authenticated user who performed the scan |
| `title` | VARCHAR(150) | Scan label / location tag |
| `original_image` | VARCHAR(100) | File path of uploaded road image |
| `processed_image`| VARCHAR(100) | File path of marked/annotated image |
| `cropped_sign` | VARCHAR(100) | File path of cropped sign ROI |
| `sign_detected` | TINYINT(1) | Boolean indicator if sign was found |
| `sign_name` | VARCHAR(100) | Recognized sign name (e.g., "Stop Sign") |
| `category` | VARCHAR(30) | Category (regulatory, mandatory, warning, information, none) |
| `confidence` | DOUBLE | Recognition confidence percentage |
| `detected_shape` | VARCHAR(50) | Classified geometry (Circle, Octagon, Triangle, Rectangle) |
| `dominant_color` | VARCHAR(50) | Dominant color (Red, Blue, Yellow) |
| `bounding_box` | VARCHAR(100) | Coordinates string (`x=..., y=..., w=..., h=...`) |
| `processing_time`| DOUBLE | Actual dynamic latency in seconds (`time.perf_counter()`) |
| `meaning` | TEXT | Legal meaning of the detected traffic sign |
| `driver_action` | TEXT | Specific action required of the driver |
| `description` | TEXT | Comprehensive safety note / description |
| `created_at` | DATETIME | Timestamp of scan |

---

## 📁 Project Folder Structure

```
Final project/
│
├── manage.py                          # Django management script
├── requirements.txt                   # Python packages
├── README.md                          # Comprehensive documentation
│
├── traffic_sign_project/              # Project Configuration
│   ├── __init__.py                    # PyMySQL initialization
│   ├── settings.py                    # MySQL DB, Auth redirects, media settings
│   ├── urls.py                        # Root URL routing
│   ├── asgi.py
│   └── wsgi.py
│
├── detector/                          # Core Application
│   ├── admin.py                       # Django Admin configuration with fieldsets
│   ├── apps.py                        # App configuration
│   ├── models.py                      # MySQL models (TrafficSignDetection, ContactMessage)
│   ├── forms.py                       # UserRegisterForm, UserLoginForm, ImageUploadForm, ContactForm
│   ├── views.py                       # Auth, Dashboard, Detection, Result, Guide, History
│   ├── urls.py                        # Application URL patterns
│   ├── tests.py                       # 16 Automated test suites
│   │
│   └── core/                          # Computer Vision Engine
│       ├── __init__.py
│       ├── preprocessing.py           # Safe byte decoding, CLAHE, noise reduction
│       ├── detection.py               # SIGN_CATALOG, HSV segmentation, contour geometry
│       └── utils.py                   # Bounding box drawing, cropping, ROI extraction
│
├── static/                            # Static Assets
│   ├── css/
│   │   ├── style.css                  # Modern responsive design & mobile rules
│   │   └── animations.css             # Spinner, fade-in, and keyframe animations
│   ├── js/
│   │   ├── main.js                    # Mobile hamburger toggle, alert auto-dismiss, tabs
│   │   ├── upload.js                  # Preview, clear/remove, loading spinner, anti-duplicate click
│   │   └── history.js                 # Filter submit, delete confirmation dialog
│   └── images/
│       └── sample_signs/              # 7 pre-generated test road images
│
├── templates/                         # HTML Templates
│   ├── base.html                      # Layout, session-aware navbar, footer
│   └── detector/
│       ├── home.html                  # Landing page
│       ├── about.html                 # About project & computer vision theory
│       ├── guide.html                 # Traffic sign guide (meaning & driver actions)
│       ├── register.html              # User registration
│       ├── login.html                 # User login
│       ├── dashboard.html             # Driver & system stats dashboard
│       ├── detect.html                # Detection studio & 1-click test gallery
│       ├── result.html                # Inspection results, meaning & driver action
│       ├── history.html               # Filterable history (desktop table / mobile cards)
│       └── contact.html               # Project inquiry form
│
├── media/                             # User Media Storage
│   ├── uploads/
│   ├── processed/
│   └── crops/
│
└── sample_data/
    └── generate_samples.py            # Synthetic test road image generator
```

---

## 🚀 Installation & Setup Guide

### 1. Prerequisites
- Python 3.10+
- MySQL Server 8.0 running on port 3306

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure MySQL Database
Ensure your MySQL server is running and create the database:
```sql
CREATE DATABASE IF NOT EXISTS traffic_sign_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

> **Default connection credentials (`traffic_sign_project/settings.py`):**
> - **Host:** `127.0.0.1` | **Port:** `3306`
> - **User:** `root` | **Password:** `root`
> - **Database:** `traffic_sign_db`

### 4. Run Migrations
```bash
python manage.py makemigrations detector
python manage.py migrate
```

### 5. Generate Test Road Signs (Optional)
```bash
python sample_data/generate_samples.py
```

### 6. Create Superuser (For Django Admin)
```bash
python manage.py createsuperuser
```

### 7. Run Development Server
```bash
python manage.py runserver
```
Visit **http://127.0.0.1:8000/** in your web browser.

---

## 🧪 Automated Test Suite

Run the full automated test suite verifying all 16 test cases:
```bash
python manage.py test
```

**Test Coverage Highlights:**
- Public pages return HTTP 200 (Home, About, Guide, Login, Register, Contact)
- Protected pages redirect unauthenticated users to Login
- User registration and login with username or email
- Dashboard rendering with user and system statistics
- 1-click sample detection and safe byte decoding
- Upload flow for JPG, PNG (RGBA 4-channel), and CMYK JPEGs
- Rejection and user-friendly error alerts for corrupted or empty files
- Dynamic processing time calculation (`time.perf_counter()`)
- Persistent MySQL database storage of Sign Name, Meaning, Driver Action, and Latency
- User-scoped detection history and filtering

---

## 🎓 Student Viva / Demonstration Guide

When demonstrating the project during a review or viva examination:
1. **Visitor Landing (`/`)**: Show the responsive navbar (Home, About, Traffic Signs, Login, Register) and the live scan counter.
2. **Register & Login (`/register/` & `/login/`)**: Register a new user or log in with username/email to demonstrate Django session-based authentication.
3. **Dashboard (`/dashboard/`)**: Highlight user-specific statistics (Total Detections, Signs Recognized, Recent Scans, Average Processing Latency) and the explanation cards.
4. **Traffic Sign Guide (`/signs/`)**: Walk through standardized signs organized into Regulatory, Warning, Mandatory, and Informative cards detailing their legal meanings and required driver actions.
5. **Detection Studio (`/detect/`)**:
   - Demonstrate drag-and-drop file upload with live image preview (dimensions, size, filename).
   - Test the "Remove / Change" button.
   - Use the **1-Click Test Samples** (e.g. Stop Sign, Speed Limit 60, Empty Road) and show the animated processing loader.
6. **Result Page (`/result/<id>/`)**:
   - Point out the **Sign Meaning** and **Driver Action** boxes prominently displayed above the technical metrics.
   - Show the dynamic processing time (e.g., `0.38s`).
   - Toggle between Detected/Annotated View and Original Upload.
   - Click **Print Report** to demonstrate report generation.
7. **Detection History (`/history/`)**:
   - Demonstrate that each scan is saved in MySQL.
   - Test the search bar and category filter.
   - Resize browser to mobile width to demonstrate how the desktop table seamlessly turns into clean, responsive mobile cards without horizontal scrolling!
8. **Django Admin (`/admin/`)**:
   - Log into the administrative dashboard to show organized fieldsets for detection info, meanings, driver actions, technical metrics, and contact messages.

---

## 📄 License & Credits
Developed as an Engineering Final Project demonstration. Built with Python, Django, OpenCV, and MySQL.
