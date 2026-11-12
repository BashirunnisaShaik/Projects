from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
import os
import io
import cv2
import numpy as np
from PIL import Image

from detector.models import TrafficSignDetection, Feedback
from detector.core.detection import detect_traffic_signs, SIGN_CATALOG, CATEGORY_EXPLANATIONS
from detector.core.preprocessing import load_image_from_bytes, load_image


class TrafficSignSystemTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            username='testdriver',
            email='testdriver@example.com',
            password='Password123!'
        )
        self.admin_user = User.objects.create_superuser(
            username='adminuser',
            email='admin@example.com',
            password='AdminPassword123!'
        )

    def test_public_pages_status_codes(self):
        """Verify public pages return HTTP 200 for unauthenticated visitors."""
        public_urls = ['home', 'about', 'traffic_signs_guide', 'login', 'register', 'feedback']
        for name in public_urls:
            response = self.client.get(reverse(name))
            self.assertEqual(response.status_code, 200, f"Public page '{name}' failed with status {response.status_code}")

    def test_protected_pages_redirect_unauthenticated(self):
        """Verify protected pages redirect unauthenticated users to login."""
        protected_urls = ['dashboard', 'detect', 'history']
        for name in protected_urls:
            response = self.client.get(reverse(name))
            self.assertEqual(response.status_code, 302, f"Protected page '{name}' did not redirect")
            self.assertIn(reverse('login'), response.url)

    def test_user_registration_flow(self):
        """Test full user registration flow with automatic login and redirect to dashboard."""
        reg_data = {
            'username': 'newuser',
            'email': 'newuser@example.com',
            'password': 'StrongPassword123!',
            'confirm_password': 'StrongPassword123!',
        }
        response = self.client.post(reverse('register'), reg_data)
        self.assertRedirects(response, reverse('dashboard'))
        self.assertTrue(User.objects.filter(username='newuser').exists())

    def test_user_login_normal_redirects_to_user_dashboard(self):
        """Verify normal user logging in is redirected to User Dashboard (/dashboard/)."""
        response = self.client.post(reverse('login'), {
            'username_or_email': 'testdriver',
            'password': 'Password123!',
        })
        self.assertRedirects(response, reverse('dashboard'))

    def test_admin_login_redirects_to_admin_dashboard(self):
        """Verify staff/admin user logging in is redirected to Admin Dashboard (/admin-dashboard/)."""
        response = self.client.post(reverse('login'), {
            'username_or_email': 'adminuser',
            'password': 'AdminPassword123!',
        })
        self.assertRedirects(response, reverse('admin_dashboard'))

    def test_admin_dashboard_access_control(self):
        """Verify Admin Dashboard is strictly protected: normal users get 403, admin gets 200."""
        # 1. Unauthenticated -> redirect to login
        resp_anon = self.client.get(reverse('admin_dashboard'))
        self.assertEqual(resp_anon.status_code, 302)

        # 2. Normal user -> redirected to dashboard with error message
        self.client.force_login(self.user)
        resp_user = self.client.get(reverse('admin_dashboard'))
        self.assertEqual(resp_user.status_code, 302)
        self.assertRedirects(resp_user, reverse('dashboard'))

        # 3. Admin user -> 200 OK
        self.client.force_login(self.admin_user)
        resp_admin = self.client.get(reverse('admin_dashboard'))
        self.assertEqual(resp_admin.status_code, 200)
        self.assertContains(resp_admin, 'Admin Dashboard')
        self.assertContains(resp_admin, 'Feedback')
        self.assertContains(resp_admin, "Today's Detections")

    def test_user_logout(self):
        """Test logout clears session and redirects to login."""
        self.client.force_login(self.user)
        response = self.client.get(reverse('logout'))
        self.assertRedirects(response, reverse('login'))

    def test_dashboard_authenticated(self):
        """Test dashboard renders properly with stats for logged-in user."""
        self.client.force_login(self.user)
        response = self.client.get(reverse('dashboard'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'testdriver')
        self.assertContains(response, 'Avg Processing Time')

    def test_sample_quick_detection(self):
        """Verify 1-click sample detection creates record with meaning, driver_action, and processing_time."""
        self.client.force_login(self.user)
        response = self.client.post(reverse('detect'), {'sample_choice': 'sample_stop_sign.jpg'})
        self.assertEqual(response.status_code, 302)
        self.assertEqual(TrafficSignDetection.objects.count(), 1)
        det = TrafficSignDetection.objects.first()
        self.assertEqual(det.user, self.user)
        self.assertTrue(det.sign_detected)
        self.assertEqual(det.sign_name, 'Stop Sign')
        self.assertGreater(det.confidence, 80.0)
        self.assertIsNotNone(det.meaning)
        self.assertIsNotNone(det.driver_action)
        self.assertGreater(det.processing_time, 0.0)

        # Test viewing the result page
        res_page = self.client.get(reverse('detection_result', args=[det.id]))
        self.assertEqual(res_page.status_code, 200)
        self.assertContains(res_page, 'Stop Sign')
        self.assertContains(res_page, 'Sign Meaning')
        self.assertContains(res_page, 'Driver Action')
        self.assertContains(res_page, 'Category')

    def test_empty_road_detection(self):
        """Verify empty road image is correctly labeled as No Sign Detected with graceful undetected display."""
        self.client.force_login(self.user)
        response = self.client.post(reverse('detect'), {'sample_choice': 'sample_empty_road.jpg'})
        self.assertEqual(response.status_code, 302)
        det = TrafficSignDetection.objects.first()
        self.assertFalse(det.sign_detected)
        self.assertEqual(det.sign_name, 'No Sign Detected')

        res_page = self.client.get(reverse('detection_result', args=[det.id]))
        self.assertEqual(res_page.status_code, 200)
        self.assertContains(res_page, 'Traffic sign could not be identified confidently.')

    def test_upload_jpg_pipeline_flow(self):
        """
        Verify end-to-end flow for JPG upload:
        Upload JPG -> Save -> Decode safely via OpenCV/numpy -> Preprocess -> Detect -> Display Result -> Save MySQL
        """
        self.client.force_login(self.user)
        # Create a synthetic road image with red Stop-like circle
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        img[:] = (200, 180, 150)  # background
        cv2.circle(img, (150, 150), 70, (20, 20, 220), -1)  # Red circle
        cv2.putText(img, "STOP", (100, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        _, enc = cv2.imencode('.jpg', img)
        jpg_file = SimpleUploadedFile('test_road_sign.jpg', enc.tobytes(), content_type='image/jpeg')

        response = self.client.post(reverse('detect'), {
            'title': 'Test Highway Scan',
            'original_image': jpg_file,
        }, follow=True)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(TrafficSignDetection.objects.count(), 1)
        record = TrafficSignDetection.objects.first()
        self.assertTrue(record.sign_detected)
        self.assertEqual(record.sign_name, 'Stop Sign')
        self.assertIsNotNone(record.meaning)
        self.assertIsNotNone(record.driver_action)
        self.assertGreater(record.processing_time, 0.0)
        self.assertIsNotNone(record.processed_image)
        self.assertTrue(os.path.exists(record.processed_image.path))

    def test_upload_png_rgba_pipeline_flow(self):
        """Verify PNG image with 4-channel alpha transparency decodes and processes correctly."""
        self.client.force_login(self.user)
        pil_img = Image.new('RGBA', (200, 200), color=(0, 0, 255, 255))
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')

        png_file = SimpleUploadedFile('test_transparent.png', buf.getvalue(), content_type='image/png')
        response = self.client.post(reverse('detect'), {
            'title': 'PNG Alpha Test',
            'original_image': png_file,
        }, follow=True)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(TrafficSignDetection.objects.count(), 1)
        record = TrafficSignDetection.objects.first()
        self.assertTrue(os.path.exists(record.original_image.path))

    def test_upload_corrupted_file_handling(self):
        """Verify invalid/corrupted bytes are rejected gracefully with user-friendly alert."""
        self.client.force_login(self.user)
        corrupted_bytes = b"NOT_A_VALID_IMAGE_HEADER_RANDOM_TEXT_DATA"
        bad_file = SimpleUploadedFile('corrupted.jpg', corrupted_bytes, content_type='image/jpeg')

        response = self.client.post(reverse('detect'), {
            'title': 'Corrupted File Test',
            'original_image': bad_file,
        }, follow=True)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(TrafficSignDetection.objects.count(), 0)
        self.assertContains(response, 'Upload a valid image')

    def test_history_user_scope_and_isolation(self):
        """Verify detection history is strictly isolated per user."""
        other_user = User.objects.create_user(
            username='otherdriver',
            email='other@example.com',
            password='Password123!'
        )

        det_user1 = TrafficSignDetection.objects.create(
            user=self.user,
            title='User 1 Scan',
            sign_name='Stop Sign',
            category='regulatory',
            sign_detected=True,
            confidence=95.0,
            meaning='Driver must stop.',
            driver_action='Stop vehicle.',
            processing_time=0.35,
            original_image='uploads/test1.jpg'
        )

        TrafficSignDetection.objects.create(
            user=other_user,
            title='Other User Scan',
            sign_name='Speed Limit Sign',
            category='prohibitory',
            sign_detected=True,
            confidence=90.0,
            meaning='Max speed 60.',
            driver_action='Do not exceed 60 km/h.',
            processing_time=0.40,
            original_image='uploads/test2.jpg'
        )

        # Logged in as other_user: should see ONLY their own scan
        self.client.force_login(other_user)
        response = self.client.get(reverse('history'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Speed Limit Sign')
        self.assertNotContains(response, 'User 1 Scan')

        # Other user trying to access User 1's detection result is blocked
        res_blocked = self.client.get(reverse('detection_result', args=[det_user1.id]))
        self.assertEqual(res_blocked.status_code, 302)
        self.assertRedirects(res_blocked, reverse('history'))

        # Admin can view User 1's detection result
        self.client.force_login(self.admin_user)
        res_admin = self.client.get(reverse('detection_result', args=[det_user1.id]))
        self.assertEqual(res_admin.status_code, 200)
        self.assertContains(res_admin, 'Stop Sign')

    def test_feedback_submission_visitor_and_user(self):
        """Verify feedback submission by visitor and authenticated user."""
        # 1. Visitor feedback
        resp_visitor = self.client.post(reverse('feedback'), {
            'name': 'Visitor John',
            'email': 'visitor@example.com',
            'rating': 5,
            'message': 'Excellent detection platform!',
        })
        self.assertRedirects(resp_visitor, reverse('feedback'))
        self.assertEqual(Feedback.objects.count(), 1)
        fb1 = Feedback.objects.first()
        self.assertEqual(fb1.name, 'Visitor John')
        self.assertIsNone(fb1.user)
        self.assertEqual(fb1.rating, 5)

        # 2. Authenticated user feedback
        self.client.force_login(self.user)
        resp_user = self.client.post(reverse('feedback'), {
            'name': 'Test Driver',
            'email': 'testdriver@example.com',
            'rating': 4,
            'message': 'Very fast response time and accurate.',
        })
        self.assertRedirects(resp_user, reverse('feedback'))
        self.assertEqual(Feedback.objects.count(), 2)
        fb2 = Feedback.objects.filter(user=self.user).first()
        self.assertIsNotNone(fb2)
        self.assertEqual(fb2.rating, 4)

    def test_traffic_signs_guide_page(self):
        """Verify guide page shows all sign categories with meaning and actions."""
        response = self.client.get(reverse('traffic_signs_guide'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Warning Signs')
        self.assertContains(response, 'Regulatory Signs')
        self.assertContains(response, 'Mandatory Signs')
        self.assertContains(response, 'Prohibited / Prohibitory Signs')
        self.assertContains(response, 'Information Signs')
