from django import forms
from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from .models import TrafficSignDetection, ContactMessage, Feedback


class UserRegisterForm(forms.ModelForm):
    """
    User registration form with username, email, password, and confirm password validations.
    """
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Create a secure password',
            'autocomplete': 'new-password',
        }),
        min_length=6,
        help_text="Password must be at least 6 characters long."
    )
    confirm_password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Re-enter your password to confirm',
            'autocomplete': 'new-password',
        })
    )

    class Meta:
        model = User
        fields = ['username', 'email']
        widgets = {
            'username': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Choose a username (e.g. alex_driver)',
                'autocomplete': 'username',
            }),
            'email': forms.EmailInput(attrs={
                'class': 'form-control',
                'placeholder': 'your.email@example.com',
                'autocomplete': 'email',
            }),
        }

    def clean_username(self):
        username = self.cleaned_data.get('username', '').strip()
        if User.objects.filter(username__iexact=username).exists():
            raise forms.ValidationError("This username is already taken. Please choose another one.")
        return username

    def clean_email(self):
        email = self.cleaned_data.get('email', '').strip().lower()
        if not email:
            raise forms.ValidationError("Email address is required.")
        if User.objects.filter(email__iexact=email).exists():
            raise forms.ValidationError("An account with this email address already exists.")
        return email

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get('password')
        confirm_password = cleaned_data.get('confirm_password')

        if password and confirm_password and password != confirm_password:
            self.add_error('confirm_password', "Passwords do not match. Please enter identical passwords.")

        return cleaned_data


class UserLoginForm(forms.Form):
    """
    User login form supporting either username or email with password.
    """
    username_or_email = forms.CharField(
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your username or email',
            'autocomplete': 'username',
        }),
        label="Username or Email"
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your password',
            'autocomplete': 'current-password',
        }),
        label="Password"
    )

    def clean(self):
        cleaned_data = super().clean()
        identifier = cleaned_data.get('username_or_email', '').strip()
        password = cleaned_data.get('password')

        if identifier and password:
            user = None
            if '@' in identifier:
                user_obj = User.objects.filter(email__iexact=identifier).first()
                if user_obj:
                    user = authenticate(username=user_obj.username, password=password)
            else:
                user = authenticate(username=identifier, password=password)

            if not user:
                raise forms.ValidationError("Invalid credentials. Please verify your username/email and password.")

            cleaned_data['user'] = user

        return cleaned_data


class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = TrafficSignDetection
        fields = ['title', 'original_image']
        widgets = {
            'title': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter scan title (optional e.g., Highway Checkpoint A)',
            }),
            'original_image': forms.FileInput(attrs={
                'class': 'file-input-hidden',
                'id': 'image-upload-input',
                'accept': 'image/jpeg,image/png,image/jpg',
            }),
        }

    def clean_original_image(self):
        image = self.cleaned_data.get('original_image')
        if not image:
            raise forms.ValidationError("Please upload an image file.")

        # Validate file extension
        valid_extensions = ['.jpg', '.jpeg', '.png']
        extension = '.' + image.name.split('.')[-1].lower() if '.' in image.name else ''
        if extension not in valid_extensions:
            raise forms.ValidationError(f"Unsupported format '{extension}'. Please upload a valid JPG, JPEG or PNG image.")

        # Validate file size (non-empty and max 10MB)
        if image.size == 0:
            raise forms.ValidationError("The uploaded file is empty (0 bytes). Please upload a valid image.")

        if image.size > 10 * 1024 * 1024:
            raise forms.ValidationError("Image file size should not exceed 10 MB.")

        # Ensure file pointer is at the beginning for Django storage saving
        if hasattr(image, 'seek'):
            image.seek(0)

        return image


class FeedbackForm(forms.ModelForm):
    """
    Feedback submission form for users and visitors with 1-5 star rating.
    """
    RATING_CHOICES = [(i, f"{i} Star{'s' if i > 1 else ''}") for i in range(1, 6)]

    rating = forms.ChoiceField(
        choices=RATING_CHOICES,
        initial=5,
        widget=forms.Select(attrs={
            'class': 'form-control',
            'id': 'rating-select',
        }),
        help_text="Rate your experience from 1 to 5 stars."
    )

    class Meta:
        model = Feedback
        fields = ['name', 'email', 'rating', 'message']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Your full name',
                'required': True,
            }),
            'email': forms.EmailInput(attrs={
                'class': 'form-control',
                'placeholder': 'your.email@example.com (optional)',
            }),
            'message': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Write your feedback, experience, or suggestions here...',
                'required': True,
            }),
        }

    def clean_name(self):
        name = self.cleaned_data.get('name', '').strip()
        if not name:
            raise forms.ValidationError("Please provide your name.")
        return name

    def clean_message(self):
        message = self.cleaned_data.get('message', '').strip()
        if not message:
            raise forms.ValidationError("Please enter your feedback message.")
        return message


class ContactForm(forms.ModelForm):
    class Meta:
        model = ContactMessage
        fields = ['name', 'email', 'subject', 'message']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Your full name',
                'required': True,
            }),
            'email': forms.EmailInput(attrs={
                'class': 'form-control',
                'placeholder': 'your.email@example.com',
                'required': True,
            }),
            'subject': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Subject of your query or feedback',
                'required': True,
            }),
            'message': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 5,
                'placeholder': 'Write your message or inquiry here...',
                'required': True,
            }),
        }

