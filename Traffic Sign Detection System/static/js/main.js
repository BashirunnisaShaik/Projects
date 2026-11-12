/**
 * Main JavaScript for Traffic Sign Detection System
 * Handles UI interactions, mobile menu, alerts, and tab switching.
 */

document.addEventListener('DOMContentLoaded', function () {
  // 1. Mobile Menu Toggle
  const navToggle = document.getElementById('navToggle');
  const navMenu = document.getElementById('navMenu');

  if (navToggle && navMenu) {
    navToggle.addEventListener('click', function () {
      navMenu.classList.toggle('show');
    });
  }

  // 2. Auto-dismiss Alert Messages after 5 seconds
  const alerts = document.querySelectorAll('.alert');
  alerts.forEach(function (alert) {
    const closeBtn = alert.querySelector('.alert-close');
    if (closeBtn) {
      closeBtn.addEventListener('click', function () {
        alert.style.opacity = '0';
        setTimeout(() => alert.remove(), 250);
      });
    }

    setTimeout(function () {
      alert.style.transition = 'opacity 0.4s ease';
      alert.style.opacity = '0';
      setTimeout(() => alert.remove(), 400);
    }, 6000);
  });

  // 3. Image Comparison Tab Switcher (Result Page)
  const tabBtns = document.querySelectorAll('.image-tab-btn');
  const imageViews = document.querySelectorAll('.image-view-target');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', function () {
      const target = this.getAttribute('data-target');

      // Update button active state
      tabBtns.forEach(b => b.classList.remove('active'));
      this.classList.add('active');

      // Show requested image
      imageViews.forEach(view => {
        if (view.id === target) {
          view.style.display = 'block';
        } else {
          view.style.display = 'none';
        }
      });
    });
  });

  // 4. Print Report Button
  const printBtn = document.getElementById('printReportBtn');
  if (printBtn) {
    printBtn.addEventListener('click', function () {
      window.print();
    });
  }
});
