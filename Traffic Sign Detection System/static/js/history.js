/**
 * History Module JavaScript
 * Handles search submission, category filter triggers, and delete confirmations.
 */

document.addEventListener('DOMContentLoaded', function () {
  // Category filter select change
  const categoryFilter = document.getElementById('categoryFilter');
  if (categoryFilter) {
    categoryFilter.addEventListener('change', function () {
      this.form.submit();
    });
  }

  // Delete single record confirmation
  const deleteForms = document.querySelectorAll('.delete-form');
  deleteForms.forEach(form => {
    form.addEventListener('submit', function (e) {
      const confirmed = confirm('Are you sure you want to delete this traffic sign scan record? This cannot be undone.');
      if (!confirmed) {
        e.preventDefault();
      }
    });
  });

  // Clear all history confirmation
  const clearAllForm = document.getElementById('clearAllHistoryForm');
  if (clearAllForm) {
    clearAllForm.addEventListener('submit', function (e) {
      const confirmed = confirm('WARNING: Are you sure you want to clear the ENTIRE detection history? All records will be removed from the database.');
      if (!confirmed) {
        e.preventDefault();
      }
    });
  }
});
